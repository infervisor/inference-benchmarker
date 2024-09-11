use std::sync::{Arc};
use std::time::Duration;
use log::{debug, info};
use serde::Serialize;
use tokio::fs;
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio::sync::mpsc::{Receiver, Sender};
use crate::requests::{TextGenerationBackend, TextRequestGenerator};
use crate::{executors, scheduler};
use crate::results::{BenchmarkReport, BenchmarkResults};
use crate::scheduler::{ExecutorType, SchedulerProgress};

#[derive(Clone, Debug, strum_macros::Display)]
pub enum BenchmarkKind {
    Throughput,
    Sweep,
    Optimum,
}

pub struct EventMessage {
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: log::Level,
}

pub struct BenchmarkEvent {
    pub id: String,
    pub scheduler_type: ExecutorType,
    pub request_throughput: Option<f64>,
    pub progress: f64,
    pub results: Option<BenchmarkResults>,
}

pub enum Event {
    BenchmarkStart(BenchmarkEvent),
    BenchmarkProgress(BenchmarkEvent),
    BenchmarkEnd(BenchmarkEvent),
    Message(EventMessage),
    BenchmarkReportEnd
}

pub struct Benchmark {
    start_time: Option<std::time::Instant>,
    end_time: Option<std::time::Instant>,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    report: BenchmarkReport,
    config: BenchmarkConfig,
    event_bus: mpsc::UnboundedSender<Event>,
    stop_sender: broadcast::Sender<()>,
}

#[derive(Clone)]
pub struct BenchmarkConfig {
    pub max_vus: u64,
    pub duration: Duration,
    pub benchmark_kind: BenchmarkKind,
    pub warmup_duration: Duration,
    pub rate: Option<f64>,
}

pub struct BenchmarkProgress {
    id: String,
    progress: SchedulerProgress,
}

impl Benchmark {
    pub fn new(config: BenchmarkConfig, backend: Box<dyn TextGenerationBackend + Send + Sync>, requests: Arc<Mutex<dyn TextRequestGenerator + Send>>, event_bus: mpsc::UnboundedSender<Event>, stop_sender: broadcast::Sender<()>) -> Benchmark {
        Benchmark {
            start_time: None,
            end_time: None,
            report: BenchmarkReport::new(),
            config: config.clone(),
            backend,
            requests,
            event_bus,
            stop_sender,
        }
    }

    pub fn get_report(&self) -> BenchmarkReport {
        self.report.clone()
    }

    pub async fn run(&mut self) -> anyhow::Result<BenchmarkReport> {
        self.start_time = Some(std::time::Instant::now());
        info!("Prewarming backend");
        self.warmup().await?;
        info!("Prewarm complete");
        match self.config.benchmark_kind {
            BenchmarkKind::Throughput => {
                self.run_throughput().await?;
            }
            BenchmarkKind::Sweep => {
                self.run_sweep().await?;
            }
            BenchmarkKind::Optimum => {
                todo!()
            }
        }
        self.end_time = Some(std::time::Instant::now());
        self.event_bus.send(Event::Message(EventMessage {
            message: format!("Benchmark complete in {:?}", self.duration().expect("duration exists")),
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;
        Ok(self.report.clone())
    }

    pub fn duration(&self) -> Option<std::time::Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    async fn handle_progress(&self, id: String) -> Sender<Option<SchedulerProgress>> {
        let (tx, mut rx): (Sender<Option<SchedulerProgress>>, Receiver<Option<SchedulerProgress>>) = mpsc::channel(8);
        let event_bus = self.event_bus.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    None => {
                        break;
                    }
                    Some(progress) => {
                        let progress_evt = BenchmarkProgress {
                            id: id.clone(),
                            progress,
                        };
                        event_bus.send(Event::BenchmarkProgress(BenchmarkEvent {
                            id: progress_evt.id,
                            scheduler_type: ExecutorType::ConstantVUs,
                            request_throughput: Some(progress_evt.progress.requests_throughput),
                            progress: progress_evt.progress.progress,
                            results: None,
                        })).unwrap();
                    }
                }
            }
        });
        tx
    }

    pub async fn warmup(&mut self) -> anyhow::Result<()> {
        // run a warmup benchmark to prewarm the server

        let id = "warmup".to_string();

        // notify start event
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: id.to_string(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: None,
            progress: 0.0,
            results: None,
        }))?;

        // create progress handler
        let tx = self.handle_progress(id.clone()).await;

        // start scheduler
        let mut scheduler = scheduler::Scheduler::new(id, self.backend.clone(), ExecutorType::ConstantVUs, executors::ExecutorConfig {
            max_vus: 1,
            duration: self.config.warmup_duration,
            rate: None,
        }, self.requests.clone(), tx.clone(), self.stop_sender.clone());
        scheduler.run().await?;

        let results = scheduler.get_results().lock().await.clone();
        self.report.add_benchmark_result(results.clone());

        // send None to close the progress handler
        tx.send(None).await.unwrap();

        // notify end event
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: "warmup".to_string(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: results.successful_request_rate().ok(),
            progress: 100.0,
            results: None,
        }))?;
        Ok(())
    }

    pub async fn run_throughput(&mut self) -> anyhow::Result<()> {
        info!("Running throughput benchmark");

        let id = "throughput".to_string();

        // notify start event
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: None,
            progress: 0.0,
            results: None,
        }))?;

        // create progress handler
        let tx = self.handle_progress(id.clone()).await;

        // start scheduler
        let mut scheduler = scheduler::Scheduler::new(id.clone(), self.backend.clone(), ExecutorType::ConstantVUs, executors::ExecutorConfig {
            max_vus: self.config.max_vus,
            duration: self.config.duration,
            rate: None,
        }, self.requests.clone(), tx.clone(), self.stop_sender.clone());
        scheduler.run().await?;
        let results = scheduler.get_results().lock().await.clone();
        let rate = results.successful_request_rate().ok();
        self.report.add_benchmark_result(results.clone());

        // send None to close the progress handler
        tx.send(None).await.unwrap();

        // notify end event
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: rate,
            progress: 100.0,
            results: Some(results.clone()),
        }))?;
        Ok(())
    }

    pub async fn run_sweep(&mut self) -> anyhow::Result<()> {
        // run a throughput benchmark to retrieve the maximum throughput of server
        self.run_throughput().await?;
        // get the max throughput from the second benchmark result (first is warmup)
        let max_throughput = self.report.get_results()[1].successful_request_rate()?;
        // run a sweep benchmark for 10 different rates from 1req/s to computed max throughput
        // notify event bus
        self.event_bus.send(Event::Message(EventMessage {
            message: format!("Max throughput detected at: {:.2} req/s", max_throughput),
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;
        let mut rates = Vec::new();
        let num_rates = 5;
        for i in 1..=num_rates {
            rates.push(i as f64 * max_throughput / num_rates as f64);
        }
        for rate in rates {
            debug!("Running sweep benchmark with rate: {} req/s", rate);

            let id = format!("constant@{:.2}req/s", rate);

            // notify start event
            self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
                id: id.clone(),
                scheduler_type: ExecutorType::ConstantArrivalRate,
                request_throughput: None,
                progress: 0.0,
                results: None,
            }))?;

            // create progress handler
            let tx = self.handle_progress(id.clone()).await;

            // start scheduler
            let mut scheduler = scheduler::Scheduler::new(id, self.backend.clone(), scheduler::ExecutorType::ConstantArrivalRate, executors::ExecutorConfig {
                max_vus: self.config.max_vus,
                duration: self.config.duration,
                rate: Some(rate),
            }, self.requests.clone(), tx.clone(), self.stop_sender.clone());
            scheduler.run().await?;
            let results = scheduler.get_results().lock().await.clone();
            self.report.add_benchmark_result(results.clone());

            // send None to close the progress handler
            tx.send(None).await.unwrap();

            // notify end event
            self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
                id: format!("constant@{:.2}req/s", rate),
                scheduler_type: ExecutorType::ConstantArrivalRate,
                request_throughput: results.successful_request_rate().ok(),
                progress: 100.0,
                results: Some(results.clone()),
            }))?;
        }
        Ok(())
    }
}


#[derive(Serialize)]
pub struct BenchmarkResultsWriter {
    id: String,
    executor_type: String,
    config: executors::ExecutorConfig,
    total_requests: u64,
    total_tokens: u64,
    token_throughput_secs: f64,
    duration_ms: u128,
    time_to_first_token_ms_avg: u128,
    time_to_first_token_ms_p90: u128,
    time_to_first_token_ms_p95: u128,
    inter_token_latency_ms_avg: u128,
    inter_token_latency_ms_p90: u128,
    inter_token_latency_ms_p95: u128,
    failed_requests: u64,
    successful_requests: u64,
    request_rate: f64,
}

impl BenchmarkResultsWriter {
    pub fn new(results: BenchmarkResults) -> anyhow::Result<BenchmarkResultsWriter> {
        Ok(BenchmarkResultsWriter {
            id: results.id.clone(),
            executor_type: results.executor_type().to_string(),
            config: results.executor_config(),
            total_requests: results.total_requests() as u64,
            total_tokens: results.total_tokens() as u64,
            token_throughput_secs: results.token_throughput_secs()?,
            duration_ms: results.duration().ok().unwrap().as_millis(),
            time_to_first_token_ms_avg: results.time_to_first_token_avg().ok().unwrap().as_millis(),
            time_to_first_token_ms_p90: results.time_to_first_token_percentile(0.9)?.as_millis(),
            time_to_first_token_ms_p95: results.time_to_first_token_percentile(0.95)?.as_millis(),
            inter_token_latency_ms_avg: results.inter_token_latency_avg().ok().unwrap().as_millis(),
            inter_token_latency_ms_p90: results.inter_token_latency_percentile(0.9)?.as_millis(),
            inter_token_latency_ms_p95: results.inter_token_latency_percentile(0.95)?.as_millis(),
            failed_requests: results.failed_requests() as u64,
            successful_requests: results.successful_requests() as u64,
            request_rate: results.successful_request_rate()?,
        })
    }
}

pub struct BenchmarkReportWriter {}


impl BenchmarkReportWriter {
    pub async fn json(report: BenchmarkReport, path: &str) -> anyhow::Result<()> {
        // write the benchmark report to json
        let mut results: Vec<BenchmarkResultsWriter> = Vec::new();
        for result in report.get_results() {
            let writer = BenchmarkResultsWriter::new(result)?;
            results.push(writer);
        }
        let report = serde_json::to_string(&results).unwrap();
        let path = path.to_string();
        fs::write(path, report).await?;
        Ok(())
    }
}