use std::sync::{Arc};
use std::time::Duration;
use log::{debug, info};
use reqwest::Request;
use serde::Serialize;
use tokio::fs;
use tokio::sync::{mpsc, Mutex};
use crate::requests::{TextGenerationBackend, TextRequestGenerator};
use crate::{executors, scheduler};
use crate::results::{BenchmarkReport, BenchmarkResults};
use crate::scheduler::ExecutorType;

#[derive(Clone, Debug)]
pub enum BenchmarkKind {
    Throughput,
    Sweep,
    Optimum,
}

pub(crate) struct EventMessage {
    pub(crate) message: String,
    pub(crate) timestamp: chrono::DateTime<chrono::Utc>,
    pub(crate) level: log::Level,
}

pub(crate) struct BenchmarkEvent {
    pub(crate) id: String,
    pub(crate) scheduler_type: ExecutorType,
    pub(crate) request_throughput: Option<f64>,
    pub(crate) progress: f64,
}

pub(crate) enum Event {
    BenchmarkStart(BenchmarkEvent),
    BenchmarkProgress(BenchmarkEvent),
    BenchmarkEnd(BenchmarkEvent),
    Message(EventMessage),
}

pub(crate) struct Benchmark {
    id: String,
    start_time: Option<std::time::Instant>,
    end_time: Option<std::time::Instant>,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    report: BenchmarkReport,
    config: BenchmarkConfig,
    event_bus: mpsc::UnboundedSender<Event>,
}

#[derive(Clone)]
pub struct BenchmarkConfig {
    pub max_vus: u64,
    pub duration: Duration,
    pub benchmark_kind: BenchmarkKind,
    pub warmup_duration: Duration,
    pub rate: Option<f64>,
}

impl Benchmark {
    pub(crate) fn new(id: String, config: BenchmarkConfig, backend: Box<dyn TextGenerationBackend + Send + Sync>, requests: Arc<Mutex<dyn TextRequestGenerator + Send>>, event_bus: mpsc::UnboundedSender<Event>) -> Benchmark {
        Benchmark {
            id,
            start_time: None,
            end_time: None,
            report: BenchmarkReport::new(),
            config: config.clone(),
            backend,
            requests,
            event_bus,
        }
    }

    pub(crate) fn get_report(&self) -> BenchmarkReport {
        self.report.clone()
    }

    pub(crate) async fn run(&mut self) -> anyhow::Result<BenchmarkReport> {
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
        Ok(self.report.clone())
    }

    pub(crate) fn duration(&self) -> Option<std::time::Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    pub(crate) async fn warmup(&mut self) -> anyhow::Result<()> {
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: "warmup".to_string(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: None,
            progress: 0.0,
        }))?;
        let scheduler = scheduler::Scheduler::new(self.backend.clone(), ExecutorType::ConstantVUs, executors::ExecutorConfig {
            max_vus: 1,
            duration: self.config.warmup_duration,
            rate: None,
        }, self.requests.clone());
        scheduler.run().await;
        let results = scheduler.results.lock().await.clone();
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: "warmup".to_string(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: results.request_rate().ok(),
            progress: 100.0,
        }))?;
        Ok(())
    }

    pub(crate) async fn run_throughput(&mut self) -> anyhow::Result<()> {
        info!("Running throughput benchmark");
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: "throughput".to_string(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: None,
            progress: 0.0,
        }))?;
        let scheduler = scheduler::Scheduler::new(self.backend.clone(), ExecutorType::ConstantVUs, executors::ExecutorConfig {
            max_vus: 1,
            duration: self.config.duration,
            rate: None,
        }, self.requests.clone());
        scheduler.run().await;
        let results = scheduler.results.lock().await.clone();
        let rate=results.request_rate().ok();
        self.report.add_benchmark_result(results);
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: "throughput".to_string(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: rate,
            progress: 100.0,
        }))?;
        Ok(())
    }

    pub(crate) async fn run_sweep(&mut self) -> anyhow::Result<()> {
        // run a throughput benchmark to retrieve the maximum throughput of server
        let throughput_results = self.run_throughput().await?;
        //run a sweep benchmark for 10 different rates from 1req/s to computed max throughput
        let max_throughput = self.report.get_results()[0].request_rate()?;
        self.event_bus.send(Event::Message(EventMessage {
            message: format!("Max throughput detected at: {:.2} req/s", max_throughput),
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;
        let mut rates = Vec::new();
        for i in 1..=3 {
            rates.push(i as f64 * max_throughput / 10.0);
        }
        for rate in rates {
            debug!("Running sweep benchmark with rate: {} req/s", rate);
            self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
                id: format!("constant@{:.2}req/s", rate),
                scheduler_type: ExecutorType::ConstantArrivalRate,
                request_throughput: None,
                progress: 0.0,
            }))?;
            let scheduler = scheduler::Scheduler::new(self.backend.clone(), scheduler::ExecutorType::ConstantArrivalRate, executors::ExecutorConfig {
                max_vus: self.config.max_vus,
                duration: self.config.duration,
                rate: Some(rate),
            }, self.requests.clone());
            scheduler.run().await;
            let results = scheduler.results.lock().await.clone();
            self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
                id: format!("constant@{:.2}req/s", rate),
                scheduler_type: ExecutorType::ConstantArrivalRate,
                request_throughput: results.request_rate().ok(),
                progress: 100.0,
            }))?;
        }
        Ok(())
    }
}


#[derive(Serialize)]
pub(crate) struct BenchmarkResultsWriter {
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
    pub(crate) fn new(results: BenchmarkResults) -> anyhow::Result<BenchmarkResultsWriter> {
        Ok(BenchmarkResultsWriter {
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
            request_rate: results.request_rate()?,
        })
    }
}

pub(crate) struct BenchmarkReportWriter {}


impl BenchmarkReportWriter {
    pub(crate) async fn json(report: BenchmarkReport, path: &str) -> anyhow::Result<()> {
        // write the benchmark report to json
        let mut results: Vec<BenchmarkResultsWriter> = Vec::new();
        for result in report.get_results() {
            let writer = BenchmarkResultsWriter::new(result).unwrap();
            results.push(writer);
        }
        let report = serde_json::to_string(&results).unwrap();
        let path = path.to_string();
        fs::write(path, report).await?;
        Ok(())
    }
}