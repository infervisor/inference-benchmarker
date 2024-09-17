use std::sync::{Arc};
use std::time::Duration;
use log::{debug, info};
use serde::Serialize;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, System};
use tokio::fs;
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio::sync::mpsc::{Receiver, Sender};
use crate::requests::{TextGenerationBackend, TextRequestGenerator, TokenizeOptions};
use crate::{executors, scheduler};
use crate::results::{BenchmarkReport, BenchmarkResults};
use crate::scheduler::{ExecutorType, SchedulerProgress};

#[derive(Clone, Debug, strum_macros::Display, Serialize)]
pub enum BenchmarkKind {
    Throughput,
    Sweep,
    Rate,
}

pub struct MessageEvent {
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
    pub successful_requests: u64,
    pub failed_requests: u64,
}

pub enum Event {
    BenchmarkStart(BenchmarkEvent),
    BenchmarkProgress(BenchmarkEvent),
    BenchmarkEnd(BenchmarkEvent),
    Message(MessageEvent),
    BenchmarkReportEnd,
    BenchmarkError(String),
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

#[serde_with::serde_as]
#[derive(Clone, Serialize)]
pub struct BenchmarkConfig {
    pub max_vus: u64,
    #[serde(rename = "duration_secs")]
    #[serde_as(as = "serde_with::DurationSeconds<u64>")]
    pub duration: Duration,
    pub benchmark_kind: BenchmarkKind,
    #[serde(rename = "warmup_duration_secs")]
    #[serde_as(as = "serde_with::DurationSeconds<u64>")]
    pub warmup_duration: Duration,
    pub rate: Option<f64>,
    pub num_rates: u64,
    pub prompt_options: Option<TokenizeOptions>,
    pub decode_options: Option<TokenizeOptions>,
}

impl BenchmarkConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.max_vus <= 0 {
            return Err(anyhow::anyhow!("max_vus must be greater than 0"));
        }
        if self.duration.as_secs() <= 0 {
            return Err(anyhow::anyhow!("duration must be greater than 0"));
        }
        if self.warmup_duration.as_secs() <= 0 {
            return Err(anyhow::anyhow!("warmup_duration must be greater than 0"));
        }
        match self.benchmark_kind {
            BenchmarkKind::Throughput => {
                if self.rate.is_some() {
                    return Err(anyhow::anyhow!("rate must not be specified for throughput benchmark"));
                }
            }
            BenchmarkKind::Sweep => {
                if self.rate.is_some() {
                    return Err(anyhow::anyhow!("rate must not be specified for sweep benchmark"));
                }
            }
            BenchmarkKind::Rate => {
                if self.rate.is_none() {
                    return Err(anyhow::anyhow!("rate must be specified for sweep benchmark"));
                }
            }
        }
        Ok(())
    }
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
        self.report.start();
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
            BenchmarkKind::Rate => {
                self.run_rate(self.config.rate.expect("config already validated")).await?;
            }
        }
        self.end_time = Some(std::time::Instant::now());
        self.event_bus.send(Event::Message(MessageEvent {
            message: format!("Benchmark complete in {:?}", self.duration().expect("duration exists")),
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;
        self.report.end();
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
                            successful_requests: progress_evt.progress.successful_requests,
                            failed_requests: progress_evt.progress.failed_requests,
                            results: None,
                        })).unwrap(); //FIXME remove unwrap
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
            successful_requests: 0,
            failed_requests: 0,
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
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
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
            successful_requests: 0,
            failed_requests: 0,
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
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
        }))?;
        Ok(())
    }

    pub async fn run_sweep(&mut self) -> anyhow::Result<()> {
        // run a throughput benchmark to retrieve the maximum throughput of server
        self.run_throughput().await?;
        // get the max throughput from the second benchmark result (first is warmup)
        let throughput_results = &self.report.get_results()[1];
        let max_throughput = throughput_results.successful_request_rate()?;
        let max_tokens_throughput = throughput_results.token_throughput_secs()?;
        // run a sweep benchmark for 10 different rates from 1req/s to computed max throughput
        // notify event bus
        self.event_bus.send(Event::Message(MessageEvent {
            message: format!("Max throughput detected at: {:.2} req/s | {:.2} tokens/s", max_throughput, max_tokens_throughput),
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;
        let mut rates = Vec::new();
        let num_rates = self.config.num_rates;
        for i in 1..=num_rates {
            rates.push(i as f64 * max_throughput / num_rates as f64);
        }
        for rate in rates {
            self.run_rate(rate).await?;
        }
        Ok(())
    }

    pub async fn run_rate(&mut self, rate: f64) -> anyhow::Result<()> {
        debug!("Running benchmark with rate: {} req/s", rate);

        let id = format!("constant@{:.2}req/s", rate);

        // notify start event
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantArrivalRate,
            request_throughput: None,
            progress: 0.0,
            results: None,
            successful_requests: 0,
            failed_requests: 0,
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
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
        }))?;
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
    total_tokens_sent: u64,
    e2e_latency_ms_avg: u128,
    e2e_latency_ms_p90: u128,
    e2e_latency_ms_p95: u128,
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
            total_tokens_sent: results.total_tokens_sent(),
            e2e_latency_ms_avg: results.e2e_latency_avg().ok().unwrap().as_millis(),
            e2e_latency_ms_p90: results.e2e_latency_percentile(0.9)?.as_millis(),
            e2e_latency_ms_p95: results.e2e_latency_percentile(0.95)?.as_millis(),
        })
    }
}

#[derive(Serialize)]
pub struct SystemInfo {
    pub cpu: Vec<String>,
    pub memory: String,
    pub os_name: String,
    pub os_version: String,
    pub kernel: String,
    pub hostname: String,
}

impl SystemInfo {
    pub fn new() -> SystemInfo {
        let s = System::new_with_specifics(
            sysinfo::RefreshKind::new()
                .with_memory(MemoryRefreshKind::everything())
                .with_cpu(CpuRefreshKind::everything())
        );
        let cpu_info = s.cpus().iter().map(|cpu| format!("{} {}@{:.0}MHz", cpu.brand(), cpu.name(), cpu.frequency())).collect::<Vec<String>>();
        SystemInfo {
            cpu: cpu_info,
            memory: format!("{:.2} GB", s.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0),
            os_name: System::name().ok_or("N/A").unwrap(),
            os_version: System::os_version().ok_or("N/A").unwrap(),
            kernel: System::kernel_version().ok_or("N/A").unwrap(),
            hostname: System::host_name().ok_or("N/A").unwrap(),
        }
    }
}

#[derive(Serialize)]
pub struct BenchmarkReportWriter {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResultsWriter>,
    start_time: String,
    end_time: String,
    system: SystemInfo,
}


impl BenchmarkReportWriter {
    pub fn new(config: BenchmarkConfig, report: BenchmarkReport) -> anyhow::Result<BenchmarkReportWriter> {
        let mut results: Vec<BenchmarkResultsWriter> = Vec::new();
        for result in report.get_results() {
            let writer = BenchmarkResultsWriter::new(result)?;
            results.push(writer);
        }
        Ok(BenchmarkReportWriter {
            config,
            results,
            start_time: report.start_time().ok_or(anyhow::anyhow!("start_time not set"))?.to_rfc3339(),
            end_time: report.end_time().ok_or(anyhow::anyhow!("end_time not set"))?.to_rfc3339(),
            system: SystemInfo::new(),
        })
    }
    pub async fn json(&self, path: &str) -> anyhow::Result<()> {
        // write the benchmark report to json
        let report = serde_json::to_string(&self).unwrap();
        let path = path.to_string();
        // create path
        if !std::path::Path::new(&path).exists() {
            fs::create_dir_all(&path).await?;
        }
        fs::write(format!("{}/results.json", path), report).await?;
        Ok(())
    }
}