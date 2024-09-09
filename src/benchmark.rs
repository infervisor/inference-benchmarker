use std::sync::{Arc};
use std::time::Duration;
use log::{debug, info};
use reqwest::Request;
use serde::Serialize;
use tokio::fs;
use tokio::sync::Mutex;
use crate::requests::{TextGenerationBackend, TextRequestGenerator};
use crate::{executors, scheduler};
use crate::results::{BenchmarkReport, BenchmarkResults};
use crate::scheduler::ExecutorType;

pub(crate) enum BenchmarkKind {
    Throughput,
    Sweep,
    Optimum,
}

pub(crate) struct Benchmark {
    id: String,
    start_time: Option<std::time::Instant>,
    end_time: Option<std::time::Instant>,
    max_vus: u64,
    kind: BenchmarkKind,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    report: BenchmarkReport,
    duration: Duration,
}

pub(crate) struct BenchmarkConfig {
    pub(crate) max_vus: u64,
    pub(crate) duration: std::time::Duration,
    pub(crate) benchmark_kind: BenchmarkKind,
}

impl Benchmark {
    pub(crate) fn new(id: String, config: BenchmarkConfig, backend: Box<dyn TextGenerationBackend + Send + Sync>, requests: Arc<Mutex<dyn TextRequestGenerator + Send>>) -> Benchmark {
        Benchmark {
            id,
            start_time: None,
            end_time: None,
            report: BenchmarkReport::new(),
            kind: config.benchmark_kind,
            max_vus: config.max_vus,
            duration: config.duration,
            backend,
            requests,
        }
    }

    pub(crate) fn get_report(&self) -> BenchmarkReport {
        self.report.clone()
    }

    pub(crate) async fn run(&mut self) -> anyhow::Result<BenchmarkReport> {
        self.start_time = Some(std::time::Instant::now());
        match self.kind {
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

    pub(crate) async fn run_throughput(&mut self) -> anyhow::Result<()> {
        info!("Running throughput benchmark");
        let scheduler = scheduler::Scheduler::new(self.backend.clone(), scheduler::ExecutorType::Throughput, executors::ExecutorConfig {
            max_vus: 1,
            duration: self.duration,
            rate: None,
        }, self.requests.clone());
        scheduler.run().await;
        let results = scheduler.results.lock().await.clone();
        self.report.add_benchmark_result(results);
        Ok(())
    }

    pub(crate) async fn run_sweep(&mut self) -> anyhow::Result<()> {
        // run a throughput benchmark to retrieve the maximum throughput of server
        let throughput_results = self.run_throughput().await?;
        //run a sweep benchmark for 10 different rates from 1req/s to computed max throughput
        let max_throughput = self.report.get_results()[0].request_rate()?;
        let mut rates = Vec::new();
        for i in 1..=3 {
            rates.push(i as f64 * max_throughput / 10.0);
        }
        for rate in rates {
            debug!("Running sweep benchmark with rate: {} req/s", rate);
            let scheduler = scheduler::Scheduler::new(self.backend.clone(), scheduler::ExecutorType::ConstantArrivalRate, executors::ExecutorConfig {
                max_vus: self.max_vus,
                duration: self.duration,
                rate: Some(rate),
            }, self.requests.clone());
            scheduler.run().await;
            let results = scheduler.results.lock().await.clone();
            self.report.add_benchmark_result(results);
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