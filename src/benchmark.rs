use std::sync::{Arc};
use std::time::Duration;
use log::{debug, info};
use reqwest::Request;
use tokio::sync::Mutex;
use crate::requests::{TextGenerationBackend, TextRequestGenerator};
use crate::{executors, scheduler};
use crate::results::{BenchmarkReport, BenchmarkResults};

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

    pub(crate) async fn run(&mut self) -> anyhow::Result<BenchmarkReport> {
        self.start_time = Some(std::time::Instant::now());
        let mut results: BenchmarkResults;
        match self.kind {
            BenchmarkKind::Throughput => {
                self.run_throughput().await;
            }
            BenchmarkKind::Sweep => {
                self.run_sweep().await;
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
        for i in 1..=10 {
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