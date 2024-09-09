use std::sync::{Arc};
use futures_util::StreamExt;
use log::info;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use crate::benchmark::{BenchmarkConfig, BenchmarkReportWriter, BenchmarkResultsWriter};
use crate::executors::Executor;
use crate::requests::{OpenAITextGenerationBackend, TextGenerationAggregatedResponse, TextGenerationRequest, TextGenerationResponse};

mod requests;
mod executors;
mod tokens;
mod scheduler;
mod results;
mod benchmark;

pub async fn run() {
    info!("Starting benchmark");
    let filepath = "data.json".to_string();
    let backend = OpenAITextGenerationBackend::new("".to_string(), "http://10.90.11.68:8000".to_string());
    let requests = requests::ShareGPTTextRequestGenerator::new(filepath, "gpt2".to_string(), 50, 10, 10, 10);

    // // Throughput executor
    // let scheduler = scheduler::Scheduler::new(Box::new(backend), scheduler::ExecutorType::Throughput, executors::ExecutorConfig {
    //     max_vus: 1,
    //     duration: std::time::Duration::from_secs(10),
    //     rate: None,
    // }, Arc::from(Mutex::from(requests)));
    // scheduler.run().await;

    // // Constant arrival rate executor
    // let scheduler = scheduler::Scheduler::new(Box::new(backend.clone()), scheduler::ExecutorType::ConstantArrivalRate, executors::ExecutorConfig {
    //     max_vus: 10,
    //     duration: std::time::Duration::from_secs(10),
    //     rate: Some(1),
    // }, Arc::from(Mutex::from(requests.clone())));
    // scheduler.run().await;
    let config = BenchmarkConfig {
        max_vus: 10,
        duration: std::time::Duration::from_secs(10),
        benchmark_kind: benchmark::BenchmarkKind::Sweep,
        prewarm_duration: std::time::Duration::from_secs(5),
    };
    let mut benchmark = benchmark::Benchmark::new("benchmark".to_string(), config, Box::new(backend), Arc::from(Mutex::from(requests)));
    let results = match benchmark.run().await {
        Ok(results) => results.get_results(),
        Err(e) => {
            info!("Error running benchmark: {:?}", e);
            return;
        }
    };
    info!("Throughput is {requests_throughput} req/s",requests_throughput = results[0].request_rate().unwrap());
    let report = benchmark.get_report();
    let path = "results.json".to_string();
    BenchmarkReportWriter::json(report, &path).await.unwrap();
}