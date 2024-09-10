use std::fs::File;
use std::io::Write;
use std::sync::{Arc};
use chrono::Local;
use futures_util::StreamExt;
use log::{error, info, Level, LevelFilter};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use crate::benchmark::{BenchmarkReportWriter, BenchmarkResultsWriter, Event, EventMessage};
use crate::executors::Executor;
use crate::requests::{OpenAITextGenerationBackend, TextGenerationAggregatedResponse, TextGenerationRequest, TextGenerationResponse};
pub use crate::benchmark::{BenchmarkKind, BenchmarkConfig};
pub use crate::app::run_console;

mod requests;
mod executors;
mod tokens;
mod scheduler;
mod results;
mod benchmark;
mod app;
mod event;

pub async fn run(url: String,
                 tokenizer_name: String,
                 max_vus: u64,
                 duration: std::time::Duration,
                 rate: Option<f64>,
                 benchmark_kind: String,
                 prewarm_duration: std::time::Duration,
                 interactive: bool,
) {
    info!("Starting benchmark");
    let filepath = "data.json".to_string();
    // let backend = OpenAITextGenerationBackend::new("".to_string(), "http://10.90.11.68:8000".to_string());
    let backend = OpenAITextGenerationBackend::new("".to_string(), url);
    let requests = requests::ShareGPTTextRequestGenerator::new(filepath, tokenizer_name, 50, 10, 10, 10);

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
        max_vus,
        duration,
        benchmark_kind: match benchmark_kind.as_str() {
            "Throughput" => BenchmarkKind::Throughput,
            "Sweep" => BenchmarkKind::Sweep,
            "Optimum" => BenchmarkKind::Optimum,
            _ => BenchmarkKind::Sweep,
        },
        warmup_duration: prewarm_duration,
        rate,
    };
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    if interactive {
        // send logs to file
        let target = Box::new(File::create("log.txt").expect("Can't create file"));
        env_logger::Builder::new()
            .target(env_logger::Target::Pipe(target))
            .filter(Some("text_generation_inference_benchmark"), LevelFilter::Debug)
            .format(|buf, record| {
                writeln!(
                    buf,
                    "[{} {} {}:{}] {}",
                    Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                    record.level(),
                    record.file().unwrap_or("unknown"),
                    record.line().unwrap_or(0),
                    record.args()
                )
            })
            .init();
    } else {
        env_logger::init();
    }
    let config_clone = config.clone();
    let ui_thread = tokio::spawn(async move {
        if interactive {
            run_console(config_clone, rx).await;
        }
    });
    let mut benchmark = benchmark::Benchmark::new("benchmark".to_string(), config, Box::new(backend), Arc::from(Mutex::from(requests)), tx.clone());
    let results = match benchmark.run().await {
        Ok(results) => results.get_results(),
        Err(e) => {
            error!("Error running benchmark: {:?}", e.to_string());
            return;
        }
    };
    info!("Throughput is {requests_throughput} req/s",requests_throughput = results[0].successful_request_rate().unwrap());
    let report = benchmark.get_report();
    let path = "results.json".to_string();
    BenchmarkReportWriter::json(report, &path).await.unwrap();
    let _ = tx.send(Event::Message(EventMessage {
        message: "Benchmark finished".to_string(),
        timestamp: chrono::Utc::now(),
        level: Level::Info,
    }));
    ui_thread.await.unwrap();
}