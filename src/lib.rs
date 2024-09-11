use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use chrono::Local;
use log::{error, info, LevelFilter};
use tokio::sync::broadcast::{Sender};
use tokio::sync::Mutex;

pub use crate::app::run_console;
use crate::benchmark::{BenchmarkReportWriter, Event};
pub use crate::benchmark::{BenchmarkConfig, BenchmarkKind};
use crate::requests::{OpenAITextGenerationBackend};

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
                 stop_sender: Sender<()>,
) {
    info!("Starting benchmark");
    let filepath = "data.json".to_string();
    // let backend = OpenAITextGenerationBackend::new("".to_string(), "http://10.90.11.68:8000".to_string());
    let backend = OpenAITextGenerationBackend::new("".to_string(), url);
    let requests = requests::ShareGPTTextRequestGenerator::new(filepath, tokenizer_name, 50, 10, 10, 10);

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
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
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
    let mut stop_receiver = stop_sender.subscribe();
    let stop_sender_clone = stop_sender.clone();
    let ui_thread = tokio::spawn(async move {
        tokio::select! {
            _ = stop_receiver.recv() => {
                error!("Received stop signal, stopping benchmark");
            }
            _ = async{
                if interactive {
                    run_console(config_clone, rx, stop_sender_clone).await;
                } else {
                    // consume the channel to avoid closed channel error
                    while let Some(_) = rx.recv().await {}
                }
            } => {}
        }
    });
    let mut benchmark = benchmark::Benchmark::new("benchmark".to_string(), config, Box::new(backend), Arc::from(Mutex::from(requests)), tx.clone(), stop_sender.clone());
    let mut stop_receiver = stop_sender.subscribe();
    tokio::select! {
        results = benchmark.run() => {
            let results = match results {
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
        }
        _ = stop_receiver.recv() => {
            error!("Received stop signal, stopping benchmark");
        }
    }
    let _ = tx.send(Event::BenchmarkReportEnd);
    ui_thread.await.unwrap();
}