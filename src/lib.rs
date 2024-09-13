use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use chrono::Local;
use log::{error, info, Level, LevelFilter};
use tokio::sync::broadcast::{Sender};
use tokio::sync::Mutex;

pub use crate::app::run_console;
use crate::benchmark::{BenchmarkReportWriter, Event, MessageEvent};
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
mod flux;

pub async fn run(url: String,
                 tokenizer_name: String,
                 prompt_length: u64,
                 prompt_variance: u64,
                 decode_length: u64,
                 max_vus: u64,
                 duration: std::time::Duration,
                 rate: Option<f64>,
                 num_rates: u64,
                 benchmark_kind: String,
                 prewarm_duration: std::time::Duration,
                 interactive: bool,
                 stop_sender: Sender<()>,
) -> anyhow::Result<()> {
    info!("Starting benchmark");
    // set process system limits
    sysinfo::set_open_files_limit(0);
    // let backend = OpenAITextGenerationBackend::new("".to_string(), "http://10.90.11.68:8000".to_string());
    let backend = OpenAITextGenerationBackend::new("".to_string(), url, tokenizer_name.clone());

    let config = BenchmarkConfig {
        max_vus,
        duration,
        benchmark_kind: match benchmark_kind.to_lowercase().as_str() {
            "throughput" => BenchmarkKind::Throughput,
            "sweep" => BenchmarkKind::Sweep,
            "rate" => BenchmarkKind::Rate,
            _ => BenchmarkKind::Sweep,
        },
        warmup_duration: prewarm_duration,
        rate,
        num_rates,
        prompt_length,
        prompt_variance,
        decode_length,
    };
    config.validate()?;
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

    // download prompts dataset
    info!("Downloading dataset");
    let _ = tx.send(Event::Message(MessageEvent {
        message: "Downloading dataset".to_string(),
        timestamp: chrono::Utc::now(),
        level: Level::Info,
    }));
    let filepath = requests::ShareGPTTextRequestGenerator::download_dataset("hlarcher/share_gpt_small".to_string(), "share_gpt_filtered_small.json".to_string()).expect("Can't download dataset");
    let requests = requests::ShareGPTTextRequestGenerator::new(filepath, tokenizer_name, prompt_length, 1, prompt_length * 2, prompt_variance);

    let mut benchmark = benchmark::Benchmark::new(config.clone(), Box::new(backend), Arc::from(Mutex::from(requests)), tx.clone(), stop_sender.clone());
    let mut stop_receiver = stop_sender.subscribe();
    tokio::select! {
        report = benchmark.run() => {
            match report {
                Ok(results) => {
                    info!("Throughput is {requests_throughput} req/s",requests_throughput = results.get_results()[0].successful_request_rate().unwrap());
                    let report = benchmark.get_report();
                    let path = "results/".to_string();
                    let writer=BenchmarkReportWriter::new(config.clone(), report)?;
                    writer.json(&path).await?;
                },
                Err(e) => {
                    error!("Error running benchmark: {:?}", e.to_string());
                    let _ = tx.send(Event::BenchmarkError(e.to_string()));
                }
            };
        }
        _ = stop_receiver.recv() => {
            error!("Received stop signal, stopping benchmark");
        }
    }
    let _ = tx.send(Event::BenchmarkReportEnd);
    ui_thread.await.unwrap();
    Ok(())
}
