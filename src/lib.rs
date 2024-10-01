use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use chrono::Local;
use log::{debug, error, info, Level, LevelFilter};
use tokio::sync::broadcast::Sender;
use tokio::sync::Mutex;
use writers::BenchmarkReportWriter;
pub use crate::app::run_console;
use crate::benchmark::{Event, MessageEvent};
pub use crate::benchmark::{BenchmarkConfig, BenchmarkKind};
use crate::requests::OpenAITextGenerationBackend;
pub use crate::requests::TokenizeOptions;

mod requests;
mod executors;
mod tokens;
mod scheduler;
mod results;
mod benchmark;
mod app;
mod event;
mod flux;
mod writers;

pub struct RunConfiguration {
    pub url: String,
    pub tokenizer_name: String,
    pub max_vus: u64,
    pub duration: std::time::Duration,
    pub rates: Option<Vec<f64>>,
    pub num_rates: u64,
    pub benchmark_kind: String,
    pub warmup_duration: std::time::Duration,
    pub interactive: bool,
    pub prompt_options: Option<TokenizeOptions>,
    pub decode_options: Option<TokenizeOptions>,
    pub dataset: String,
    pub dataset_file: String,
    pub hf_token: Option<String>,
    pub extra_metadata: Option<HashMap<String, String>>,
}

pub async fn run(run_config: RunConfiguration,
                 stop_sender: Sender<()>,
) -> anyhow::Result<()> {
    info!("Starting benchmark");
    // set process system limits
    sysinfo::set_open_files_limit(0);
    // let backend = OpenAITextGenerationBackend::new("".to_string(), "http://10.90.11.68:8000".to_string());
    let backend = OpenAITextGenerationBackend::new("".to_string(), run_config.url.clone(), run_config.tokenizer_name.clone());

    let config = BenchmarkConfig {
        max_vus: run_config.max_vus,
        duration: run_config.duration,
        benchmark_kind: match run_config.benchmark_kind.to_lowercase().as_str() {
            "throughput" => BenchmarkKind::Throughput,
            "sweep" => BenchmarkKind::Sweep,
            "rate" => BenchmarkKind::Rate,
            _ => BenchmarkKind::Sweep,
        },
        warmup_duration: run_config.warmup_duration,
        rates: run_config.rates,
        num_rates: run_config.num_rates,
        prompt_options: run_config.prompt_options.clone(),
        decode_options: run_config.decode_options.clone(),
        tokenizer: run_config.tokenizer_name.clone(),
        extra_metadata: run_config.extra_metadata.clone(),
    };
    config.validate()?;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    if run_config.interactive {
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
                debug!("Received stop signal, stopping benchmark");
            }
            _ = async{
                if run_config.interactive {
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
    let filepath = requests::ConversationTextRequestGenerator::download_dataset(run_config.dataset, run_config.dataset_file, run_config.hf_token.clone()).expect("Can't download dataset");
    let requests = requests::ConversationTextRequestGenerator::load(filepath, run_config.tokenizer_name.clone(), run_config.prompt_options, run_config.decode_options, run_config.hf_token)?;

    let mut benchmark = benchmark::Benchmark::new(config.clone(), Box::new(backend), Arc::from(Mutex::from(requests)), tx.clone(), stop_sender.clone());
    let mut stop_receiver = stop_sender.subscribe();
    tokio::select! {
        report = benchmark.run() => {
            match report {
                Ok(results) => {
                    info!("Throughput is {requests_throughput} req/s",requests_throughput = results.get_results()[0].successful_request_rate().unwrap());
                    let report = benchmark.get_report();
                    let path = format!("results/{}_{}.json",run_config.tokenizer_name.replace("/","_"), chrono::Utc::now().format("%Y-%m-%d-%H-%M-%S"));
                    let path=Path::new(&path);
                    let writer=BenchmarkReportWriter::new(config.clone(), report)?;
                    writer.json(path).await?;
                    info!("Report saved to {:?}",path);
                },
                Err(e) => {
                    error!("Error running benchmark: {:?}", e.to_string());
                    let _ = tx.send(Event::BenchmarkError(e.to_string()));
                }
            };
        }
        _ = stop_receiver.recv() => {
            debug!("Received stop signal, stopping benchmark");
        }
    }
    let _ = tx.send(Event::BenchmarkReportEnd);
    info!("Benchmark finished");
    if !run_config.interactive { // quit app if not interactive
        let _ = stop_sender.send(());
    }
    ui_thread.await.unwrap();
    Ok(())
}
