use std::sync::{Arc};
use futures_util::StreamExt;
use log::info;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use crate::executors::Executor;
use crate::requests::{OpenAITextGenerationBackend, TextGenerationAggregatedResponse, TextGenerationRequest, TextGenerationResponse};

mod requests;
mod executors;
mod tokens;
mod scheduler;
mod results;

pub async fn run() {
    info!("Starting benchmark");
    let filepath = "data.json".to_string();
    let backend = OpenAITextGenerationBackend::new("".to_string(), "http://10.90.11.68:8000".to_string());
    //let executor = executors::ThroughputExecutor::new(Box::new(backend), 2, std::time::Duration::from_secs(10));
    let requests= requests::ShareGPTTextRequestGenerator::new(filepath, "gpt2".to_string(), 50, 10, 100, 10);
    let scheduler= scheduler::Scheduler::new(Box::new(backend), scheduler::ExecutorType::Throughput, executors::ExecutorConfig {
        max_vus: 1,
        duration: std::time::Duration::from_secs(10),
    }, Arc::from(Mutex::from(requests)));
    scheduler.run().await;
    // let (tx, rx): (UnboundedSender<TextGenerationAggregatedResponse>, UnboundedReceiver<TextGenerationAggregatedResponse>) = tokio::sync::mpsc::unbounded_channel();
    // let rx = UnboundedReceiverStream::new(rx);
    // tokio::spawn(async move {
    //     rx.for_each(|response| async move {
    //         info!("Received response: {:?}", response);
    //     }).await;
    // });
    // executor.run(Arc::from(Mutex::from(requests)), tx).await;
}