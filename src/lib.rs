use futures_util::StreamExt;
use log::info;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;
use crate::load_generator::Executor;
use crate::requests::{OpenAITextGenerationBackend, TextGenerationRequest, TextGenerationResponse};

mod requests;
mod load_generator;
mod tokens;

pub async fn run() {
    info!("Starting benchmark");
    let filepath = "data.json".to_string();
    let backend = OpenAITextGenerationBackend::new("".to_string(), "http://10.90.11.68:8000".to_string());
    let executor = load_generator::ThroughputExecutor::new(Box::new(backend), 2, std::time::Duration::from_secs(10));
    let requests = requests::ShareGPTTextRequestGenerator::new(filepath, "gpt2".to_string(), 50, 10, 100, 10);
    let (tx, rx): (UnboundedSender<TextGenerationResponse>, UnboundedReceiver<TextGenerationResponse>) = tokio::sync::mpsc::unbounded_channel();
    let rx = UnboundedReceiverStream::new(rx);
    tokio::spawn(async move {
        rx.for_each(|response| async move {
            info!("Received response: {:?}", response);
        }).await;
    });
    executor.run(Box::new(requests), tx).await;
}