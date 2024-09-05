use log::info;
use crate::load_generator::Executor;
use crate::requests::OpenAITextGenerationBackend;

mod requests;
mod load_generator;
mod tokens;

pub async fn run() {
    info!("Starting benchmark");
    let filepath = "data.json".to_string();
    let backend = OpenAITextGenerationBackend::new("".to_string(), "http://localhost:8000".to_string());
    let executor = load_generator::ThroughputExecutor::new(2, std::time::Duration::from_secs(10));
    let requests = requests::ShareGPTTextRequestGenerator::new(filepath, "gpt2".to_string(), 50, 10, 100, 10);
    executor.run(Box::new(requests)).await;
}