use std::sync::Arc;
use log::{info, trace, warn};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;
use crate::executors::{Executor, ExecutorConfig, ThroughputExecutor};
use crate::requests;
use crate::requests::{TextGenerationAggregatedResponse, TextGenerationBackend, TextRequestGenerator};
use crate::results::BenchmarkResult;
use futures_util::StreamExt;
use tokio::sync::Mutex;

pub(crate) enum ExecutorType {
    Throughput,
    ConstantArrivalRate,
}

pub(crate) struct Scheduler {
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    executor: Arc<dyn Executor>,
    requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>,
}

impl Scheduler {
    pub(crate) fn new(backend: Box<dyn TextGenerationBackend + Send + Sync>, executor_type: ExecutorType, config: ExecutorConfig, requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>) -> Scheduler {
        match executor_type {
            ExecutorType::Throughput => {
                return Scheduler {
                    backend: backend.clone(),
                    executor: Arc::new(ThroughputExecutor::new(backend.clone(), config.max_vus.clone(), config.duration.clone())),
                    requests_generator: requests_generator,
                };
            }
            ExecutorType::ConstantArrivalRate => {
                unimplemented!()
            }
        }
    }

    pub(crate) async fn run(&self) {
        let mut bench_result = Arc::from(Mutex::from(BenchmarkResult::new()));
        let (tx, rx): (UnboundedSender<TextGenerationAggregatedResponse>, UnboundedReceiver<TextGenerationAggregatedResponse>) = tokio::sync::mpsc::unbounded_channel();
        let rx = UnboundedReceiverStream::new(rx);
        let b= bench_result.clone();
        tokio::spawn(async move {
            rx.for_each(|response| {
                let result = bench_result.clone();
                async move {
                    trace!("Received response: {:?}", response);
                    result.lock().await.add_response(response);
                }
            }).await;
        });
        self.executor.run(self.requests_generator.clone(), tx).await;
        warn!("{:?}", b);
    }
}