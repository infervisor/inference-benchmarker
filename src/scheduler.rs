use std::sync::Arc;
use log::{info, trace, warn};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;
use crate::executors::{ConstantArrivalRateExecutor, Executor, ExecutorConfig, ThroughputExecutor};
use crate::requests;
use crate::requests::{TextGenerationAggregatedResponse, TextGenerationBackend, TextRequestGenerator};
use crate::results::BenchmarkResults;
use futures_util::StreamExt;
use tokio::sync::Mutex;

#[derive(Clone,strum_macros::Display)]
pub(crate) enum ExecutorType {
    Throughput,
    ConstantArrivalRate,
}

pub(crate) struct Scheduler {
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    executor: Arc<dyn Executor>,
    requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    pub(crate) results: Arc<Mutex<BenchmarkResults>>,
}

impl Scheduler {
    pub(crate) fn new(backend: Box<dyn TextGenerationBackend + Send + Sync>, executor_type: ExecutorType, config: ExecutorConfig, requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>) -> Scheduler {
        match executor_type {
            ExecutorType::Throughput => {
                return Scheduler {
                    backend: backend.clone(),
                    executor: Arc::new(ThroughputExecutor::new(backend.clone(), config.max_vus.clone(), config.duration.clone())),
                    results: Arc::from(Mutex::from(BenchmarkResults::new(ExecutorType::Throughput, config))),
                    requests_generator,
                };
            }
            ExecutorType::ConstantArrivalRate => {
                if config.rate.is_none() {
                    panic!("Rate must be specified for ConstantArrivalRateExecutor");
                }
                let rate = config.rate.unwrap();
                return Scheduler {
                    backend: backend.clone(),
                    executor: Arc::new(ConstantArrivalRateExecutor::new(backend.clone(), config.max_vus.clone(), config.duration.clone(), rate)),
                    results: Arc::from(Mutex::from(BenchmarkResults::new(ExecutorType::ConstantArrivalRate, config))),
                    requests_generator,
                };
            }
        }
    }

    pub(crate) async fn run(&self) {
        // add responses to the benchmark result as they arrive
        let (tx, rx): (UnboundedSender<TextGenerationAggregatedResponse>, UnboundedReceiver<TextGenerationAggregatedResponse>) = tokio::sync::mpsc::unbounded_channel();
        let rx = UnboundedReceiverStream::new(rx);
        let results = self.results.clone();
        tokio::spawn(async move {
            rx.for_each(|response| {
                let result = results.clone();
                async move {
                    trace!("Received response: {:?}", response);
                    result.lock().await.add_response(response);
                }
            }).await;
        });
        self.executor.run(self.requests_generator.clone(), tx).await;
        warn!("{:?}", self.results.clone());
    }

    pub(crate) fn get_results(&self) -> Arc<Mutex<BenchmarkResults>> {
        self.results.clone()
    }
}