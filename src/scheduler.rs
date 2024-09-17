use std::sync::Arc;
use std::time::Instant;
use log::{debug, info, trace, warn};
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use crate::executors::{ConstantArrivalRateExecutor, Executor, ExecutorConfig, ConstantVUsExecutor};
use crate::requests::{TextGenerationAggregatedResponse, TextGenerationBackend, TextRequestGenerator};
use crate::results::BenchmarkResults;
use tokio::sync::{broadcast, Mutex};
use crate::results::BenchmarkErrors::NoResponses;

#[derive(Clone, strum_macros::Display)]
pub enum ExecutorType {
    ConstantVUs,
    ConstantArrivalRate,
}

pub struct Scheduler {
    id: String,
    executor: Arc<Mutex<dyn Executor + Send>>,
    requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    results: Arc<Mutex<BenchmarkResults>>,
    progress_tx: Sender<Option<SchedulerProgress>>,
    stop_sender: broadcast::Sender<()>,
}

pub struct SchedulerProgress {
    pub progress: f64,
    pub requests_throughput: f64,
    pub successful_requests: u64,
    pub failed_requests: u64,
}

impl Scheduler {
    pub fn new(id: String,
               backend: Box<dyn TextGenerationBackend + Send + Sync>,
               executor_type: ExecutorType,
               config: ExecutorConfig,
               requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>,
               progress_tx: Sender<Option<SchedulerProgress>>,
               stop_sender: broadcast::Sender<()>,
    ) -> Scheduler {
        match executor_type {
            ExecutorType::ConstantVUs => {
                Scheduler {
                    id: id.clone(),
                    executor: Arc::from(Mutex::from(ConstantVUsExecutor::new(backend.clone(), config.max_vus.clone(), config.duration.clone()))),
                    results: Arc::from(Mutex::from(BenchmarkResults::new(id.clone(), ExecutorType::ConstantVUs, config))),
                    requests_generator,
                    progress_tx,
                    stop_sender,
                }
            }
            ExecutorType::ConstantArrivalRate => {
                if config.rate.is_none() {
                    panic!("Rate must be specified for ConstantArrivalRateExecutor");
                }
                let rate = config.rate.unwrap();
                Scheduler {
                    id: id.clone(),
                    executor: Arc::from(Mutex::from(ConstantArrivalRateExecutor::new(backend.clone(), config.max_vus.clone(), config.duration.clone(), rate))),
                    results: Arc::from(Mutex::from(BenchmarkResults::new(id.clone(), ExecutorType::ConstantArrivalRate, config))),
                    requests_generator,
                    progress_tx,
                    stop_sender,
                }
            }
        }
    }

    pub async fn run(&mut self) -> anyhow::Result<BenchmarkResults> {
        debug!("Starting scheduler '{}'", self.id);
        // add responses to the benchmark result as they arrive
        let (tx, mut rx): (UnboundedSender<TextGenerationAggregatedResponse>, UnboundedReceiver<TextGenerationAggregatedResponse>) = tokio::sync::mpsc::unbounded_channel();
        let results = self.results.clone();
        let progress_tx = self.progress_tx.clone();
        let mut stop_receiver = self.stop_sender.subscribe();
        tokio::spawn(async move {
            tokio::select! {
                _ = stop_receiver.recv() => {
                    info!("Received stop signal, stopping benchmark");
                    return
                }
                _ = async{
                    while let Some(response) = rx.recv().await{
                        let result = results.clone();
                        let progress_tx = progress_tx.clone();
                        trace!("Received response: {:?}", response);
                        if response.ended {
                            return;
                        }
                        let mut result = result.lock().await;
                        result.add_response(response);
                        let expected_duration = result.executor_config().duration.as_secs_f64();
                        let start_time = result.start_time().unwrap_or(Instant::now());
                        let _ = progress_tx.send(Some(SchedulerProgress {
                            progress: (100.0 * (1.0 - (expected_duration - start_time.elapsed().as_secs_f64()) / expected_duration)).min(100.0),
                            requests_throughput: result.successful_request_rate().unwrap_or_default(),
                            successful_requests: result.successful_requests() as u64,
                            failed_requests: result.failed_requests() as u64,
                        })).await;
                    }
                }=>{}
            }
        });
        self.executor.lock().await.run(self.requests_generator.clone(), tx, self.stop_sender.clone()).await;
        warn!("{:?}", self.results.clone());
        if self.results.lock().await.successful_requests() == 0 {
            return Err(anyhow::anyhow!(NoResponses));
        } else {
            Ok(self.results.lock().await.clone())
        }
    }

    pub fn get_results(&self) -> Arc<Mutex<BenchmarkResults>> {
        self.results.clone()
    }
}