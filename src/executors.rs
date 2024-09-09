use std::sync::Arc;
use tokio::sync::{Mutex};
use std::sync::atomic::{AtomicBool, AtomicI64};
use std::time::Duration;
use async_trait::async_trait;
use futures_util::FutureExt;
use log::{debug, info, warn};
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use crate::requests::{TextGenerationAggregatedResponse, TextGenerationBackend, TextGenerationRequest, TextGenerationResponse, TextRequestGenerator};

pub(crate) struct ExecutorConfig {
    pub(crate) max_vus: u64,
    pub(crate) duration: Duration,
    pub(crate) rate: Option<f64>,
}

#[async_trait]
pub(crate) trait Executor {
    async fn run(&self, requests: Arc<Mutex<dyn crate::requests::TextRequestGenerator + Send>>, responses_tx: UnboundedSender<TextGenerationAggregatedResponse>);
}

pub(crate) struct ThroughputExecutor {
    config: ExecutorConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
}

impl ThroughputExecutor {
    pub(crate) fn new(backend: Box<dyn TextGenerationBackend + Send + Sync>, max_vus: u64, duration: Duration) -> ThroughputExecutor {
        Self {
            backend,
            config: ExecutorConfig {
                max_vus,
                duration,
                rate: None,
            },
        }
    }
}

#[async_trait]
impl Executor for ThroughputExecutor {
    async fn run(&self, mut requests: Arc<Mutex<dyn crate::requests::TextRequestGenerator + Send>>, responses_tx: UnboundedSender<TextGenerationAggregatedResponse>) {
        let start = std::time::Instant::now();
        // channel to handle ending VUs
        let (end_tx, mut end_rx): (Sender<bool>, Receiver<bool>) = tokio::sync::mpsc::channel(self.config.max_vus as usize);
        let active_vus = Arc::new(AtomicI64::new(0));
        // start all VUs
        for _ in 0..self.config.max_vus {
            let mut requests_guard = requests.lock().await;
            let request = Arc::from(requests_guard.generate_request());
            start_vu(self.backend.clone(), request, responses_tx.clone(), end_tx.clone()).await;
            active_vus.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        // replenish VUs as they finish
        while let Some(_) = end_rx.recv().await {
            active_vus.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            if start.elapsed() > self.config.duration {
                info!("Duration reached, waiting for all VUs to finish...");
                if active_vus.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                    break;
                }
            } else {
                let mut requests_guard = requests.lock().await;
                let request = Arc::from(requests_guard.generate_request());
                active_vus.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                start_vu(self.backend.clone(), request, responses_tx.clone(), end_tx.clone()).await;
            }
        }
    }
}

async fn start_vu(backend: Box<dyn TextGenerationBackend + Send + Sync>, request: Arc<TextGenerationRequest>, responses_tx: UnboundedSender<TextGenerationAggregatedResponse>, end_tx: Sender<bool>) -> JoinHandle<bool> {
    tokio::spawn(async move {
        let (tx, mut rx): (Sender<TextGenerationAggregatedResponse>, Receiver<TextGenerationAggregatedResponse>) = tokio::sync::mpsc::channel(1);
        debug!("VU started with request: {:?}", request);
        let req_thread = tokio::spawn(async move {
            backend.generate(request.clone(), tx).await;
        });
        let send_thread = tokio::spawn(async move {
            while let Some(response) = rx.recv().await {
                // ignore errors, if the receiver is gone we want to finish the request
                // to leave remote server in clean state
                let _ = responses_tx.send(response);
            }
        });
        req_thread.await.unwrap();
        send_thread.await.unwrap();
        // signal that the VU work is done
        end_tx.send(true).await.expect("Could not send end signal to channel.");
        return true;
    })
}

pub(crate) struct ConstantArrivalRateExecutor {
    config: ExecutorConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
}

impl ConstantArrivalRateExecutor {
    pub(crate) fn new(backend: Box<dyn TextGenerationBackend + Send + Sync>, max_vus: u64, duration: Duration, rate: f64) -> ConstantArrivalRateExecutor {
        Self {
            backend,
            config: ExecutorConfig {
                max_vus,
                duration,
                rate: Some(rate),
            },
        }
    }
}

#[async_trait]
impl Executor for ConstantArrivalRateExecutor {
    async fn run(&self, requests: Arc<Mutex<dyn TextRequestGenerator + Send>>, responses_tx: UnboundedSender<TextGenerationAggregatedResponse>) {
        let start = std::time::Instant::now();
        let active_vus = Arc::new(AtomicI64::new(0));
        // channel to handle ending VUs
        let (end_tx, mut end_rx): (Sender<bool>, Receiver<bool>) = tokio::sync::mpsc::channel(self.config.max_vus as usize);
        let rate = self.config.rate.expect("checked in new()");
        // spawn new VUs every `tick_ms` to reach the expected `rate` per second, until the duration is reached
        let tick_ms = 10;
        let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));

        let backend = self.backend.clone();
        let duration = self.config.duration;
        let max_vus = self.config.max_vus;
        let active_vus_thread = active_vus.clone();
        let vu_thread = tokio::spawn(async move {
            let mut spawn_queue = rate.max(1.0); // start with at least one VU
            while start.elapsed() < duration {
                // delay spawning if we can't spawn a full VU yet
                if spawn_queue < 1.0 {
                    spawn_queue += rate * (tick_ms as f64) / 1000.0;
                    interval.tick().await;
                    continue;
                }
                // spawn VUs, keep track of the fraction of VU to spawn for the next iteration
                let to_spawn = spawn_queue.floor() as u64;
                spawn_queue -= to_spawn as f64;
                for _ in 0..to_spawn {
                    if active_vus_thread.load(std::sync::atomic::Ordering::SeqCst) < max_vus.clone() as i64 {
                        let mut requests_guard = requests.lock().await;
                        let request = Arc::from(requests_guard.generate_request());
                        start_vu(backend.clone(), request.clone(), responses_tx.clone(), end_tx.clone()).await;
                        active_vus_thread.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    } else {
                        warn!("Max VUs reached, skipping request");
                        break;
                    }
                }
                interval.tick().await;
            }
            drop(responses_tx); // drop response sender to signal VUs to stop
        });
        while let Some(_) = end_rx.recv().await {
            active_vus.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            // wait for all VUs to finish
            if start.elapsed() > self.config.duration {
                info!("Duration reached, waiting for all VUs to finish...");
                if active_vus.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                    break;
                }
            }
        }
        // wait for the VU thread to finish
        vu_thread.await.unwrap();
    }
}