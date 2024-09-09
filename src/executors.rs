use std::sync::Arc;
use tokio::sync::{Mutex};
use std::sync::atomic::{AtomicBool, AtomicI64};
use std::time::Duration;
use async_trait::async_trait;
use futures_util::FutureExt;
use log::debug;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender};
use tokio::task::JoinHandle;
use crate::requests::{TextGenerationAggregatedResponse, TextGenerationBackend, TextGenerationRequest, TextGenerationResponse};

pub(crate) struct ExecutorConfig {
    pub(crate) max_vus: u32,
    pub(crate) duration: Duration,
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
    pub(crate) fn new(backend: Box<dyn TextGenerationBackend + Send + Sync>, max_vus: u32, duration: Duration) -> ThroughputExecutor {
        Self {
            backend,
            config: ExecutorConfig {
                max_vus,
                duration,
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
        // channel to receive request responses
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
            // NOTE: as we maintain a constant number of VUs we don't need to update the active_vus counter
            if start.elapsed() > self.config.duration {
                break;
            }
            let mut requests_guard =requests.lock().await;
            let request= Arc::from(requests_guard.generate_request());
            start_vu(self.backend.clone(), request, responses_tx.clone(), end_tx.clone()).await;
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
        let send_thread=tokio::spawn(async move {
            while let Some(response) = rx.recv().await {
                responses_tx.send(response).expect("Could not send response to channel.");
            }
        });
        req_thread.await.unwrap();
        send_thread.await.unwrap();
        // signal that the VU work is done
        end_tx.send(true).await.unwrap();
        return true;
    })
}