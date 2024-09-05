use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64};
use std::time::Duration;
use log::debug;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::JoinHandle;
use crate::requests::TextGenerationRequest;

pub(crate) struct ExecutorConfig {
    pub(crate) max_vus: u32,
    pub(crate) duration: Duration,
}

pub trait Executor {
    async fn run(&self, requests: Box<dyn crate::requests::TextRequestGenerator>);
}

pub(crate) struct ThroughputExecutor {
    config: ExecutorConfig,
}

impl ThroughputExecutor {
    pub(crate) fn new(max_vus: u32, duration: Duration) -> Self {
        Self {
            config: ExecutorConfig {
                max_vus,
                duration,
            }
        }
    }
}

impl Executor for ThroughputExecutor {
    async fn run(&self, mut requests: Box<dyn crate::requests::TextRequestGenerator>) {
        let start = std::time::Instant::now();
        // channel to handle ending VUs
        let (tx, mut rx): (Sender<bool>, Receiver<bool>) = tokio::sync::mpsc::channel(self.config.max_vus as usize);
        let active_vus = Arc::new(AtomicI64::new(0));
        // start all VUs
        for _ in 0..self.config.max_vus {
            start_vu(requests.generate_request().clone(), tx.clone()).await;
            active_vus.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        // replenish VUs as they finish
        while let Some(_) = rx.recv().await {
            // NOTE: as we maintain a constant number of VUs we don't need to update the active_vus counter
            if start.elapsed() > self.config.duration {
                break;
            }
            start_vu(requests.generate_request().clone(), tx.clone()).await;
        }
    }
}

async fn start_vu(request: TextGenerationRequest, stop_ch: Sender<bool>) -> JoinHandle<bool> {
    tokio::spawn(async move {
        // do the VU work here
        debug!("VU started with request: {:?}", request);
        // random sleep to simulate work
        tokio::time::sleep(Duration::from_secs(1)).await;
        // signal that the VU work is done
        stop_ch.send(true).await.unwrap();
        return true;
    })
}