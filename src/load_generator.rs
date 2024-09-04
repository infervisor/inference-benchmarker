use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64};
use std::time::Duration;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::JoinHandle;

pub(crate) struct ExecutorConfig {
    pub(crate) max_vus: u32,
    pub(crate) duration: Duration,
}

trait Executor {
    async fn run(config: ExecutorConfig);
}

struct ThroughputExecutor {}

impl Executor for ThroughputExecutor {
    async fn run(config: ExecutorConfig) {
        let start = std::time::Instant::now();
        let (tx, rx): (Sender<bool>, Receiver<bool>) = tokio::sync::mpsc::channel(config.max_vus as usize);
        let active_vus = Arc::new(AtomicI64::new(0));
        // start all VUs
        for _ in 0..config.max_vus {
            start_vu(tx.clone()).await;
            active_vus.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        while let Some(_) = rx.recv().await {
            if start.elapsed() > config.duration {
                break;
            }
            start_vu(tx.clone());
            active_vus.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

async fn start_vu(tx: Sender<bool>) -> JoinHandle<bool> {
    tokio::spawn(async {
        
        // signal that the VU work is done
        tx.send(true).await.unwrap();
        return true;
    })
}