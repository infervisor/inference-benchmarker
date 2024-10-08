use crossterm::event;
use crossterm::event::KeyEvent;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc};

pub enum AppEvent {
    Tick,
    Key(KeyEvent),
    Resize,
}

pub async fn terminal_event_task(
    fps: u32,
    event_sender: mpsc::Sender<AppEvent>,
    mut shutdown_receiver: broadcast::Receiver<()>,
) {
    // End task if a message is received on shutdown_receiver
    // _shutdown_guard_sender will be dropped once the task is finished
    tokio::select! {
        _ = event_loop(fps, event_sender)  => {
        },
        _ = shutdown_receiver.recv() => {}
    }
}

async fn event_loop(fps: u32, event_sender: mpsc::Sender<AppEvent>) {
    // Frame budget
    let per_frame = Duration::from_secs(1) / fps;

    // When was last frame executed
    let mut last_frame = Instant::now();

    loop {
        // Sleep to avoid blocking the thread for too long
        if let Some(sleep) = per_frame.checked_sub(last_frame.elapsed()) {
            tokio::time::sleep(sleep).await;
        }

        // Get crossterm event and send a new one over the channel
        if event::poll(Duration::from_secs(0)).expect("no events available") {
            match event::read().expect("unable to read event") {
                event::Event::Key(e) => event_sender.send(AppEvent::Key(e)).await.unwrap_or(()),
                event::Event::Resize(_w, _h) => {
                    event_sender.send(AppEvent::Resize).await.unwrap_or(())
                }
                _ => (),
            }
        }

        // Frame budget exceeded
        if last_frame.elapsed() >= per_frame {
            // Send tick
            event_sender.send(AppEvent::Tick).await.unwrap_or(());
            // Rest last_frame time
            last_frame = Instant::now();
        }
    }
}
