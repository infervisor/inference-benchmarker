use std::string::ParseError;
use std::io;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use clap::builder::Str;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Rect},
    style::Stylize as OtherStylize,
    symbols::border,
    text::{Line, Text},
    widgets::{
        block::{Position, Title},
        Block, Paragraph, Widget,
    },
    DefaultTerminal, Frame,
};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::widgets::{Row, Table};
use strum_macros::EnumString;
use tokio::sync::{broadcast, mpsc};
use tokio::sync::mpsc::{Receiver, UnboundedReceiver};
use crate::benchmark::Event as BenchmarkEvent;
use crate::event::{AppEvent, terminal_event_task};


pub struct App {
    exit: bool,
    store: Arc<Mutex<Store>>,
    dispatcher: Arc<Mutex<Dispatcher>>,
    receiver: Receiver<AppEvent>,
}

pub async fn run_console(
    mut receiver: UnboundedReceiver<BenchmarkEvent>,
) {
    let (shutdown_sender, _) = broadcast::channel(1);
    let (shutdown_guard_sender, mut shutdown_guard_receiver) = mpsc::channel(1);
    let (app_tx, app_rx) = mpsc::channel(8);
    // Create event task
    tokio::spawn(terminal_event_task(
        250,
        app_tx,
        shutdown_sender.subscribe(),
        shutdown_guard_sender.clone(),
    ));
    // Drop our end of shutdown sender
    drop(shutdown_guard_sender);

    let mut app = App::new(app_rx);
    app.dispatcher.lock().expect("lock").dispatch(Action::LogMessage(LogMessageUI {
        message: "Starting benchmark".to_string(),
        level: LogLevel::Info,
        timestamp: chrono::Utc::now(),
    }));
    let mut dispatcher = app.dispatcher.clone();
    let event_thread = tokio::spawn(async move {
        while let Some(event) = receiver.recv().await {
            match event {
                BenchmarkEvent::BenchmarkStart(event) => {
                    dispatcher.lock().expect("lock").dispatch(Action::AddBenchmark(BenchmarkUI {
                        id: event.id,
                        status: BenchmarkStatus::Running,
                        progress: 0.0.to_string(),
                        throughput: "-".to_string(),
                    }));
                }
                BenchmarkEvent::BenchmarkProgress(event) => {
                    dispatcher.lock().expect("lock").dispatch(Action::AddBenchmark(BenchmarkUI {
                        id: event.id,
                        status: BenchmarkStatus::Running,
                        progress: event.progress.to_string(),
                        throughput: event.request_throughput.map_or("-".to_string(), |e| format!("{e:.2}")),
                    }));
                }
                BenchmarkEvent::BenchmarkEnd(event) => {
                    dispatcher.lock().expect("lock").dispatch(Action::LogMessage(LogMessageUI {
                        message: format!("Benchmark {} ended", event.id),
                        level: LogLevel::Info,
                        timestamp: chrono::Utc::now(),
                    }));
                    dispatcher.lock().expect("lock").dispatch(Action::AddBenchmark(BenchmarkUI {
                        id: event.id,
                        status: BenchmarkStatus::Completed,
                        progress: 100.0.to_string(),
                        throughput: event.request_throughput.map_or("-".to_string(), |e| format!("{e:.2}")),
                    }));
                }
                BenchmarkEvent::Message(event) => {
                    dispatcher.lock().expect("lock").dispatch(Action::LogMessage(LogMessageUI {
                        message: event.message,
                        level: LogLevel::Info,
                        timestamp: event.timestamp,
                    }));
                }
            }
        }
    });
    let app_thread = tokio::spawn(async move {
        let mut terminal = ratatui::init();
        let _ = app.run(&mut terminal).await;
        ratatui::restore();
    });
    event_thread.await.unwrap();
    app_thread.await.unwrap();
}

impl App {
    pub fn new(receiver: Receiver<AppEvent>) -> App {
        let store = Arc::from(Mutex::new(Store::new()));
        let dispatcher = Arc::from(Mutex::new(Dispatcher { store: store.clone() }));
        App {
            exit: false,
            store: store.clone(),
            dispatcher: dispatcher.clone(),
            receiver,
        }
    }
    pub async fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events().await?;
        }
        Ok(())
    }
    fn draw(&self, frame: &mut Frame) {
        frame.render_widget(self, frame.area())
    }
    async fn handle_events(&mut self) -> io::Result<()> {
        match self.receiver.recv().await {
            None => Err(io::Error::new(io::ErrorKind::Other, "No event")),
            Some(event) => match event {
                AppEvent::Tick => { Ok(()) }
                AppEvent::Key(key_event) => self.handle_key_event(key_event),
                AppEvent::Resize => { Ok(()) }
            }
        }
    }
    fn handle_terminal_events(&mut self) -> io::Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => { Ok(()) }
        }
    }
    fn handle_key_event(&mut self, key_event: KeyEvent) -> io::Result<()> {
        match key_event.code {
            KeyCode::Char('q') => self.exit(),
            _ => {}
        }
        Ok(())
    }
    fn exit(&mut self) {
        self.exit = true;
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let state = self.store.lock().unwrap().get_state();
        let main_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                [
                    Constraint::Percentage(50),
                    Constraint::Percentage(50)
                ]
            )
            .split(area);
        let steps_graph_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Percentage(50),
                    Constraint::Percentage(50)
                ]
            )
            .split(main_layout[0]);
        let logs_title = Title::from("Logs".bold());
        let logs_block = Block::bordered()
            .title(logs_title.alignment(Alignment::Center))
            .border_set(border::THICK);
        let logs_table = Table::new(
            state.messages.iter().map(|m| {
                let cells = vec![
                    m.formatted_timestamp().clone().gray(),
                    m.message.clone().white(),
                ];
                Row::new(cells)
            }).collect::<Vec<_>>(),
            vec![Constraint::Length(30), Constraint::Min(5)])
            .block(logs_block)
            .render(main_layout[1], buf);
        let text = Text::from(vec![Line::from(vec![
            "test".into()
        ])]);
        let steps_block_title = Title::from("Benchmark steps".bold());
        let steps_block = Block::bordered()
            .title(steps_block_title.alignment(Alignment::Center))
            .border_set(border::THICK);
        let step_rows = state.benchmarks.iter().map(|b| {
            let cells = vec![
                b.id.clone().white(),
                b.status.to_string().white(),
                format!("{:.3}%", b.progress).white(),
                format!("{} req/sec avg", b.throughput).on_green().bold(),
            ];
            Row::new(cells)
        }).collect::<Vec<_>>();
        let widths = [
            Constraint::Length(30),
            Constraint::Length(15),
            Constraint::Length(5),
            Constraint::Length(10),
        ];
        let table = Table::new(step_rows, widths)
            .block(steps_block)
            .render(steps_graph_layout[0], buf);
    }
}


// Flux pattern
#[derive(Clone)]
struct Dispatcher {
    store: Arc<Mutex<Store>>,
}

impl Dispatcher {
    fn dispatch(&mut self, action: Action) {
        let _ = self.store.lock().unwrap().update(action);
    }
}

#[derive(Clone)]
struct AppState {
    counter: i32,
    messages: Vec<LogMessageUI>,
    benchmarks: Vec<BenchmarkUI>,
}

impl AppState {
    fn new() -> Self {
        Self {
            counter: 0,
            messages: Vec::new(),
            benchmarks: Vec::new(),
        }
    }
}

struct Store {
    state: AppState,
}

impl Store {
    fn new() -> Self {
        let state = AppState::new();
        Self {
            state,
        }
    }

    fn update(&mut self, action: Action) {
        match action {
            Action::Increment => self.state.counter += 1,
            Action::Decrement => self.state.counter -= 1,
            Action::LogMessage(message) => self.state.messages.push(message),
            Action::AddBenchmark(benchmark) => {
                // add or update benchmark
                let index = self.state.benchmarks.iter().position(|b| b.id == benchmark.id);
                match index {
                    Some(i) => {
                        self.state.benchmarks[i] = benchmark;
                    }
                    None => {
                        self.state.benchmarks.push(benchmark);
                    }
                }
            }
        }
    }

    fn get_state(&self) -> AppState {
        self.state.clone()
    }
}

enum Action {
    Increment,
    Decrement,
    LogMessage(LogMessageUI),
    AddBenchmark(BenchmarkUI),
}

#[derive(Clone, strum_macros::Display)]
enum LogLevel {
    Info,
    Warning,
    Error,
}

#[derive(Clone)]
struct LogMessageUI {
    message: String,
    level: LogLevel,
    timestamp: chrono::DateTime<chrono::Utc>,
}

impl LogMessageUI {
    fn formatted_timestamp(&self) -> String {
        // self.timestamp.format("%Y-%m-%d %H:%M:%SZ").to_string()
        self.timestamp.to_rfc3339()
    }
}

#[derive(Clone)]
struct BenchmarkUI {
    id: String,
    status: BenchmarkStatus,
    progress: String,
    throughput: String,
}

#[derive(Clone, strum_macros::Display)]
enum BenchmarkStatus {
    Running,
    Completed,
    Failed,
}
