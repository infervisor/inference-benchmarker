use std::collections::HashMap;
use std::string::ParseError;
use std::io;
use std::iter::Map;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use clap::builder::Str;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{buffer::Buffer, layout::{Alignment, Rect}, style::Stylize as OtherStylize, symbols::border, text::{Line, Text}, widgets::{
    block::{Position, Title},
    Block, Paragraph, Widget,
}, DefaultTerminal, Frame, symbols};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::text::Span;
use ratatui::widgets::{Dataset, LegendPosition, List, ListItem, Row, Table};
use ratatui::widgets::ListDirection::BottomToTop;
use strum_macros::EnumString;
use tokio::sync::{broadcast, mpsc};
use tokio::sync::broadcast::Sender;
use tokio::sync::mpsc::{Receiver, UnboundedReceiver};
use crate::benchmark::Event as BenchmarkEvent;
use crate::BenchmarkConfig;
use crate::event::{AppEvent, terminal_event_task};
use crate::results::BenchmarkResults;
use crate::scheduler::ExecutorType;


pub struct App {
    exit: bool,
    store: Arc<Mutex<Store>>,
    dispatcher: Arc<Mutex<Dispatcher>>,
    receiver: Receiver<AppEvent>,
    benchmark_config: BenchmarkConfig,
    stop_sender: broadcast::Sender<()>,
}

pub async fn run_console(
    benchmark_config: BenchmarkConfig,
    mut receiver: UnboundedReceiver<BenchmarkEvent>,
    mut stop_sender: broadcast::Sender<()>,
) {
    let (app_tx, app_rx) = mpsc::channel(8);
    // Create event task
    let stop_receiver_signal = stop_sender.subscribe();
    tokio::spawn(terminal_event_task(
        250,
        app_tx,
        stop_receiver_signal,
    ));

    let mut app = App::new(benchmark_config, app_rx, stop_sender.clone());
    app.dispatcher.lock().expect("lock").dispatch(Action::LogMessage(LogMessageUI {
        message: "Starting benchmark".to_string(),
        level: LogLevel::Info,
        timestamp: chrono::Utc::now(),
    }));
    let mut dispatcher = app.dispatcher.clone();
    let mut stop_receiver_signal = stop_sender.subscribe();
    let event_thread = tokio::spawn(async move {
        tokio::select! {
            _=async{
                while let Some(event) = receiver.recv().await {
                    match event {
                        BenchmarkEvent::BenchmarkStart(event) => {
                            dispatcher.lock().expect("lock").dispatch(Action::AddBenchmark(BenchmarkUI {
                                id: event.id,
                                status: BenchmarkStatus::Running,
                                progress: 0.0,
                                throughput: "-".to_string(),
                            }));
                        }
                        BenchmarkEvent::BenchmarkProgress(event) => {
                            dispatcher.lock().expect("lock").dispatch(Action::AddBenchmark(BenchmarkUI {
                                id: event.id,
                                status: BenchmarkStatus::Running,
                                progress: event.progress,
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
                                progress: 100.0,
                                throughput: event.request_throughput.map_or("-".to_string(), |e| format!("{e:.2}")),
                            }));
                            match event.results {
                                Some(results) => {
                                    dispatcher.lock().expect("lock").dispatch(Action::AddBenchmarkResults(results));
                                }
                                None => {}
                            }
                        }
                        BenchmarkEvent::Message(event) => {
                            dispatcher.lock().expect("lock").dispatch(Action::LogMessage(LogMessageUI {
                                message: event.message,
                                level: LogLevel::Info,
                                timestamp: event.timestamp,
                            }));
                        }
                        BenchmarkEvent::BenchmarkReportEnd => {
                            dispatcher.lock().expect("lock").dispatch(Action::LogMessage(LogMessageUI {
                                message: "Benchmark report saved.".to_string(),
                                level: LogLevel::Info,
                                timestamp: chrono::Utc::now(),
                            }));
                            break;
                        }
                    }
                }
            }=>{}
            _ = stop_receiver_signal.recv() => {}
        }
    });
    let mut stop_receiver_signal = stop_sender.subscribe();
    let app_thread = tokio::spawn(async move {
        tokio::select! {
            _ = async {
                let _ = app.run(&mut ratatui::init()).await;
                ratatui::restore();
            }=>{}
            _ = stop_receiver_signal.recv() => {}
        }
    });
    event_thread.await.unwrap();
    app_thread.await.unwrap();
}

impl App {
    pub fn new(benchmark_config: BenchmarkConfig, receiver: Receiver<AppEvent>, stop_sender:Sender<()>) -> App {
        let store = Arc::from(Mutex::new(Store::new()));
        let dispatcher = Arc::from(Mutex::new(Dispatcher { store: store.clone() }));
        App {
            exit: false,
            store: store.clone(),
            dispatcher: dispatcher.clone(),
            receiver,
            benchmark_config,
            stop_sender,
        }
    }
    pub async fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events().await?;
        }
        // signal everybody to stop
        let _ = self.stop_sender.send(());
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
        match key_event {
            KeyEvent {
                code: KeyCode::Char('q'), ..
            } => self.exit(),
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => self.exit(),
            _ => {}
        }
        Ok(())
    }
    fn exit(&mut self) {
        self.exit = true;
    }

    fn create_datasets(&self, state: AppState) -> HashMap<String, Vec<(f64, f64)>> {
        let token_throughput_rate = state.results.iter().filter_map(|r| {
            match r.executor_type() {
                ExecutorType::ConstantArrivalRate => {
                    let throughput = r.token_throughput_secs().unwrap_or(0.0);
                    Some((r.executor_config().rate.unwrap(), throughput))
                }
                ExecutorType::ConstantVUs => None
            }
        }).collect::<Vec<_>>();
        let token_throughput_vus = state.results.iter().filter_map(|r| {
            match r.executor_type() {
                ExecutorType::ConstantVUs => {
                    let throughput = r.token_throughput_secs().unwrap_or(0.0);
                    Some((r.executor_config().max_vus as f64, throughput))
                }
                ExecutorType::ConstantArrivalRate => None
            }
        }).collect::<Vec<_>>();
        let inter_token_latency_rate = state.results.iter().filter_map(|r| {
            match r.executor_type() {
                ExecutorType::ConstantArrivalRate => {
                    let latency = r.inter_token_latency_avg().unwrap_or_default().as_secs_f64();
                    Some((r.executor_config().rate.unwrap(), latency))
                }
                ExecutorType::ConstantVUs => None
            }
        }).collect::<Vec<_>>();
        let inter_token_latency_vus = state.results.iter().filter_map(|r| {
            match r.executor_type() {
                ExecutorType::ConstantVUs => {
                    let latency = r.inter_token_latency_avg().unwrap_or_default().as_secs_f64();
                    Some((r.executor_config().max_vus as f64, latency))
                }
                ExecutorType::ConstantArrivalRate => None
            }
        }).collect::<Vec<_>>();
        return HashMap::from([
            ("token_throughput_rate".to_string(), token_throughput_rate),
            ("token_throughput_vus".to_string(), token_throughput_vus),
            ("inter_token_latency_rate".to_string(), inter_token_latency_rate),
            ("inter_token_latency_vus".to_string(), inter_token_latency_vus),
        ]);
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let state = self.store.lock().unwrap().get_state();
        let data = self.create_datasets(state.clone());

        let main_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                [
                    Constraint::Length(1),
                    Constraint::Min(20),
                ]
            )
            .split(area);
        let bottom_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                [
                    Constraint::Percentage(50),
                    Constraint::Percentage(50)
                ]
            )
            .split(main_layout[1]);
        let steps_graph_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Percentage(35),
                    Constraint::Percentage(65),
                ]
            )
            .split(bottom_layout[0]);
        // LOGS
        let logs_title = Title::from("Logs".bold());
        let logs_block = Block::bordered()
            .title(logs_title.alignment(Alignment::Center))
            .border_set(border::THICK);
        List::new(
            state.messages.iter().rev().map(|m| {
                let content = Line::from(vec![
                    m.formatted_timestamp().clone().gray(),
                    Span::raw(" "),
                    Span::raw(m.message.to_string()).bold(),
                ]);
                ListItem::new(content)
            }).collect::<Vec<_>>())
            .direction(BottomToTop)
            .block(logs_block)
            .render(bottom_layout[1], buf);

        // BENCHMARK config
        let config_text = Text::from(vec![Line::from(vec![
            format!("Benchmark: {kind} | Max VUs: {max_vus} | Duration: {duration} sec | Rate: {rate}req/s | Warmup: {warmup} sec",
                    kind = self.benchmark_config.benchmark_kind,
                    max_vus = self.benchmark_config.max_vus,
                    duration = self.benchmark_config.duration.as_secs_f64(),
                    rate = self.benchmark_config.rate.or(Some(0.0)).unwrap(),
                    warmup = self.benchmark_config.warmup_duration.as_secs_f64()).white().bold(),
        ])]);
        Paragraph::new(config_text.clone())
            .render(main_layout[0], buf);

        // STEPS
        let steps_block_title = Title::from("Benchmark steps".bold());
        let steps_block = Block::bordered()
            .title(steps_block_title.alignment(Alignment::Center))
            .border_set(border::THICK);
        let step_rows = state.benchmarks.iter().map(|b| {
            let cells = vec![
                b.id.clone().white(),
                b.status.to_string().white(),
                format!("{:4.0}%", b.progress).white(),
                format!("{:>6.6} req/sec avg", b.throughput).green().bold(),
            ];
            Row::new(cells)
        }).collect::<Vec<_>>();
        let widths = [
            Constraint::Length(30),
            Constraint::Length(10),
            Constraint::Length(5),
            Constraint::Length(20),
        ];
        // steps table
        Table::new(step_rows, widths)
            .block(steps_block)
            .render(steps_graph_layout[0], buf);

        // CHARTS
        let graphs_block_title = Title::from("Token throughput rate".bold());
        let graphs_block = Block::bordered()
            .title(graphs_block_title.alignment(Alignment::Center))
            .border_set(border::THICK);
        let binding = data.get("token_throughput_rate").unwrap().clone();
        let datasets = vec![
            Dataset::default()
                .name("Token throughput rate".to_string())
                .marker(symbols::Marker::Dot)
                .graph_type(ratatui::widgets::GraphType::Scatter)
                .style(ratatui::style::Style::default().fg(ratatui::style::Color::LightMagenta))
                .data(&*binding)
        ];
        let (xmax, ymax) = get_max_bounds(&binding, (10.0, 100.0));
        let x_axis = ratatui::widgets::Axis::default()
            .title("Arrival rate (req/s)".to_string())
            .style(ratatui::style::Style::default().white())
            .bounds([0.0, xmax])
            .labels(get_axis_labels(0.0, xmax, 5));
        let y_axis = ratatui::widgets::Axis::default()
            .title("Throughput (tokens/s)".to_string())
            .style(ratatui::style::Style::default().white())
            .bounds([0.0, ymax])
            .labels(get_axis_labels(0.0, ymax, 5));
        ratatui::widgets::Chart::new(datasets)
            .x_axis(x_axis)
            .y_axis(y_axis)
            .block(graphs_block)
            .legend_position(None)
            .render(steps_graph_layout[1], buf);
    }
}

fn get_max_bounds(data: &Vec<(f64, f64)>, default_max: (f64, f64)) -> (f64, f64) {
    let xmax = data.iter().map(|(x, _)| x).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&default_max.0);
    let ymax = data.iter().map(|(_, y)| y).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&default_max.1);
    (*xmax, *ymax)
}

fn get_axis_labels(min: f64, max: f64, num_labels: u32) -> Vec<String> {
    let step = (max - min) / num_labels as f64;
    (0..num_labels).map(|i| format!("{:.2}", min + i as f64 * step)).collect()
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
    results: Vec<BenchmarkResults>,
}

impl AppState {
    fn new() -> Self {
        Self {
            counter: 0,
            messages: Vec::new(),
            benchmarks: Vec::new(),
            results: Vec::new(),
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
            Action::AddBenchmarkResults(results) => {
                let index = self.state.results.iter_mut().position(|b| b.id == results.id);
                match index {
                    Some(i) => {
                        self.state.results[i] = results;
                    }
                    None => {
                        self.state.results.push(results);
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
    AddBenchmarkResults(BenchmarkResults),
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
    progress: f64,
    throughput: String,
}

#[derive(Clone, strum_macros::Display)]
enum BenchmarkStatus {
    Running,
    Completed,
    Failed,
}
