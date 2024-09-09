use std::string::ParseError;
use std::io;
use std::sync::{Arc, Mutex};
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


pub struct App {
    exit: bool,
    store: Arc<Mutex<Store>>,
    dispatcher: Dispatcher,
}

pub fn run_console(url: String,
                   tokenizer_name: String,
                   max_vus: u64,
                   duration: std::time::Duration,
                   rate: Option<f64>,
                   benchmark_kind: String,
                   prewarm_duration: std::time::Duration,
) {
    let mut terminal = ratatui::init();
    let mut app = App::new();
    app.dispatcher.dispatch(Action::LogMessage(LogMessage {
        message: "Hello world".to_string(),
        level: LogLevel::Info,
    }));
    let app_result = app.run(&mut terminal);
    ratatui::restore();
}

impl App {
    pub fn new() -> App {
        let store = Arc::from(Mutex::new(Store::new()));
        App {
            exit: false,
            store: store.clone(),
            dispatcher: Dispatcher { store: store.clone() },
        }
    }
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events()?;
        }
        Ok(())
    }
    fn draw(&self, frame: &mut Frame) {
        frame.render_widget(self, frame.area())
    }
    fn handle_events(&mut self) -> io::Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => {}
        };
        Ok(())
    }
    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('q') => self.exit(),
            _ => {}
        }
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
            state.messages.iter().map(|m| Row::new(vec![m.level.to_string().to_uppercase().red().bold(), m.message.clone().white()])).collect::<Vec<_>>(),
            vec![Constraint::Length(10), Constraint::Min(5)])
            .block(logs_block)
            .render(main_layout[1], buf);
        let text = Text::from(vec![Line::from(vec![
            "test".into()
        ])]);
        let steps_block_title = Title::from("Benchmark steps".bold());
        let steps_block = Block::bordered()
            .title(steps_block_title.alignment(Alignment::Center))
            .border_set(border::THICK);
        let rows = [Row::new(vec!["Cell1", "Cell2", "Cell3"])];
        let widths = [
            Constraint::Length(10),
            Constraint::Length(5),
            Constraint::Length(5),
        ];
        let table = Table::new(rows, widths)
            .block(steps_block)
            .render(steps_graph_layout[0], buf);
    }
}


// Flux pattern
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
    messages: Vec<LogMessage>,
}

impl AppState {
    fn new() -> Self {
        Self {
            counter: 0,
            messages: Vec::new(),
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
        }
    }

    fn get_state(&self) -> AppState {
        self.state.clone()
    }
}

enum Action {
    Increment,
    Decrement,
    LogMessage(LogMessage),
}
#[derive(Clone,strum_macros::Display)]
enum LogLevel {
    Info,
    Warning,
    Error,
}

#[derive(Clone)]
struct LogMessage {
    message: String,
    level: LogLevel,
}

