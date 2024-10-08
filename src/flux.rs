use crate::results::BenchmarkResults;
use std::sync::{Arc, Mutex};

// Flux pattern
#[derive(Clone)]
pub struct Dispatcher {
    store: Arc<Mutex<Store>>,
}

impl Dispatcher {
    pub(crate) fn new(store: Arc<Mutex<Store>>) -> Self {
        Self { store }
    }
    pub(crate) fn dispatch(&mut self, action: Action) {
        self.store.lock().unwrap().update(action);
    }
}

#[derive(Clone)]
pub struct AppState {
    pub(crate) messages: Vec<crate::app::LogMessageUI>,
    pub(crate) benchmarks: Vec<crate::app::BenchmarkUI>,
    pub(crate) results: Vec<BenchmarkResults>,
}

impl AppState {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
            benchmarks: Vec::new(),
            results: Vec::new(),
        }
    }
}

pub struct Store {
    state: AppState,
}

impl Store {
    pub(crate) fn new() -> Self {
        let state = AppState::new();
        Self { state }
    }

    fn update(&mut self, action: Action) {
        match action {
            Action::LogMessage(message) => self.state.messages.push(message),
            Action::AddBenchmark(benchmark) => {
                // add or update benchmark
                let index = self
                    .state
                    .benchmarks
                    .iter()
                    .position(|b| b.id == benchmark.id);
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
                let index = self
                    .state
                    .results
                    .iter_mut()
                    .position(|b| b.id == results.id);
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

    pub(crate) fn get_state(&self) -> AppState {
        self.state.clone()
    }
}

pub enum Action {
    LogMessage(crate::app::LogMessageUI),
    AddBenchmark(crate::app::BenchmarkUI),
    AddBenchmarkResults(BenchmarkResults),
}
