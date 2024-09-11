use std::fmt::{Debug, Display, Formatter};
use std::time::Duration;
use crate::executors::ExecutorConfig;
use crate::requests::TextGenerationAggregatedResponse;
use crate::results::BenchmarkErrors::NoResponses;
use crate::scheduler::ExecutorType;

#[derive(Debug)]
pub(crate) enum BenchmarkErrors {
    NoResponses,
}

impl Display for BenchmarkErrors {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NoResponses => write!(f, "Backend did not return any valid response."),
        }
    }
}

#[derive(Clone)]
pub struct BenchmarkResults {
    pub id: String,
    aggregated_responses: Vec<TextGenerationAggregatedResponse>,
    executor_type: ExecutorType,
    executor_config: ExecutorConfig,
}

impl BenchmarkResults {
    pub fn new(id: String, executor_type: ExecutorType, executor_config: ExecutorConfig) -> BenchmarkResults {
        BenchmarkResults {
            id,
            aggregated_responses: Vec::new(),
            executor_type,
            executor_config,
        }
    }

    pub fn add_response(&mut self, response: TextGenerationAggregatedResponse) {
        self.aggregated_responses.push(response);
    }

    pub fn total_requests(&self) -> usize {
        self.aggregated_responses.len()
    }


    pub fn start_time(&self) -> Option<std::time::Instant> {
        self.aggregated_responses.first().map_or(None, |response| response.start_time)
    }

    pub fn end_time(&self) -> Option<std::time::Instant> {
        self.aggregated_responses.last().map_or(None, |response| response.end_time)
    }

    fn is_ready(&self) -> bool {
        self.start_time().is_some() && self.end_time().is_some()
    }

    pub fn failed_requests(&self) -> usize {
        self.aggregated_responses.iter().filter(|response| response.failed).count()
    }

    pub fn successful_requests(&self) -> usize {
        self.aggregated_responses.iter().filter(|response| !response.failed).count()
    }

    pub fn token_throughput_secs(&self) -> anyhow::Result<f64> {
        if self.is_ready() {
            let total_tokens: u32 = self.total_tokens();
            Ok(total_tokens as f64 / self.duration().unwrap_or_default().as_secs_f64())
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn successful_request_rate(&self) -> anyhow::Result<f64> {
        if self.is_ready() {
            let total_requests = self.successful_requests();
            Ok(total_requests as f64 / self.duration().unwrap_or_default().as_secs_f64())
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn total_tokens(&self) -> u32 {
        self.get_successful_responses().iter().map(|response| response.num_generated_tokens).sum()
    }

    pub fn duration(&self) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            Ok(self.end_time().unwrap().duration_since(self.start_time().unwrap()))
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn time_to_first_token_avg(&self) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            let mut total_time = std::time::Duration::new(0, 0);
            for response in self.get_successful_responses() {
                total_time += response.time_to_first_token().unwrap_or_default();
            }
            Ok(total_time / self.total_requests() as u32)
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn time_to_first_token_percentile(&self, percentile: f64) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            let mut times: Vec<std::time::Duration> = self.get_successful_responses().iter().map(|response| response.time_to_first_token().unwrap_or_default()).collect();
            times.sort();
            let index = (percentile * times.len() as f64) as usize;
            if index >= times.len() {
                return Err(anyhow::anyhow!(NoResponses));
            }
            Ok(times[index])
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn inter_token_latency_avg(&self) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            let mut total_time = std::time::Duration::new(0, 0);
            for response in self.get_successful_responses() {
                total_time += response.inter_token_latency().unwrap_or_default();
            }
            Ok(total_time / self.total_requests() as u32)
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn inter_token_latency_percentile(&self, percentile: f64) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            let mut times: Vec<std::time::Duration> = self.get_successful_responses().iter().map(|response| response.inter_token_latency().unwrap_or_default()).collect();
            times.sort();
            let index = (percentile * times.len() as f64) as usize;
            Ok(times[index])
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn executor_type(&self) -> ExecutorType {
        self.executor_type.clone()
    }

    pub fn executor_config(&self) -> ExecutorConfig {
        self.executor_config.clone()
    }

    fn get_successful_responses(&self) -> Vec<&TextGenerationAggregatedResponse> {
        self.aggregated_responses.iter().filter(|response| !response.failed).collect()
    }
}

impl Debug for BenchmarkResults {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BenchmarkResult")
            .field("id", &self.id)
            .field("executor_type", &self.executor_type.to_string())
            .field("total_requests", &self.total_requests())
            .field("start_time", &self.start_time())
            .field("end_time", &self.end_time())
            .field("total_tokens", &self.total_tokens())
            .field("token_throughput_secs", &self.token_throughput_secs().or::<anyhow::Result<f64>>(Ok(-1.0)))
            .field("duration", &self.duration().or::<anyhow::Result<Duration>>(Ok(Duration::from_secs(0))))
            .field("average_time_to_first_token", &self.time_to_first_token_avg().or::<anyhow::Result<Duration>>(Ok(Duration::from_secs(0))))
            .field("average_inter_token_latency", &self.inter_token_latency_avg().or::<anyhow::Result<Duration>>(Ok(Duration::from_secs(0))))
            .field("failed_requests", &self.failed_requests())
            .field("successful_requests", &self.successful_requests())
            .field("request_rate", &self.successful_request_rate().or::<anyhow::Result<f64>>(Ok(-1.0)))
            .finish()
    }
}


#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    results: Vec<BenchmarkResults>,
}

impl BenchmarkReport {
    pub fn new() -> BenchmarkReport {
        BenchmarkReport {
            results: Vec::new(),
        }
    }

    pub fn add_benchmark_result(&mut self, result: BenchmarkResults) {
        self.results.push(result);
    }

    pub fn get_results(&self) -> Vec<BenchmarkResults> {
        self.results.clone()
    }
}