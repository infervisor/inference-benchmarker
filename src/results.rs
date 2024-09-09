use std::fmt::{Debug, Formatter};
use crate::requests::TextGenerationAggregatedResponse;

#[derive(Clone)]
pub(crate) struct BenchmarkResult {
    aggregated_responses: Vec<TextGenerationAggregatedResponse>,
}

impl BenchmarkResult {
    pub(crate) fn new() -> BenchmarkResult {
        BenchmarkResult {
            aggregated_responses: Vec::new(),
        }
    }

    pub(crate) fn add_response(&mut self, response: TextGenerationAggregatedResponse) {
        self.aggregated_responses.push(response);
    }

    pub(crate) fn total_responses(&self) -> usize {
        self.aggregated_responses.len()
    }


    pub(crate) fn start_time(&self) -> Option<std::time::Instant> {
        self.aggregated_responses.first().map_or(None, |response| response.start_time)
    }

    pub(crate) fn end_time(&self) -> Option<std::time::Instant> {
        self.aggregated_responses.last().map_or(None, |response| response.end_time)
    }

    fn is_ready(&self) -> bool {
        self.start_time().is_some() && self.end_time().is_some()
    }

    pub(crate) fn failed_requests(&self) -> usize {
        self.aggregated_responses.iter().filter(|response| response.failed).count()
    }

    pub(crate) fn successful_requests(&self) -> usize {
        self.aggregated_responses.iter().filter(|response| !response.failed).count()
    }

    pub(crate) fn token_throughput_secs(&self) -> anyhow::Result<f64> {
        if self.is_ready() {
            let total_tokens: u32 = self.aggregated_responses.iter().map(|response| response.num_generated_tokens).sum();
            Ok(total_tokens as f64 / self.duration().unwrap_or_default().as_secs_f64())
        } else {
            Err(anyhow::anyhow!("No responses to calculate throughput"))
        }
    }

    pub(crate) fn duration(&self) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            Ok(self.end_time().unwrap().duration_since(self.start_time().unwrap()))
        } else {
            Err(anyhow::anyhow!("No responses to calculate duration"))
        }
    }

    pub(crate) fn average_time_to_first_token(&self) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            let mut total_time = std::time::Duration::new(0, 0);
            for response in &self.aggregated_responses {
                total_time += response.time_to_first_token().unwrap_or_default();
            }
            Ok(total_time / self.total_responses() as u32)
        } else {
            Err(anyhow::anyhow!("No responses to calculate TTFT"))
        }
    }

    pub(crate) fn average_inter_token_latency(&self) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            let mut total_time = std::time::Duration::new(0, 0);
            for response in &self.aggregated_responses {
                total_time += response.inter_token_latency().unwrap_or_default();
            }
            Ok(total_time / self.total_responses() as u32)
        } else {
            Err(anyhow::anyhow!("No responses to calculate ITL"))
        }
    }
}

impl Debug for BenchmarkResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BenchmarkResult")
            .field("total_responses", &self.total_responses())
            .field("start_time", &self.start_time())
            .field("end_time", &self.end_time())
            .field("token_throughput_secs", &self.token_throughput_secs())
            .field("duration", &self.duration())
            .field("average_time_to_first_token", &self.average_time_to_first_token())
            .field("average_inter_token_latency", &self.average_inter_token_latency())
            .field("failed_requests", &self.failed_requests())
            .field("successful_requests", &self.successful_requests())
            .finish()
    }
}