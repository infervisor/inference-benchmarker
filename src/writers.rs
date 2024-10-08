use crate::results::{BenchmarkReport, BenchmarkResults};
use crate::{executors, BenchmarkConfig};
use serde::Serialize;
use std::path::Path;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, System};
use tokio::fs;

#[derive(Serialize)]
pub struct BenchmarkResultsWriter {
    id: String,
    executor_type: String,
    config: executors::ExecutorConfig,
    total_requests: u64,
    total_tokens: u64,
    token_throughput_secs: f64,
    duration_ms: u128,
    time_to_first_token_ms_avg: f64,
    time_to_first_token_ms_p90: f64,
    time_to_first_token_ms_p95: f64,
    inter_token_latency_ms_avg: f64,
    inter_token_latency_ms_p90: f64,
    inter_token_latency_ms_p95: f64,
    failed_requests: u64,
    successful_requests: u64,
    request_rate: f64,
    total_tokens_sent: u64,
    e2e_latency_ms_avg: u128,
    e2e_latency_ms_p90: u128,
    e2e_latency_ms_p95: u128,
}

impl BenchmarkResultsWriter {
    pub fn new(results: BenchmarkResults) -> anyhow::Result<BenchmarkResultsWriter> {
        Ok(BenchmarkResultsWriter {
            id: results.id.clone(),
            executor_type: results.executor_type().to_string(),
            config: results.executor_config(),
            total_requests: results.total_requests() as u64,
            total_tokens: results.total_tokens(),
            token_throughput_secs: results.token_throughput_secs()?,
            duration_ms: results.duration().ok().unwrap().as_micros() / 1000,
            time_to_first_token_ms_avg: results.time_to_first_token_avg().ok().unwrap().as_micros()
                as f64
                / 1000.,
            time_to_first_token_ms_p90: results.time_to_first_token_percentile(0.9)?.as_micros()
                as f64
                / 1000.,
            time_to_first_token_ms_p95: results.time_to_first_token_percentile(0.95)?.as_micros()
                as f64
                / 1000.,
            inter_token_latency_ms_avg: results.inter_token_latency_avg().ok().unwrap().as_micros()
                as f64
                / 1000.,
            inter_token_latency_ms_p90: results.inter_token_latency_percentile(0.9)?.as_micros()
                as f64
                / 1000.,
            inter_token_latency_ms_p95: results.inter_token_latency_percentile(0.95)?.as_micros()
                as f64
                / 1000.,
            failed_requests: results.failed_requests() as u64,
            successful_requests: results.successful_requests() as u64,
            request_rate: results.successful_request_rate()?,
            total_tokens_sent: results.total_tokens_sent(),
            e2e_latency_ms_avg: results.e2e_latency_avg().ok().unwrap().as_micros() / 1000,
            e2e_latency_ms_p90: results.e2e_latency_percentile(0.9)?.as_micros() / 1000,
            e2e_latency_ms_p95: results.e2e_latency_percentile(0.95)?.as_micros() / 1000,
        })
    }
}

#[derive(Serialize)]
pub struct SystemInfo {
    pub cpu: Vec<String>,
    pub memory: String,
    pub os_name: String,
    pub os_version: String,
    pub kernel: String,
    pub hostname: String,
}

impl SystemInfo {
    pub fn new() -> SystemInfo {
        let s = System::new_with_specifics(
            sysinfo::RefreshKind::new()
                .with_memory(MemoryRefreshKind::everything())
                .with_cpu(CpuRefreshKind::everything()),
        );
        let cpu_info = s
            .cpus()
            .iter()
            .map(|cpu| format!("{} {}@{:.0}MHz", cpu.brand(), cpu.name(), cpu.frequency()))
            .collect::<Vec<String>>();
        SystemInfo {
            cpu: cpu_info,
            memory: format!(
                "{:.2} GB",
                s.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0
            ),
            os_name: System::name().ok_or("N/A").unwrap(),
            os_version: System::os_version().ok_or("N/A").unwrap(),
            kernel: System::kernel_version().ok_or("N/A").unwrap(),
            hostname: System::host_name().ok_or("N/A").unwrap(),
        }
    }
}

#[derive(Serialize)]
pub struct BenchmarkReportWriter {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResultsWriter>,
    start_time: String,
    end_time: String,
    system: SystemInfo,
}

impl BenchmarkReportWriter {
    pub fn new(
        config: BenchmarkConfig,
        report: BenchmarkReport,
    ) -> anyhow::Result<BenchmarkReportWriter> {
        let mut results: Vec<BenchmarkResultsWriter> = Vec::new();
        for result in report.get_results() {
            let writer = BenchmarkResultsWriter::new(result)?;
            results.push(writer);
        }
        Ok(BenchmarkReportWriter {
            config,
            results,
            start_time: report
                .start_time()
                .ok_or(anyhow::anyhow!("start_time not set"))?
                .to_rfc3339(),
            end_time: report
                .end_time()
                .ok_or(anyhow::anyhow!("end_time not set"))?
                .to_rfc3339(),
            system: SystemInfo::new(),
        })
    }
    pub async fn json(&self, path: &Path) -> anyhow::Result<()> {
        // write the benchmark report to json
        let report = serde_json::to_string(&self)?;

        // create path hierarchy if it doesn't exist
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).await?;
            }
        }
        fs::write(path, report).await?;
        Ok(())
    }
}
