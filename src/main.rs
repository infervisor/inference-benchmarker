use text_generation_inference_benchmark::run_console;
use std::string::ParseError;
use std::time::Duration;
use clap::{Error, Parser};
use clap::error::ErrorKind::InvalidValue;
use reqwest::Url;
use text_generation_inference_benchmark::{run, BenchmarkKind};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The name of the tokenizer to use
    #[clap(short, long, env)]
    tokenizer_name: String,
    /// The maximum number of virtual users to use
    #[clap(short, long, env)]
    max_vus: u64,
    /// The duration of each benchmark step
    #[clap(default_value = "10s", short, long, env)]
    #[arg(value_parser = parse_duration)]
    duration: Duration,
    /// The rate of requests to send per second (only valid for the ConstantArrivalRate benchmark)
    #[clap(short, long, env)]
    rate: Option<f64>,
    /// The kind of benchmark to run (Throughput, Sweep, Optimum)
    #[clap(default_value = "Sweep", short, long, env)]
    benchmark_kind: String,
    /// The duration of the prewarm step ran before the benchmark to warm up the backend (JIT, caches, etc.)
    #[clap(default_value = "3s", short, long, env)]
    #[arg(value_parser = parse_duration)]
    prewarm_duration: Duration,
    /// The URL of the backend to benchmark. Must be compatible with OpenAI Message API
    #[clap(default_value = "http://localhost:8000", short, long, env)]
    #[arg(value_parser = parse_url)]
    url: String,
    #[clap(default_value = "console", short, long, env)]
    output: String,
    #[command(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

fn parse_duration(s: &str) -> Result<Duration, Error> {
    humantime::parse_duration(s).map_err(|e| Error::new(InvalidValue))
}

fn parse_url(s: &str) -> Result<String, Error> {
    match Url::parse(s) {
        Ok(_) => Ok(s.to_string()),
        Err(e) => Err(Error::new(InvalidValue)),
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let args = Args::parse();

    // run(args.url, args.tokenizer_name, args.max_vus, args.duration, args.rate, args.benchmark_kind, args.prewarm_duration).await;
    run_console(args.url, args.tokenizer_name, args.max_vus, args.duration, args.rate, args.benchmark_kind, args.prewarm_duration);
}
