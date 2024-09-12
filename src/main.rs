use std::time::Duration;
use clap::{Error, Parser};
use clap::error::ErrorKind::InvalidValue;
use log::error;
use reqwest::Url;
use tokio::sync::broadcast;
use text_generation_inference_benchmark::{run};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The name of the tokenizer to use
    #[clap(short, long, env)]
    tokenizer_name: String,
    /// The maximum number of virtual users to use
    #[clap(default_value = "128", short, long, env)]
    max_vus: u64,
    /// The duration of each benchmark step
    #[clap(default_value = "10s", short, long, env)]
    #[arg(value_parser = parse_duration)]
    duration: Duration,
    /// The rate of requests to send per second (only valid for the ConstantArrivalRate benchmark)
    #[clap(short, long, env)]
    rate: Option<f64>,
    /// The number of rates to sweep through (only valid for the "sweep" benchmark)
    /// The rates will be linearly spaced up to the detected maximum rate
    #[clap(default_value = "10", long, env)]
    num_rates: u64,
    /// The kind of benchmark to run (throughput, sweep, optimum)
    #[clap(default_value = "sweep", short, long, env)]
    benchmark_kind: String,
    /// The duration of the prewarm step ran before the benchmark to warm up the backend (JIT, caches, etc.)
    #[clap(default_value = "30s", short, long, env)]
    #[arg(value_parser = parse_duration)]
    warmup: Duration,
    /// The URL of the backend to benchmark. Must be compatible with OpenAI Message API
    #[clap(default_value = "http://localhost:8000", short, long, env)]
    #[arg(value_parser = parse_url)]
    url: String,
    #[clap(short, long, env)]
    no_console: bool,
}

fn parse_duration(s: &str) -> Result<Duration, Error> {
    humantime::parse_duration(s).map_err(|_| Error::new(InvalidValue))
}

fn parse_url(s: &str) -> Result<String, Error> {
    match Url::parse(s) {
        Ok(_) => Ok(s.to_string()),
        Err(_) => Err(Error::new(InvalidValue)),
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let (stop_sender, _) = broadcast::channel(1);
    // handle ctrl-c
    let stop_sender_clone = stop_sender.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to listen for ctrl-c");
        error!("Received stop signal, stopping benchmark");
        stop_sender_clone.send(()).expect("Failed to send stop signal");
    });

    let stop_sender_clone = stop_sender.clone();
    let main_thread = tokio::spawn(async move {
        match run(args.url,
                  args.tokenizer_name,
                  args.max_vus,
                  args.duration,
                  args.rate,
                  args.num_rates,
                  args.benchmark_kind,
                  args.warmup,
                  !args.no_console,
                  stop_sender_clone,
        ).await {
            Ok(_) => {}
            Err(e) => {
                println!("Error: {:?}", e)
            }
        };
    });
    main_thread.await.expect("Failed to run main thread");
}
