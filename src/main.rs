use std::time::Duration;
use clap::{Error, Parser};
use clap::error::ErrorKind::InvalidValue;
use log::error;
use reqwest::Url;
use tokio::sync::broadcast;
use text_generation_inference_benchmark::{run, RunConfiguration, TokenizeOptions};

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
    #[clap(default_value = "60s", short, long, env)]
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
    /// Disable console UI
    #[clap(short, long, env)]
    no_console: bool,
    /// Constraints for prompt length.
    /// No value means use the input prompt as defined in input dataset.
    /// We sample the number of tokens to generate from a normal distribution.
    /// Specified as a comma-separated list of key=value pairs.
    /// * num_tokens: target number of prompt tokens
    /// * min_tokens: minimum number of prompt tokens
    /// * max_tokens: maximum number of prompt tokens
    /// * variance: variance in the number of prompt tokens
    ///
    /// Example: num_tokens=50,max_tokens=60,min_tokens=40,variance=10
    #[clap(
        default_value = "num_tokens=50,max_tokens=60,min_tokens=40,variance=10",
        long,
        env,
        value_parser(parse_tokenizer_options)
    )]
    prompt_options: Option<TokenizeOptions>,
    /// Constraints for the generated text.
    /// We sample the number of tokens to generate from a normal distribution.
    /// Specified as a comma-separated list of key=value pairs.
    /// * num_tokens: target number of generated tokens
    /// * min_tokens: minimum number of generated tokens
    /// * max_tokens: maximum number of generated tokens
    /// * variance: variance in the number of generated tokens
    ///
    /// Example: num_tokens=50,max_tokens=60,min_tokens=40,variance=10
    #[clap(
        default_value = "num_tokens=50,max_tokens=60,min_tokens=40,variance=10",
        long,
        env,
        value_parser(parse_tokenizer_options)
    )]
    decode_options: Option<TokenizeOptions>,
    /// Hugging Face dataset to use for prompt generation
    #[clap(default_value="hlarcher/share_gpt_small",long,env)]
    dataset: String,
    /// File to use in the Dataset
    #[clap(default_value="share_gpt_filtered_small.json",long,env)]
    dataset_file: String,
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

fn parse_tokenizer_options(s: &str) -> Result<TokenizeOptions, Error> {
    let mut tokenizer_options = TokenizeOptions::new();
    let items = s.split(",").collect::<Vec<&str>>();
    for item in items.iter() {
        let key_value = item.split("=").collect::<Vec<&str>>();
        if key_value.len() != 2 {
            return Err(Error::new(InvalidValue));
        }
        match key_value[0] {
            "num_tokens" => tokenizer_options.num_tokens = key_value[1].parse::<u64>().unwrap(),
            "min_tokens" => tokenizer_options.min_tokens = key_value[1].parse::<u64>().unwrap(),
            "max_tokens" => tokenizer_options.max_tokens = key_value[1].parse::<u64>().unwrap(),
            "variance" => tokenizer_options.variance = key_value[1].parse::<u64>().unwrap(),
            _ => return Err(Error::new(InvalidValue)),
        }
    };
    if tokenizer_options.num_tokens == 0 || tokenizer_options.min_tokens == 0 || tokenizer_options.max_tokens == 0 || tokenizer_options.min_tokens > tokenizer_options.max_tokens {
        return Err(Error::new(InvalidValue));
    }
    Ok(tokenizer_options)
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
    let run_config = RunConfiguration {
        url: args.url.clone(),
        tokenizer_name: args.tokenizer_name.clone(),
        max_vus: args.max_vus,
        duration: args.duration,
        rate: args.rate,
        num_rates: args.num_rates,
        benchmark_kind: args.benchmark_kind.clone(),
        warmup_duration: args.warmup,
        interactive: !args.no_console,
        prompt_options: args.prompt_options.clone(),
        decode_options: args.decode_options.clone(),
        dataset: args.dataset.clone(),
        dataset_file: args.dataset_file.clone(),
    };
    let main_thread = tokio::spawn(async move {
        match run(run_config,
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
