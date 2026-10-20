//! Run a benchmark from a spec file, without a terminal UI: built for batch
//! jobs on the machine that serves the model.
//!
//! ```text
//! inference-benchmarker-run --spec spec.json --out results/
//! ```
//!
//! Writes `report.json`, `trace.jsonl` and `result.json` into `--out`. The
//! server's API key is read from `BENCH_API_KEY` and a Hugging Face token (for
//! gated tokenizers) from `HF_TOKEN`; neither belongs in the spec. SIGINT and
//! SIGTERM stop the run, and the steps that finished are still written. The
//! exit code is 0 for a completed or stopped run and 1 for a failed one.

use inference_benchmarker::spec::{run_spec, JobSpec, SpecStatus};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use tokio::sync::broadcast;

fn usage() -> ExitCode {
    eprintln!("usage: inference-benchmarker-run --spec <spec.json> --out <dir>");
    ExitCode::from(2)
}

async fn stop_on_signal(stop: broadcast::Sender<()>) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = term.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
            let _ = stop.send(());
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
    let _ = stop.send(());
}

#[tokio::main]
async fn main() -> ExitCode {
    let (mut spec_path, mut out) = (None::<PathBuf>, None::<PathBuf>);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--spec" => spec_path = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            _ => return usage(),
        }
    }
    let (Some(spec_path), Some(out)) = (spec_path, out) else {
        return usage();
    };
    let spec: JobSpec = match std::fs::read(&spec_path)
        .map_err(|e| e.to_string())
        .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
    {
        Ok(spec) => spec,
        Err(e) => {
            eprintln!("cannot read the spec {}: {e}", spec_path.display());
            return ExitCode::from(2);
        }
    };
    let hf_token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    let cache = inference_benchmarker::hub::default_cache_dir();
    let tokenizer = match inference_benchmarker::hub::load_tokenizer_or_fallback(
        &spec.config.tokenizer,
        hf_token.as_deref(),
        Some(&cache),
    )
    .await
    {
        Ok((t, _)) => Arc::new(t),
        Err(e) => {
            eprintln!("cannot load the tokenizer {}: {e}", spec.config.tokenizer);
            return ExitCode::FAILURE;
        }
    };
    let (stop, _) = broadcast::channel(4);
    tokio::spawn(stop_on_signal(stop.clone()));
    let api_key = std::env::var("BENCH_API_KEY").unwrap_or_default();
    match run_spec(spec, api_key, tokenizer, &out, stop).await {
        Ok(result) => {
            println!("{}", serde_json::to_string(&result).unwrap_or_default());
            if result.status == SpecStatus::Failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("cannot write the results: {e}");
            ExitCode::FAILURE
        }
    }
}
