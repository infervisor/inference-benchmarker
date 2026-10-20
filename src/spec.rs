//! Run a benchmark from a file: a [`JobSpec`] in, a report, a trace and a
//! result summary out. This is what makes the benchmarker usable from a batch
//! job on the GPU machine itself (the `inference-benchmarker-run` binary), where
//! there is no network hop between the load generator and the server.
//!
//! The spec holds everything a run needs except secrets: the server's API key is
//! given separately, so a spec file is safe to keep and share.

use crate::checks::Check;
use crate::record::{RecordSink, RequestRecord};
use crate::requests::{ConversationEntry, OpenAITextGenerationMessage, TextGenerationRequest};
use crate::{run_benchmark, BenchmarkConfig, Cancelled, RunOptions, Target, Workload};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokenizers::Tokenizer;
use tokio::io::AsyncWriteExt;
use tokio::sync::{broadcast, mpsc};

/// Largest trace file by default; later records are counted, not kept.
pub const DEFAULT_MAX_TRACE_BYTES: u64 = 64 * 1024 * 1024;
/// File names a run writes into its output directory.
pub const REPORT_FILE: &str = "report.json";
pub const TRACE_FILE: &str = "trace.jsonl";
pub const RESULT_FILE: &str = "result.json";

/// One request of a fixed list, as it is written in a spec.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpecRequest {
    pub prompt: String,
    /// Messages sent before `prompt`: the system prompt and earlier turns.
    #[serde(default)]
    pub history: Vec<OpenAITextGenerationMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub turn: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check: Option<Check>,
    /// Replay: when to send it, in milliseconds from the start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_decode_tokens: Option<u64>,
}

impl From<SpecRequest> for TextGenerationRequest {
    fn from(r: SpecRequest) -> Self {
        TextGenerationRequest {
            prompt: r.prompt,
            history: r.history,
            conversation_id: r.conversation_id,
            turn: r.turn,
            check: r.check,
            offset: r.offset_ms.map(Duration::from_millis),
            num_decode_tokens: r.num_decode_tokens,
            ..Default::default()
        }
    }
}

impl From<&TextGenerationRequest> for SpecRequest {
    fn from(r: &TextGenerationRequest) -> Self {
        SpecRequest {
            prompt: r.prompt.clone(),
            history: r.history.clone(),
            conversation_id: r.conversation_id.clone(),
            turn: r.turn,
            check: r.check.clone(),
            offset_ms: r.offset.map(|o| o.as_millis() as u64),
            num_decode_tokens: r.num_decode_tokens,
        }
    }
}

/// What to send, as written in a spec (see [`Workload`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpecWorkload {
    Synthetic,
    Requests { requests: Vec<SpecRequest> },
    Conversations { entries: Vec<ConversationEntry> },
}

impl From<SpecWorkload> for Workload {
    fn from(w: SpecWorkload) -> Self {
        match w {
            SpecWorkload::Synthetic => Workload::Synthetic,
            SpecWorkload::Requests { requests } => {
                Workload::Requests(requests.into_iter().map(Into::into).collect())
            }
            SpecWorkload::Conversations { entries } => Workload::Conversations(entries),
        }
    }
}

impl From<&Workload> for SpecWorkload {
    fn from(w: &Workload) -> Self {
        match w {
            Workload::Synthetic => SpecWorkload::Synthetic,
            Workload::Requests(r) => SpecWorkload::Requests {
                requests: r.iter().map(Into::into).collect(),
            },
            Workload::Conversations(c) => SpecWorkload::Conversations { entries: c.clone() },
        }
    }
}

fn default_max_trace_bytes() -> u64 {
    DEFAULT_MAX_TRACE_BYTES
}

/// A whole benchmark run, as a file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobSpec {
    pub config: BenchmarkConfig,
    /// The server, without the `/v1/...` path.
    pub url: String,
    pub model: String,
    pub workload: SpecWorkload,
    /// Keep the messages sent and the text generated in the trace.
    #[serde(default)]
    pub include_text: bool,
    #[serde(default = "default_max_trace_bytes")]
    pub max_trace_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecStatus {
    Completed,
    /// Stopped early; the report holds the steps that finished.
    Cancelled,
    Failed,
}

/// `result.json`: how the run ended, next to the report and trace files.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpecResult {
    pub status: SpecStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Why the first failed request failed: the useful part of a bad run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_error: Option<String>,
    pub records: u64,
    /// More records were seen than `max_trace_bytes` allowed to keep.
    pub truncated: bool,
}

#[derive(Default)]
struct Trace {
    bytes: Vec<u8>,
    records: u64,
    truncated: bool,
    first_error: Option<String>,
}

async fn collect(mut rx: mpsc::Receiver<RequestRecord>, max_bytes: u64) -> Trace {
    let mut trace = Trace::default();
    while let Some(record) = rx.recv().await {
        if trace.first_error.is_none() && !record.ok {
            trace.first_error = record.error.as_ref().map(|e| e.message.clone());
        }
        let Ok(mut line) = serde_json::to_vec(&record) else {
            continue;
        };
        if (trace.bytes.len() + line.len() + 1) as u64 > max_bytes {
            trace.truncated = true;
            continue;
        }
        line.push(b'\n');
        trace.bytes.extend_from_slice(&line);
        trace.records += 1;
    }
    trace
}

async fn write(dir: &Path, name: &str, bytes: &[u8]) -> anyhow::Result<()> {
    let mut file = tokio::fs::File::create(dir.join(name)).await?;
    file.write_all(bytes).await?;
    file.flush().await?;
    Ok(())
}

/// Run `spec` against its server and write `report.json`, `trace.jsonl` and
/// `result.json` into `out`. A run that fails or is stopped still writes what
/// it has; only a failure to write the files is an error here.
pub async fn run_spec(
    spec: JobSpec,
    api_key: String,
    tokenizer: Arc<Tokenizer>,
    out: &Path,
    stop: broadcast::Sender<()>,
) -> anyhow::Result<SpecResult> {
    tokio::fs::create_dir_all(out).await?;
    let (tx, rx) = mpsc::channel(4096);
    let trace = tokio::spawn(collect(rx, spec.max_trace_bytes));
    let target = Target {
        url: spec.url.parse()?,
        api_key,
        model: spec.model,
    };
    let options = RunOptions {
        events: None,
        records: Some(RecordSink {
            tx,
            include_text: spec.include_text,
        }),
    };
    let result = run_benchmark(
        spec.config,
        target,
        spec.workload.into(),
        tokenizer,
        options,
        stop,
    )
    .await;
    // the sink is dropped with the run, which ends the collector
    let trace = trace.await?;
    let (status, error, report) = match result {
        Ok(report) => (SpecStatus::Completed, None, Some(report)),
        Err(e) => match e.downcast::<Cancelled>() {
            Ok(cancelled) => (SpecStatus::Cancelled, None, cancelled.partial.map(|r| *r)),
            Err(e) => (SpecStatus::Failed, Some(e.to_string()), None),
        },
    };
    if let Some(report) = &report {
        write(out, REPORT_FILE, &serde_json::to_vec_pretty(report)?).await?;
    }
    if !trace.bytes.is_empty() {
        write(out, TRACE_FILE, &trace.bytes).await?;
    }
    let summary = SpecResult {
        status,
        error,
        first_error: trace.first_error,
        records: trace.records,
        truncated: trace.truncated,
    };
    write(out, RESULT_FILE, &serde_json::to_vec_pretty(&summary)?).await?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checks::Check;

    #[test]
    fn requests_round_trip_through_a_spec() {
        let request = TextGenerationRequest {
            prompt: "2+2?".into(),
            history: vec![OpenAITextGenerationMessage::new("system", "Be brief.")],
            conversation_id: Some("q1".into()),
            turn: 2,
            check: Some(Check::Choice { answer: "B".into() }),
            offset: Some(Duration::from_millis(2500)),
            num_decode_tokens: Some(40),
            ..Default::default()
        };
        let wire = SpecRequest::from(&request);
        let json = serde_json::to_string(&wire).unwrap();
        let back: TextGenerationRequest =
            serde_json::from_str::<SpecRequest>(&json).unwrap().into();
        assert_eq!(back, request);
        // optional fields may be left out of a hand-written spec
        let minimal: SpecRequest = serde_json::from_str(r#"{"prompt":"hi"}"#).unwrap();
        assert_eq!(
            (minimal.turn, minimal.history.len(), minimal.offset_ms),
            (0, 0, None)
        );
    }

    #[test]
    fn a_spec_is_plain_json() {
        let spec = JobSpec {
            config: BenchmarkConfig {
                max_vus: 4,
                duration: Duration::from_secs(30),
                ..Default::default()
            },
            url: "http://127.0.0.1:8000".into(),
            model: "m".into(),
            workload: SpecWorkload::Conversations {
                entries: vec![ConversationEntry {
                    id: "c".into(),
                    conversations: vec![],
                }],
            },
            include_text: true,
            max_trace_bytes: 1024,
        };
        let json = serde_json::to_value(&spec).unwrap();
        assert_eq!(json["workload"]["kind"], "conversations");
        assert_eq!(json["config"]["duration_secs"], 30);
        assert!(
            json.get("api_key").is_none(),
            "the key never travels in the spec"
        );
        let back: JobSpec = serde_json::from_value(json).unwrap();
        assert_eq!((back.model.as_str(), back.max_trace_bytes), ("m", 1024));
    }

    #[tokio::test]
    async fn a_run_writes_report_trace_and_result() {
        let mut server = mockito::Server::new_async().await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"B\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n"
        );
        server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .expect_at_least(1)
            .create_async()
            .await;
        let spec = JobSpec {
            config: BenchmarkConfig {
                max_vus: 1,
                benchmark_kind: crate::BenchmarkKind::Once,
                tokenizer: "gpt2".into(),
                model_name: "m".into(),
                run_id: "spec-test".into(),
                ..Default::default()
            },
            url: server.url(),
            model: "m".into(),
            workload: SpecWorkload::Requests {
                requests: vec![SpecRequest {
                    prompt: "pick".into(),
                    history: vec![],
                    conversation_id: Some("q1".into()),
                    turn: 0,
                    check: Some(Check::Choice { answer: "B".into() }),
                    offset_ms: None,
                    num_decode_tokens: None,
                }],
            },
            include_text: true,
            max_trace_bytes: DEFAULT_MAX_TRACE_BYTES,
        };
        let dir = std::env::temp_dir().join(format!("spec-run-{}", uuid::Uuid::new_v4()));
        let (stop, _) = broadcast::channel(1);
        let result = run_spec(
            spec,
            String::new(),
            crate::test_util::tokenizer(),
            &dir,
            stop,
        )
        .await
        .unwrap();
        assert_eq!(
            (result.status, result.records, result.truncated),
            (SpecStatus::Completed, 1, false),
            "{result:?}"
        );
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join(REPORT_FILE)).unwrap()).unwrap();
        assert_eq!(report["results"][0]["accuracy"]["passed"], 1);
        let trace = std::fs::read_to_string(dir.join(TRACE_FILE)).unwrap();
        let record: RequestRecord = serde_json::from_str(trace.lines().next().unwrap()).unwrap();
        assert!(record.ok && record.response.as_deref() == Some("B"));
        let written: SpecResult =
            serde_json::from_slice(&std::fs::read(dir.join(RESULT_FILE)).unwrap()).unwrap();
        assert_eq!(written, result);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_failing_run_still_writes_a_result() {
        let spec = JobSpec {
            config: BenchmarkConfig {
                max_vus: 1,
                benchmark_kind: crate::BenchmarkKind::Once,
                ..Default::default()
            },
            url: "http://127.0.0.1:9".into(),
            model: "m".into(),
            workload: SpecWorkload::Requests { requests: vec![] },
            include_text: false,
            max_trace_bytes: 1024,
        };
        let dir = std::env::temp_dir().join(format!("spec-run-{}", uuid::Uuid::new_v4()));
        let (stop, _) = broadcast::channel(1);
        let result = run_spec(
            spec,
            String::new(),
            crate::test_util::tokenizer(),
            &dir,
            stop,
        )
        .await
        .unwrap();
        assert_eq!(result.status, SpecStatus::Failed);
        assert!(result.error.is_some());
        assert!(!dir.join(REPORT_FILE).exists() && dir.join(RESULT_FILE).exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
