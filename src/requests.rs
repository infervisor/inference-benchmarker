use async_trait::async_trait;
use futures_util::StreamExt;
use hf_hub::api::sync::ApiBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, trace, warn};
use rand_distr::Distribution;
use rayon::iter::split;
use rayon::prelude::*;
use reqwest_eventsource::{Error, Event, EventSource};
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::atomic::AtomicI64;
use std::sync::{Arc, Mutex};
use std::time;
use tokenizers::{FromPretrainedParameters, Tokenizer};
use tokio::sync::mpsc::Sender;
use tokio::time::sleep;

#[derive(Debug, Clone)]
pub struct TextGenerationRequest {
    pub prompt: String,
    pub num_prompt_tokens: u64, // this includes the system prompt if present
    pub num_decode_tokens: Option<u64>,
    pub system_prompt: Option<String>,
}

#[async_trait]
pub trait TextGenerationBackend: TextGenerationBackendClone {
    async fn generate(
        &self,
        request: Arc<TextGenerationRequest>,
        sender: Sender<TextGenerationAggregatedResponse>,
    );
}

pub trait TextGenerationBackendClone {
    fn clone_box(&self) -> Box<dyn TextGenerationBackend + Send + Sync>;
}

impl<T> TextGenerationBackendClone for T
where
    T: 'static + TextGenerationBackend + Clone + Send + Sync,
{
    fn clone_box(&self) -> Box<dyn TextGenerationBackend + Send + Sync> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn TextGenerationBackend + Send + Sync> {
    fn clone(&self) -> Box<dyn TextGenerationBackend + Send + Sync> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct OpenAITextGenerationBackend {
    pub api_key: String,
    pub base_url: String,
    pub model_name: String,
    pub client: reqwest::Client,
    pub tokenizer: Arc<Tokenizer>,
    pub timeout: time::Duration,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAITextGenerationMessage {
    pub content: String,
    pub role: String,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAITextGenerationDelta {
    pub content: String,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAITextGenerationChoice {
    pub message: Option<OpenAITextGenerationMessage>,
    pub finish_reason: Option<String>,
    pub delta: Option<OpenAITextGenerationDelta>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct OpenAITextGenerationResponse {
    pub choices: Vec<OpenAITextGenerationChoice>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct OpenAITextGenerationRequest {
    pub model: String,
    pub messages: Vec<OpenAITextGenerationMessage>,
    pub max_tokens: Option<u64>,
    pub stream: bool,
    pub stop: Option<String>,
    pub temperature: f64,
}

impl OpenAITextGenerationBackend {
    pub fn try_new(
        api_key: String,
        base_url: String,
        model_name: String,
        tokenizer: Arc<Tokenizer>,
        timeout: time::Duration,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            base_url,
            model_name,
            tokenizer,
            timeout,
        })
    }
}

#[async_trait]
impl TextGenerationBackend for OpenAITextGenerationBackend {
    async fn generate(
        &self,
        request: Arc<TextGenerationRequest>,
        sender: Sender<TextGenerationAggregatedResponse>,
    ) {
        let url = format!("{base_url}/v1/chat/completions", base_url = self.base_url);
        let mut aggregated_response = TextGenerationAggregatedResponse::default();
        let messages = match &request.system_prompt {
            None => vec![OpenAITextGenerationMessage {
                role: "user".to_string(),
                content: request.prompt.clone(),
            }],
            Some(system_prompt) => vec![
                OpenAITextGenerationMessage {
                    role: "system".to_string(),
                    content: system_prompt.clone(),
                },
                OpenAITextGenerationMessage {
                    role: "user".to_string(),
                    content: request.prompt.clone(),
                },
            ],
        };
        let body = OpenAITextGenerationRequest {
            model: self.model_name.clone(),
            messages,
            max_tokens: request.num_decode_tokens,
            stream: true,
            stop: None,
            temperature: 0.0,
        };
        let req = self
            .client
            .post(url)
            .header(
                "Authorization",
                format!("Bearer {token}", token = self.api_key),
            )
            .json(&serde_json::json!(body))
            .timeout(self.timeout);
        // start timer
        aggregated_response.start(request.num_prompt_tokens);
        let mut es = EventSource::new(req).unwrap();
        let mut final_response = "".to_string();
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => trace!("SSE connection opened"),
                Ok(Event::Message(message)) => {
                    if message.data == "\n" || message.data == "[DONE]" {
                        aggregated_response.stop();
                        continue;
                    }
                    if message.data.starts_with("{\"error\":") {
                        error!("Error from OpenAI API: {message}", message = message.data);
                        aggregated_response.fail();
                        es.close();
                        break;
                    }
                    // deserialize message data
                    let oai_response: OpenAITextGenerationResponse =
                        match serde_json::from_str(&message.data) {
                            Ok(response) => response,
                            Err(e) => {
                                error!("Error deserializing OpenAI API response: {e}", e = e);
                                aggregated_response.fail();
                                es.close();
                                break;
                            }
                        };
                    let choices = oai_response.choices;
                    let content = choices[0].clone().delta.unwrap().content;
                    if content.is_empty() {
                        // skip empty responses
                        continue;
                    }
                    // we need to count the number of tokens generated as each delta chunk may contain multiple tokens
                    // that's the case with vLLM chunked prefill or speculative decoding
                    let num_tokens =
                        self.tokenizer.encode(content.clone(), false).unwrap().len() as u64;
                    if num_tokens > 1 {
                        warn!(
                            "Generated more than one token: {num_tokens}",
                            num_tokens = num_tokens
                        );
                    }
                    match choices[0].clone().finish_reason {
                        None => {
                            aggregated_response.add_tokens(num_tokens);
                            final_response += content.as_str();
                        }
                        Some(_) => {
                            aggregated_response.add_tokens(num_tokens);
                            aggregated_response.stop();
                            trace!("Generated text using OpenAI API | prompt: {prompt}, max tokens: {max_tokens:?}, response: {message}", prompt = request.prompt, max_tokens = request.num_decode_tokens,message = &content);
                        }
                    };
                }
                Err(e) => {
                    match e {
                        Error::Utf8(_) => {
                            aggregated_response.fail();
                        }
                        Error::Parser(_) => {
                            aggregated_response.fail();
                        }
                        Error::Transport(_) => {
                            aggregated_response.fail();
                        }
                        Error::InvalidContentType(_, _) => {
                            aggregated_response.fail();
                        }
                        Error::InvalidStatusCode(_, _) => {
                            aggregated_response.fail();
                        }
                        Error::InvalidLastEventId(_) => {
                            aggregated_response.fail();
                        }
                        Error::StreamEnded => {
                            if aggregated_response.num_generated_tokens == 0 {
                                // server sent no data
                                aggregated_response.fail();
                            }
                            if aggregated_response.end_time.is_none() {
                                // server closed the connection before we received the final response
                                warn!("Connection closed before completion. Received :: {num_tokens}/{max_tokens} tokens. Response: {final_response}", num_tokens = aggregated_response.num_generated_tokens, max_tokens = request.num_decode_tokens.unwrap_or(0));
                                aggregated_response.fail();
                            }
                        }
                    }
                    es.close();
                }
            };
        }
        sender
            .send(aggregated_response.clone())
            .await
            .expect("Error sending response to channel");
        //debug!("Final response: {response}", response = final_response);
    }
}

#[derive(Debug, Clone)]
pub struct DummyTextGenerationBackend {
    time_to_generate: time::Duration,
}

impl DummyTextGenerationBackend {
    pub fn new(time_to_generate: time::Duration) -> Self {
        Self { time_to_generate }
    }
}

impl Default for DummyTextGenerationBackend {
    fn default() -> Self {
        Self::new(time::Duration::from_secs(1))
    }
}

#[async_trait]
impl TextGenerationBackend for DummyTextGenerationBackend {
    async fn generate(
        &self,
        request: Arc<TextGenerationRequest>,
        sender: Sender<crate::requests::TextGenerationAggregatedResponse>,
    ) {
        let mut response = TextGenerationAggregatedResponse::default();
        response.start(request.num_prompt_tokens);
        let num_tokens = request.num_decode_tokens.unwrap_or(10);
        let time_per_token = self
            .time_to_generate
            .checked_div(num_tokens as u32)
            .unwrap();
        for _ in 0..num_tokens {
            sleep(time_per_token).await;
            response.add_tokens(1);
        }
        response.stop();
        sender
            .send(response.clone())
            .await
            .expect("Error sending response to channel");
    }
}

pub trait TextRequestGenerator: Sync {
    fn generate_request(&mut self) -> TextGenerationRequest;
}

#[derive(Clone)]
pub struct ConversationTextRequestGenerator {
    pub requests: Vec<TextGenerationRequest>,
    current_index: Arc<AtomicI64>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Conversation {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ConversationEntry {
    pub id: String,
    pub conversations: Vec<Conversation>,
}

#[derive(Clone, Serialize, Debug)]
pub struct TokenizeOptions {
    pub num_tokens: Option<u64>,
    pub min_tokens: u64,
    pub max_tokens: u64,
    pub variance: u64,
}

impl TokenizeOptions {
    pub fn new() -> Self {
        Self {
            num_tokens: None,
            min_tokens: 0,
            max_tokens: u64::MAX,
            variance: 0,
        }
    }
}

impl Default for TokenizeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for TokenizeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "num_tokens={num_tokens:?},min_tokens={min_tokens},max_tokens={max_tokens},variance={variance}",
            num_tokens = self.num_tokens,
            min_tokens = self.min_tokens,
            max_tokens = self.max_tokens,
            variance = self.variance
        )
    }
}

impl ConversationTextRequestGenerator {
    pub fn load(
        filepath: PathBuf,
        tokenizer: String,
        prompt_tokenize_opts: Option<TokenizeOptions>,
        decode_tokenize_opts: Option<TokenizeOptions>,
        hf_token: Option<String>,
    ) -> anyhow::Result<Self> {
        let params = FromPretrainedParameters {
            auth_token: hf_token,
            ..Default::default()
        };
        let tokenizer = match Tokenizer::from_pretrained(tokenizer, Some(params)) {
            Ok(tokenizer) => tokenizer,
            Err(e) => {
                return Err(anyhow::anyhow!("Error loading tokenizer: {e}"));
            }
        };
        let tokenizer = Arc::new(tokenizer);
        // load json file
        let input = std::fs::read_to_string(&filepath)?;
        let data: Vec<ConversationEntry> = serde_json::from_str(&input).expect("Unable to parse input file. Check that it is valid JSON and matches the expected format.");
        // generate requests
        let requests: Arc<Mutex<Vec<TextGenerationRequest>>> = Arc::from(Mutex::from(Vec::new()));
        info!(
            "Generating requests from {filepath}",
            filepath = filepath.display().to_string()
        );
        let bar = ProgressBar::new(data.len() as u64);
        bar.set_style(ProgressStyle::with_template(
            "Tokenizing prompts [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )?);
        split(data, entry_splitter).for_each(|subrange| {
            for entry in subrange {
                bar.inc(1);
                if entry.conversations.is_empty() {
                    continue;
                }
                let system_prompt = entry
                    .conversations
                    .iter()
                    .find(|c| c.role == "system")
                    .map(|c| c.content.clone());
                let system_prompt_tokens = match system_prompt {
                    Some(ref prompt) => {
                        let (_, num_tokens) = match tokenize_prompt(
                            prompt.clone(),
                            tokenizer.clone(),
                            &TokenizeOptions::default(),
                        ) {
                            Ok((prompt, num_tokens)) => (prompt, num_tokens),
                            Err(e) => {
                                debug!("Error tokenizing system prompt: {e}");
                                return;
                            }
                        };
                        num_tokens
                    }
                    None => 0,
                };
                entry
                    .conversations
                    .iter()
                    .filter(|c| c.role == "user")
                    .for_each(|c| {
                        let prompt = c.content.clone();
                        let num_decode_tokens = decode_tokenize_opts.clone().map_or_else(
                            || None,
                            |opts| {
                                opts.num_tokens.map(|num_tokens| {
                                    sample_num_tokens(
                                        num_tokens,
                                        opts.min_tokens,
                                        opts.max_tokens,
                                        opts.variance,
                                    )
                                })
                            },
                        );
                        match &prompt_tokenize_opts {
                            None => {
                                let (_, num_tokens) = match tokenize_prompt(
                                    prompt.clone(),
                                    tokenizer.clone(),
                                    &TokenizeOptions::default(),
                                ) {
                                    Ok((prompt, num_tokens)) => (prompt, num_tokens),
                                    Err(e) => {
                                        debug!("Error tokenizing prompt: {e}");
                                        return;
                                    }
                                };
                                requests.lock().unwrap().push(TextGenerationRequest {
                                    prompt,
                                    num_prompt_tokens: num_tokens + system_prompt_tokens,
                                    num_decode_tokens,
                                    system_prompt: system_prompt.clone(),
                                });
                            }
                            Some(options) => {
                                // compute number of tokens to generate using a Gaussian distribution
                                let (sampled_prompt, prompt_tokens) = match tokenize_prompt(
                                    prompt.clone(),
                                    tokenizer.clone(),
                                    options,
                                ) {
                                    Ok(prompt) => prompt,
                                    Err(e) => {
                                        debug!("Error tokenizing prompt: {e}");
                                        return;
                                    }
                                };
                                requests.lock().unwrap().push(TextGenerationRequest {
                                    prompt: sampled_prompt,
                                    num_prompt_tokens: prompt_tokens + system_prompt_tokens,
                                    num_decode_tokens,
                                    system_prompt: system_prompt.clone(),
                                });
                            }
                        }
                    });
                // TODO: check that we have enough requests
            }
        });
        let requests = requests.lock().unwrap();
        info!(
            "Generated {num_requests} requests",
            num_requests = requests.len()
        );
        Ok(Self {
            current_index: Arc::from(AtomicI64::new(0)),
            requests: requests.to_vec(),
        })
    }

    pub fn download_dataset(
        repo_name: String,
        filename: String,
        hf_token: Option<String>,
    ) -> anyhow::Result<PathBuf> {
        let api = ApiBuilder::new().with_token(hf_token).build()?;
        let repo = api.dataset(repo_name);
        let dataset = repo.get(&filename)?;
        Ok(dataset)
    }
}

fn sample_num_tokens(num_tokens: u64, min_tokens: u64, max_tokens: u64, variance: u64) -> u64 {
    let normal = rand_distr::Normal::new(num_tokens as f64, variance as f64).unwrap();
    let mut num_tokens = normal.sample(&mut rand::thread_rng()) as u64;
    if num_tokens < min_tokens {
        num_tokens = min_tokens;
    }
    if num_tokens > max_tokens {
        num_tokens = max_tokens;
    }
    num_tokens
}

fn entry_splitter(
    gen: Vec<ConversationEntry>,
) -> (Vec<ConversationEntry>, Option<Vec<ConversationEntry>>) {
    if gen.len() <= 2 {
        return (gen, None);
    }
    let middle = gen.len() / 2;
    let (left, right) = gen.split_at(middle);
    let left = left.to_vec();
    let right = right.to_vec();
    (left, Some(right))
}

impl TextRequestGenerator for ConversationTextRequestGenerator {
    fn generate_request(&mut self) -> TextGenerationRequest {
        let idx = self
            .current_index
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if idx >= (self.requests.len() - 1) as i64 {
            self.current_index
                .store(0, std::sync::atomic::Ordering::SeqCst);
        }
        self.requests[idx as usize].clone()
    }
}

pub struct DummyTextRequestGenerator {}

impl DummyTextRequestGenerator {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DummyTextRequestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TextRequestGenerator for DummyTextRequestGenerator {
    fn generate_request(&mut self) -> TextGenerationRequest {
        TextGenerationRequest {
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(10),
            system_prompt: None,
        }
    }
}

fn tokenize_prompt(
    prompt: String,
    tokenizer: Arc<Tokenizer>,
    options: &TokenizeOptions,
) -> anyhow::Result<(String, u64)> {
    let prompt_tokens = tokenizer
        .encode(prompt.clone(), false)
        .map_err(|_| anyhow::anyhow!("Error tokenizing prompt"))?;
    match options.num_tokens {
        None => {
            // check if we have a min/max number of tokens, skip prompts that are too short or too long
            if prompt_tokens.len() > options.max_tokens as usize
                || prompt_tokens.len() < options.min_tokens as usize
            {
                return Err(anyhow::anyhow!(format!(
                    "Prompt is too short or too long, skipping: {}<{}<{}",
                    options.min_tokens,
                    prompt_tokens.len(),
                    options.max_tokens
                )));
            }
            Ok((prompt, prompt_tokens.len() as u64))
        }
        Some(num_tokens) => {
            if prompt_tokens.len() < num_tokens as usize {
                return Err(anyhow::anyhow!(format!(
                    "Prompt is too short to tokenize: {}<{}",
                    prompt_tokens.len(),
                    num_tokens
                )));
            }
            let tokens = prompt_tokens
                .get_ids()
                .iter()
                .take(num_tokens as usize)
                .copied()
                .collect::<Vec<u32>>();
            let prompt = tokenizer.decode(&tokens, true).unwrap();
            Ok((prompt, num_tokens))
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextGenerationAggregatedResponse {
    pub start_time: Option<tokio::time::Instant>,
    pub end_time: Option<tokio::time::Instant>,
    pub num_generated_tokens: u64,
    pub num_prompt_tokens: u64,
    pub times_to_tokens: Vec<std::time::Duration>,
    last_received_token_time: tokio::time::Instant,
    pub failed: bool,
    pub ended: bool,
}

impl Default for TextGenerationAggregatedResponse {
    fn default() -> Self {
        Self {
            start_time: None,
            end_time: None,
            num_generated_tokens: 0,
            num_prompt_tokens: 0,
            times_to_tokens: Vec::new(),
            last_received_token_time: tokio::time::Instant::now(),
            failed: false,
            ended: false,
        }
    }
}

impl TextGenerationAggregatedResponse {
    pub fn new_as_ended() -> Self {
        Self {
            start_time: None,
            end_time: None,
            num_generated_tokens: 0,
            num_prompt_tokens: 0,
            times_to_tokens: Vec::new(),
            last_received_token_time: tokio::time::Instant::now(),
            failed: false,
            ended: true,
        }
    }
    fn start(&mut self, num_prompt_tokens: u64) {
        self.start_time = Some(tokio::time::Instant::now());
        self.last_received_token_time = tokio::time::Instant::now();
        self.num_prompt_tokens = num_prompt_tokens;
    }

    fn stop(&mut self) {
        self.end_time = Some(tokio::time::Instant::now());
    }

    fn fail(&mut self) {
        self.end_time = Some(tokio::time::Instant::now());
        self.failed = true;
    }

    fn add_tokens(&mut self, num_tokens: u64) {
        self.num_generated_tokens += num_tokens;
        let time_to_generate = self.last_received_token_time.elapsed();
        // make the assumption that when returned simultaneously, tokens were generated at a constant rate
        time_to_generate.checked_div(num_tokens as u32).unwrap();
        self.last_received_token_time = tokio::time::Instant::now();
        self.times_to_tokens.push(time_to_generate);
    }

    pub fn time_to_first_token(&self) -> Option<std::time::Duration> {
        match self.start_time {
            Some(_) => self.times_to_tokens.first().copied(),
            None => None,
        }
    }

    pub fn inter_token_latency(&self) -> Option<std::time::Duration> {
        match self.times_to_tokens.len() {
            0 => None,
            1 => Some(std::time::Duration::new(0, 0)),
            _ => {
                let mut total_time = std::time::Duration::new(0, 0);
                for i in 1..self.times_to_tokens.len() {
                    total_time += self.times_to_tokens[i];
                }
                Some(total_time / (self.num_generated_tokens as u32 - 1))
            }
        }
    }
    pub fn e2e_latency(&self) -> Option<std::time::Duration> {
        match self.start_time {
            Some(start_time) => self.end_time.map(|end_time| end_time - start_time),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executors::ExecutorConfig;
    use crate::results::BenchmarkResults;
    use crate::scheduler::ExecutorType;
    use std::sync::atomic::AtomicU64;
    use std::thread::sleep;
    use std::time::Duration;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn test_openai_token_count() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": {\"content\": \"Hello, world!Hello, world!Hello, world!Hello, world!\", \"role\": \"user\"}, \"finish_reason\": \"stop\", \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(10),
            system_prompt: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx).await;
        });
        let num_tokens = Arc::new(AtomicU64::new(0));
        let num_tokens_clone = num_tokens.clone();
        let t = tokio::spawn(async move {
            while let Some(item) = rx.recv().await {
                let response = item;
                assert_eq!(response.failed, false);
                num_tokens_clone.fetch_add(
                    response.num_generated_tokens,
                    std::sync::atomic::Ordering::SeqCst,
                );
            }
        });
        t.await.unwrap();
        assert_eq!(
            num_tokens.load(std::sync::atomic::Ordering::SeqCst),
            16 as u64
        );
    }

    /// Test that the timings are correct
    /// The tests may be flaky due to the nature of the SSE connection (it may depend on the testing environment)
    /// We need to account for the time it takes to establish the connection
    /// and the time it takes to receive the first message
    #[tokio::test]
    async fn test_openai_timings() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                // sleep for 500ms
                sleep(std::time::Duration::from_millis(500));
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": {\"content\": \"Hello, world!Hello, world!Hello, world!Hello, world!\", \"role\": \"user\"}, \"finish_reason\": \"stop\", \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
            system_prompt: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let results = BenchmarkResults::new(
            "test".to_string(),
            ExecutorType::ConstantArrivalRate,
            ExecutorConfig {
                max_vus: 1,
                duration: Duration::from_secs(10),
                rate: None,
            },
        );
        let results = Arc::new(RwLock::new(results));
        let results_clone = results.clone();
        let t = tokio::spawn(async move {
            let mut handle = results_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.add_response(response);
            }
        });
        t.await.unwrap();
        let results = results.read().await;
        let e2e_latency_avg = results.e2e_latency_avg().unwrap();
        let inter_token_latency_avg = results.inter_token_latency_avg().unwrap();
        let ttft = results.time_to_first_token_avg().unwrap();

        let e2e_timing_overhead = Duration::from_millis(10);
        let expected_e2e_latency_avg = Duration::from_millis(500);
        let expected_inter_token_latency_avg = Duration::from_millis(33); // 16-1 tokens with a 500ms delay
        let inter_token_latency_overhead = Duration::from_millis(3);
        let expected_ttft = Duration::from_millis(3); // account for http overhead
        assert!(
            e2e_latency_avg > expected_e2e_latency_avg
                && e2e_latency_avg < expected_e2e_latency_avg + e2e_timing_overhead,
            "e2e_latency_avg: {:?} < {:?} < {:?}",
            expected_e2e_latency_avg,
            e2e_latency_avg,
            expected_e2e_latency_avg + e2e_timing_overhead
        );
        assert!(
            inter_token_latency_avg > expected_inter_token_latency_avg
                && inter_token_latency_avg
                    < expected_inter_token_latency_avg + inter_token_latency_overhead,
            "inter_token_latency_avg: {:?} < {:?} < {:?}",
            expected_inter_token_latency_avg,
            inter_token_latency_avg,
            expected_inter_token_latency_avg + inter_token_latency_overhead
        );
        assert!(
            ttft < expected_ttft,
            "TTFT: {:?} < {:?}",
            ttft,
            expected_ttft
        );
    }

    /// Test that server errors are handled correctly
    #[tokio::test]
    async fn test_openai_fails_on_error() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| w.write_all(b"data: {\"error\": \"Internal server error\"}\n\n"))
            .create_async()
            .await;
        let url = s.url();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
            system_prompt: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let responses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = responses.clone();
        let t = tokio::spawn(async move {
            let mut handle = responses_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.push(response);
            }
        });
        t.await.unwrap();
        let responses = responses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that bad responses are handled correctly
    #[tokio::test]
    async fn test_openai_fails_on_bad_response() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| w.write_all(b"this is wrong\n\n"))
            .create_async()
            .await;
        let url = s.url();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
            system_prompt: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let responses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = responses.clone();
        let t = tokio::spawn(async move {
            let mut handle = responses_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.push(response);
            }
        });
        t.await.unwrap();
        let responses = responses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that malformed JSON responses are handled correctly
    #[tokio::test]
    async fn test_openai_fails_on_malformed_json() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| w.write_all(b"data: {\"foo\": \"bar\"}\n\n"))
            .create_async()
            .await;
        let url = s.url();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
            system_prompt: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let responses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = responses.clone();
        let t = tokio::spawn(async move {
            let mut handle = responses_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.push(response);
            }
        });
        t.await.unwrap();
        let responses = responses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that request timeout is handled correctly
    #[tokio::test]
    async fn test_timeout_should_fail_request() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                // sleep for 5s
                sleep(std::time::Duration::from_secs(5));
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(1),
        )
        .unwrap();
        let request = TextGenerationRequest {
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
            system_prompt: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let reponses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = reponses.clone();
        let t = tokio::spawn(async move {
            while let Some(item) = rx.recv().await {
                responses_clone.write().await.push(item);
            }
        });
        t.await.unwrap();
        let responses = reponses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that conversations are correctly loaded
    #[tokio::test]
    async fn test_load_conversations_from_file() {
        let filepath = PathBuf::from("test_data/conversations.json");
        let tokenizer = "gpt2".to_string();
        let prompt_tokenize_opts = TokenizeOptions::default();
        let decode_tokenize_opts = TokenizeOptions::default();
        let hf_token = None;
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            tokenizer,
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
            hf_token,
        )
        .unwrap();
        assert_eq!(generator.requests.len(), 17005);
    }

    /// Test that conversations are bounded by the min/max number of tokens
    #[tokio::test]
    async fn test_load_conversations_bounded() {
        let filepath = PathBuf::from("test_data/conversations.json");
        let tokenizer = "gpt2".to_string();
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: None,
            min_tokens: 4,
            max_tokens: 1024,
            variance: 0,
        };
        let decode_tokenize_opts = TokenizeOptions::default();
        let hf_token = None;
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            tokenizer,
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
            hf_token,
        )
        .unwrap();
        let min_tokens = generator
            .requests
            .iter()
            .map(|r| r.num_prompt_tokens)
            .min()
            .unwrap();
        let max_tokens = generator
            .requests
            .iter()
            .map(|r| r.num_prompt_tokens)
            .max()
            .unwrap();
        assert!(min_tokens >= 4, "Min tokens: {}", min_tokens);
        assert!(max_tokens <= 1024, "Max tokens: {}", max_tokens);
    }

    /// Test that conversations prompts have the correct number of tokens
    #[tokio::test]
    async fn test_load_conversations_fixed_tokens() {
        let filepath = PathBuf::from("test_data/conversations.json");
        let tokenizer = "gpt2".to_string();
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: Some(200),
            min_tokens: 200,
            max_tokens: 200,
            variance: 0,
        };
        let decode_tokenize_opts = TokenizeOptions::default();
        let hf_token = None;
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            tokenizer,
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
            hf_token,
        )
        .unwrap();
        for r in generator.requests.iter() {
            assert_eq!(r.num_prompt_tokens, 200);
        }
    }
}
