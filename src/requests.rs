use std::path::PathBuf;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicI64;
use tokio::sync::mpsc::Sender;
use reqwest_eventsource::{Error, Event, EventSource};
use log::{debug, error, info, trace};
use rand_distr::Distribution;
use tokenizers::Tokenizer;
use futures_util::StreamExt;
use hf_hub::api::sync::Api;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::split;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone)]
pub struct TextGenerationRequest {
    pub prompt: String,
    pub num_prompt_tokens: u64, // this includes the system prompt if present
    pub num_decode_tokens: Option<u64>,
    pub system_prompt: Option<String>,
}

#[async_trait]
pub trait TextGenerationBackend: TextGenerationBackendClone {
    async fn generate(&self, request: Arc<TextGenerationRequest>, sender: Sender<TextGenerationAggregatedResponse>);
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
}

impl OpenAITextGenerationBackend {
    pub fn new(api_key: String, base_url: String, model_name: String) -> Self {
        Self {
            api_key,
            base_url,
            model_name,
        }
    }
}

#[async_trait]
impl TextGenerationBackend for OpenAITextGenerationBackend {
    async fn generate(&self, request: Arc<TextGenerationRequest>, sender: Sender<TextGenerationAggregatedResponse>) {
        let url = format!("{base_url}/v1/chat/completions", base_url = self.base_url);
        let mut aggregated_response = TextGenerationAggregatedResponse::default();
        let messages = match &request.system_prompt {
            None => vec![
                OpenAITextGenerationMessage {
                    role: "user".to_string(),
                    content: request.prompt.clone(),
                }
            ],
            Some(system_prompt) => vec![
                OpenAITextGenerationMessage {
                    role: "system".to_string(),
                    content: system_prompt.clone(),
                },
                OpenAITextGenerationMessage {
                    role: "user".to_string(),
                    content: request.prompt.clone(),
                }
            ]
        };
        let body = OpenAITextGenerationRequest {
            model: self.model_name.clone(),
            messages,
            max_tokens: request.num_decode_tokens,
            stream: true,
        };
        let req = reqwest::Client::new().post(url)
            .header("Authorization", format!("Bearer {token}", token = self.api_key))
            .json(&serde_json::json!(body));
        // start timer
        aggregated_response.start(request.num_prompt_tokens);
        let mut es = EventSource::new(req).unwrap();
        let mut final_response = "".to_string();
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => trace!("SSE connection opened"),
                Ok(Event::Message(message)) => {
                    if message.data == "\n" || message.data == "[DONE]" {
                        continue;
                    }
                    if message.data.starts_with("{\"error\":") {
                        error!("Error from OpenAI API: {message}", message = message.data);
                        aggregated_response.fail();
                        es.close();
                        break;
                    }
                    // deserialize message data FIXME: handle JSON errors
                    let oai_response: OpenAITextGenerationResponse = serde_json::from_str(&message.data).unwrap();
                    let choices = oai_response.choices;
                    match choices[0].clone().finish_reason {
                        None => {
                            aggregated_response.add_tokens(1);
                            final_response += &*choices[0].clone().delta.unwrap().content;
                        }
                        Some(_) => {
                            aggregated_response.add_tokens(1);
                            aggregated_response.stop();
                            let content = choices[0].clone().delta.unwrap().content;
                            trace!("Generated text using OpenAI API | prompt: {prompt}, max tokens: {max_tokens:?}, response: {message}", prompt = request.prompt, max_tokens = request.num_decode_tokens,message = &content);
                        }
                    };
                }
                Err(e) => {
                    match e {
                        Error::Utf8(_) => { aggregated_response.fail(); }
                        Error::Parser(_) => { aggregated_response.fail(); }
                        Error::Transport(_) => { aggregated_response.fail(); }
                        Error::InvalidContentType(_, _) => { aggregated_response.fail(); }
                        Error::InvalidStatusCode(_, _) => { aggregated_response.fail(); }
                        Error::InvalidLastEventId(_) => { aggregated_response.fail(); }
                        Error::StreamEnded => {}
                    }
                    es.close();
                }
            };
        };
        sender.send(aggregated_response.clone()).await.expect("Error sending response to channel");
        //debug!("Final response: {response}", response = final_response);
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
    pub num_tokens: u64,
    pub min_tokens: u64,
    pub max_tokens: u64,
    pub variance: u64,
}

impl TokenizeOptions {
    pub fn new() -> Self {
        Self {
            num_tokens: 0,
            min_tokens: 0,
            max_tokens: 0,
            variance: 0,
        }
    }
}

impl ConversationTextRequestGenerator {
    pub fn load(filepath: PathBuf, tokenizer: String, prompt_tokenize_opts: Option<TokenizeOptions>, decode_tokenize_opts: Option<TokenizeOptions>) -> anyhow::Result<Self> {
        let tokenizer = Arc::new(Tokenizer::from_pretrained(tokenizer, None).expect("Unable to load tokenizer"));
        // load json file
        let input = std::fs::read_to_string(&filepath)?;
        let data: Vec<ConversationEntry> = serde_json::from_str(&input).expect("Unable to parse input file. Check that it is valid JSON and matches the expected format.");
        // generate requests
        let requests: Arc<Mutex<Vec<TextGenerationRequest>>> = Arc::from(Mutex::from(Vec::new()));
        info!("Generating requests from {filepath}", filepath = filepath.display().to_string());
        let bar = ProgressBar::new(data.len() as u64);
        bar.set_style(ProgressStyle::with_template("Tokenizing prompts [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap());
        split(data, entry_splitter).for_each(|subrange| {
            for entry in subrange {
                bar.inc(1);
                if entry.conversations.len() == 0 {
                    continue;
                }
                let system_prompt = entry.conversations.iter().find(|c| c.role == "system").map(|c| c.content.clone());
                let system_prompt_tokens = match system_prompt {
                    Some(ref prompt) => {
                        let (_, num_tokens) = match tokenize_prompt(prompt.clone(), tokenizer.clone(), None) {
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
                entry.conversations.iter().filter(|c| c.role == "user").for_each(|c| {
                    let prompt = c.content.clone();
                    let num_decode_tokens = decode_tokenize_opts.clone().map_or_else(|| None, |opts| Some(sample_num_tokens(opts.num_tokens, opts.min_tokens, opts.max_tokens, opts.variance)));
                    match &prompt_tokenize_opts {
                        None => {
                            let (_, num_tokens) = match tokenize_prompt(prompt.clone(), tokenizer.clone(), None) {
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
                            let num_tokens = options.num_tokens;
                            let min_tokens = options.min_tokens;
                            let max_tokens = options.max_tokens;
                            let variance = options.variance;
                            // compute number of tokens to generate using a Gaussian distribution
                            let num_tokens = sample_num_tokens(num_tokens, min_tokens, max_tokens, variance);
                            let sampled_prompt = match tokenize_prompt(prompt.clone(), tokenizer.clone(), Some(num_tokens)) {
                                Ok(prompt) => prompt,
                                Err(e) => {
                                    debug!("Error tokenizing prompt: {e}");
                                    return;
                                }
                            };
                            requests.lock().unwrap().push(TextGenerationRequest {
                                prompt: sampled_prompt.0,
                                num_prompt_tokens: num_tokens + system_prompt_tokens,
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
        info!("Generated {num_requests} requests", num_requests = requests.len());
        Ok(Self {
            current_index: Arc::from(AtomicI64::new(0)),
            requests: requests.to_vec(),
        })
    }

    pub fn download_dataset(repo_name: String, filename: String) -> anyhow::Result<PathBuf> {
        let api = Api::new().unwrap();
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

fn entry_splitter(gen: Vec<ConversationEntry>) -> (Vec<ConversationEntry>, Option<Vec<ConversationEntry>>) {
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
        let idx = self.current_index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if idx >= (self.requests.len() - 1) as i64 {
            self.current_index.store(0, std::sync::atomic::Ordering::SeqCst);
        }
        self.requests[idx as usize].clone()
    }
}


fn tokenize_prompt(prompt: String, tokenizer: Arc<Tokenizer>, num_tokens: Option<u64>) -> anyhow::Result<(String, u64)> {
    let prompt_tokens = tokenizer.encode(prompt.clone(), false).map_err(|_| anyhow::anyhow!("Error tokenizing prompt"))?;
    match num_tokens {
        None => {
            Ok((prompt, prompt_tokens.len() as u64))
        }
        Some(num_tokens) => {
            if prompt_tokens.len() < num_tokens as usize {
                return Err(anyhow::anyhow!("Prompt is too short to tokenize"));
            }
            // let's do a binary search to find the right number of tokens
            let mut low = 1;
            let mut high = prompt.len() as u64;
            let mut prompt_sub = String::new();
            while low < high {
                let mid = (low + high) / 2;
                prompt_sub = prompt.chars().skip((low - 1) as usize).take(high as usize).collect::<String>();
                let tokenized_len = match tokenizer.encode(prompt_sub.clone(), false) {
                    Ok(tokens) => tokens.len(),
                    Err(_) => {
                        return Err(anyhow::anyhow!("Error tokenizing prompt"));
                    }
                };
                if tokenized_len == num_tokens as usize {
                    return Ok((prompt_sub.to_string(), num_tokens));
                } else if tokenized_len > num_tokens as usize {
                    high = mid;
                } else {
                    low = mid + 1;
                }
            }
            Ok((prompt_sub.to_string(), prompt_tokens.len() as u64))
        }
    }
}


#[derive(Debug, Clone)]
pub struct TextGenerationAggregatedResponse {
    pub start_time: Option<std::time::Instant>,
    pub end_time: Option<std::time::Instant>,
    pub num_generated_tokens: u64,
    pub num_prompt_tokens: u64,
    pub times_to_tokens: Vec<std::time::Duration>,
    last_received_token_time: std::time::Instant,
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
            last_received_token_time: std::time::Instant::now(),
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
            last_received_token_time: std::time::Instant::now(),
            failed: false,
            ended: true,
        }
    }
    fn start(&mut self, num_prompt_tokens: u64) {
        self.start_time = Some(std::time::Instant::now());
        self.last_received_token_time = std::time::Instant::now();
        self.num_prompt_tokens = num_prompt_tokens;
    }

    fn stop(&mut self) {
        self.end_time = Some(std::time::Instant::now());
    }

    fn fail(&mut self) {
        self.end_time = Some(std::time::Instant::now());
        self.failed = true;
    }

    fn add_tokens(&mut self, num_tokens: u64) {
        self.num_generated_tokens += num_tokens;
        let time_to_generate = self.last_received_token_time.elapsed();
        self.last_received_token_time = std::time::Instant::now();
        self.times_to_tokens.push(time_to_generate);
    }

    pub fn time_to_first_token(&self) -> Option<std::time::Duration> {
        match self.start_time {
            Some(start_time) => {
                match self.times_to_tokens.first() {
                    Some(time_to_first_token) => {
                        Some(time_to_first_token.clone())
                    }
                    None => {
                        Some(start_time.elapsed())
                    }
                }
            }
            None => {
                None
            }
        }
    }

    pub fn inter_token_latency(&self) -> Option<std::time::Duration> {
        match self.times_to_tokens.len() {
            0 => {
                None
            }
            1 => {
                Some(std::time::Duration::new(0, 0))
            }
            _ => {
                let mut total_time = std::time::Duration::new(0, 0);
                for i in 1..self.times_to_tokens.len() {
                    total_time += self.times_to_tokens[i];
                }
                Some(total_time / (self.times_to_tokens.len() as u32))
            }
        }
    }
    pub fn e2e_latency(&self) -> Option<std::time::Duration> {
        match self.start_time {
            Some(start_time) => {
                match self.end_time {
                    Some(end_time) => {
                        Some(end_time - start_time)
                    }
                    None => {
                        None
                    }
                }
            }
            None => {
                None
            }
        }
    }
}