use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::AtomicI64;
use tokio::sync::mpsc::Sender;
use reqwest_eventsource::{Event, EventSource};
use log::{debug, info, trace};
use rand_distr::Distribution;
use tokenizers::Tokenizer;
use tokio::fs;
use futures_util::StreamExt;

#[derive(Debug, Clone)]
pub(crate) struct TextGenerationRequest {
    pub prompt: String,
    pub max_tokens: u32,
}

#[derive(Debug)]
pub(crate) struct TextGenerationResponse {
    pub text: String,
    pub response_type: TextGenerationResponseType,
}

#[derive(Debug)]
pub(crate) enum TextGenerationResponseType {
    Chunk,
    Final,
}

#[async_trait]
pub(crate) trait TextGenerationBackend:TextGenerationBackendClone {
    async fn generate(&self, request: Arc<TextGenerationRequest>, sender: Sender<TextGenerationResponse>);
}

pub trait TextGenerationBackendClone{
    fn clone_box(&self) -> Box<dyn TextGenerationBackend+Send+Sync>;
}

impl<T> TextGenerationBackendClone for T
where T: 'static + TextGenerationBackend + Clone + Send + Sync {
    fn clone_box(&self) -> Box<dyn TextGenerationBackend+Send+Sync> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn TextGenerationBackend+Send+Sync> {
    fn clone(&self) -> Box<dyn TextGenerationBackend+Send+Sync> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct OpenAITextGenerationBackend {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
}

#[derive(serde::Deserialize, Clone)]
pub(crate) struct OpenAITextGenerationMessage {
    pub(crate) content: String,
    pub(crate) role: String,
}

#[derive(serde::Deserialize, Clone)]
pub(crate) struct OpenAITextGenerationDelta {
    pub(crate) content: String,
}

#[derive(serde::Deserialize, Clone)]
pub(crate) struct OpenAITextGenerationChoice {
    pub(crate) message: Option<OpenAITextGenerationMessage>,
    pub(crate) finish_reason: Option<String>,
    pub(crate) delta: Option<OpenAITextGenerationDelta>,
}

#[derive(serde::Deserialize, Clone)]
pub(crate) struct OpenAITextGenerationResponse {
    pub(crate) choices: Vec<OpenAITextGenerationChoice>,
}

impl OpenAITextGenerationBackend {
    pub(crate) fn new(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
        }
    }
}

#[async_trait]
impl TextGenerationBackend for OpenAITextGenerationBackend {
    async fn generate(&self, request: Arc<TextGenerationRequest>, sender: Sender<TextGenerationResponse>) {
        let url = format!("{base_url}/v1/chat/completions", base_url = self.base_url);
        debug!("Requesting {url} with prompt: {prompt}, max tokens: {max_tokens}", prompt = request.prompt, max_tokens = request.max_tokens);
        let req=reqwest::Client::new().post(url)
            .header("Authorization", format!("Bearer {token}", token = self.api_key))
            .json(&serde_json::json!({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ],
                "max_tokens": request.max_tokens,
                "temperature": 0.7,
                "stop": ["\n"],
                "stream": true,
            }));
        let mut es = EventSource::new(req).unwrap();
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => info!("Connection opened"),
                Ok(Event::Message(message)) => {
                    // deserialize message data FIXME: handle JSON errors
                    let oai_response: OpenAITextGenerationResponse = serde_json::from_str(&message.data).unwrap();
                    let choices = oai_response.choices;
                    let mut response: TextGenerationResponse;
                    match choices[0].clone().message {
                        None => {
                            response = TextGenerationResponse {
                                response_type: TextGenerationResponseType::Chunk,
                                text: choices[0].clone().delta.unwrap().content,
                            };
                        }
                        Some(message) => {
                            response = TextGenerationResponse {
                                response_type: TextGenerationResponseType::Final,
                                text: message.content,
                            };
                            trace!("Generated text using OpenAI API | prompt: {prompt}, max tokens: {max_tokens}, response: {message}", prompt = request.prompt, max_tokens = request.max_tokens,message = &response.text);
                        }
                    };
                    sender.send(response).await.unwrap();
                }
                Err(e) => {
                    es.close();
                }
            }
        }
    }
}

pub(crate) trait TextRequestGenerator {
    fn generate_request(&mut self) -> TextGenerationRequest;
}

#[derive(Clone)]
pub(crate) struct ShareGPTTextRequestGenerator {
    pub(crate) filepath: String,
    pub conversations: Vec<ShareGPTEntry>,
    pub requests: Vec<TextGenerationRequest>,
    current_index: Arc<AtomicI64>,
}

#[derive(serde::Deserialize,Clone)]
pub(crate) struct ShareGPTConversation {
    pub(crate) from: String,
    pub(crate) value: String,
}

#[derive(serde::Deserialize,Clone)]
pub(crate) struct ShareGPTEntry {
    pub(crate) id: String,
    pub(crate) conversations: Vec<ShareGPTConversation>,
}

impl ShareGPTTextRequestGenerator {
    pub(crate) fn new(filepath: String, tokenizer: String, prompt_tokens: u32, min_tokens: u32, max_tokens: u32, variance: u32) -> Self {
        let tokenizer = Arc::new(Tokenizer::from_pretrained(tokenizer, None).expect("Unable to load tokenizer"));
        // load json file
        let input = std::fs::read_to_string(&filepath).expect("Unable to read input file");
        let data: Vec<ShareGPTEntry> = serde_json::from_str(&input).expect("Unable to parse input file");
        // generate requests
        let mut requests = Vec::new();
        info!("Generating requests from {filepath}", filepath = filepath);
        for entry in data.iter() {
            if entry.conversations.len() == 0 {
                continue;
            }
            let prompt = entry.conversations[0].value.clone();
            // compute number of tokens to generate using a Gaussian distribution
            let normal = rand_distr::Normal::new(prompt_tokens as f64, variance as f64).unwrap();
            let mut num_tokens = normal.sample(&mut rand::thread_rng()) as u32;
            if num_tokens < min_tokens {
                num_tokens = min_tokens;
            }
            if num_tokens > max_tokens {
                num_tokens = max_tokens;
            }
            let sampled_prompt = match tokenize_prompt(prompt, tokenizer.clone(), num_tokens) {
                Ok(prompt) => prompt,
                Err(e) => {
                    debug!("Error tokenizing prompt: {e}");
                    continue;
                }
            };
            requests.push(TextGenerationRequest {
                prompt: sampled_prompt,
                max_tokens,
            });
            // TODO: check that we have enough requests
        }
        info!("Generated {num_requests} requests", num_requests = requests.len());
        Self {
            conversations: data,
            current_index: Arc::from(AtomicI64::new(0)),
            filepath,
            requests,
        }
    }
}

impl TextRequestGenerator for ShareGPTTextRequestGenerator {
    fn generate_request(&mut self) -> TextGenerationRequest {
        let idx = self.current_index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if idx >= (self.requests.len() - 1) as i64 {
            self.current_index.store(0, std::sync::atomic::Ordering::SeqCst);
        }
        self.requests[idx as usize].clone()
    }
}


fn tokenize_prompt(prompt: String, tokenizer: Arc<Tokenizer>, num_tokens: u32) -> anyhow::Result<String> {
    let prompt_tokens=tokenizer.encode(prompt.clone(), false).map_err(|_| anyhow::anyhow!("Error tokenizing prompt"))?;
    if prompt_tokens.len() < num_tokens as usize {
        return Err(anyhow::anyhow!("Prompt is too short to tokenize"));
    }
    // let's do a binary search to find the right number of tokens
    let mut low = 1;
    let mut high = prompt.len() as u32;
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
            return Ok(prompt_sub.to_string());
        } else if tokenized_len > num_tokens as usize {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Ok(prompt_sub.to_string())
}