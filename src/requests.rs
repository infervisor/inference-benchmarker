use std::sync::mpsc::Sender;
use reqwest_eventsource::{Event, EventSource};
use log::{debug, info};
use tokenizers::Tokenizer;
use tokio::fs;

pub(crate) struct TextGenerationRequest {
    pub prompt: String,
    pub max_tokens: u32,
}

pub(crate) struct TextGenerationResponse {
    pub text: String,
    pub response_type: TextGenerationResponseType,
}

pub(crate) enum TextGenerationResponseType {
    Chunk,
    Final,
}

trait TextGenerationBackend {
    async fn generate(&self, request: TextGenerationRequest, sender: Sender<TextGenerationResponse>) -> String;
}

pub(crate) struct OpenAITextGenerationBackend {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
}

#[derive(serde::Deserialize)]
pub(crate) struct OpenAITextGenerationMessage {
    pub(crate) content: String,
    pub(crate) role: String,
}

pub(crate) struct OpenAITextGenerationDelta {
    pub(crate) content: String,
}

#[derive(serde::Deserialize)]
pub(crate) struct OpenAITextGenerationChoice {
    pub(crate) message: Option<OpenAITextGenerationMessage>,
    pub(crate) finish_reason: String,
    pub(crate) delta: Option<OpenAITextGenerationDelta>,
}

#[derive(serde::Deserialize)]
pub(crate) struct OpenAITextGenerationResponse {
    pub(crate) choices: Vec<OpenAITextGenerationChoice>,
}

impl TextGenerationBackend for OpenAITextGenerationBackend {
    async fn generate(&self, request: TextGenerationRequest, sender: Sender<TextGenerationResponse>) {
        let url = format!("{base_url}/v1", base_url = self.base_url).as_str();
        debug!("Requesting {url} with prompt: {prompt}, max tokens: {max_tokens}", prompt = request.prompt, max_tokens = request.max_tokens);
        let mut es = EventSource::get(url).unwrap();
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => info!("Connection opened"),
                Ok(Event::Message(message)) => {
                    // deserialize message data
                    let oai_response: OpenAITextGenerationResponse = serde_json::from_str(&message.data).unwrap();
                    let choices = oai_response.choices;
                    let mut response: TextGenerationResponse;
                    match Some(choices[0].clone().message) {
                        None => {
                            response = TextGenerationResponse {
                                response_type: TextGenerationResponseType::Chunk,
                                text: oai_response.choices[0].clone().delta.unwrap().content,
                            };
                        }
                        Some(message) => {
                            response = TextGenerationResponse {
                                response_type: TextGenerationResponseType::Final,
                                text: message.content,
                            };
                        }
                    };
                    sender.send(response).unwrap();
                }
                Err(e) => {
                    es.close()
                }
            }
            debug!("Generated text using OpenAI API with prompt: {prompt}, max tokens: {max_tokens}", prompt = request.prompt, max_tokens = request.max_tokens)
        }
    }
}

trait TextRequestGenerator {
    fn generate_request(&self, max_tokens: u32) -> TextGenerationRequest;
}

pub(crate) struct ShareGPTTextRequestGenerator {
    pub(crate) filepath: String,
    pub conversations: Vec<ShareGPTEntry>,
    pub current_index: u64,
}

#[derive(serde::Deserialize)]
pub(crate) struct ShareGPTConversation {
    pub(crate) from: String,
    pub(crate) value: String,
}

#[derive(serde::Deserialize)]
pub(crate) struct ShareGPTEntry {
    pub(crate) id: String,
    pub(crate) conversations: Vec<ShareGPTConversation>,
}

impl ShareGPTTextRequestGenerator {
    fn new(filepath: String, tokenizer: String, min_tokens: u32, max_tokens: u32, variance: u32) -> Self {
        let tokenizer = Tokenizer::from_pretrained(tokenizer, None)?;
        // load json file
        let input = std::fs::read_to_string(&filepath).expect("Unable to read input file");
        let data: Vec<ShareGPTEntry> = serde_json::from_str(&input).expect("Unable to parse input file");
        // generate requests
        let mut requests = Vec::new();
        for entry in data.iter() {
            let prompt = entry.conversations[0].value.clone();
            requests.push(TextGenerationRequest {
                prompt,
                max_tokens,
            });
        }
        Self {
            filepath,
            conversations: data,
            current_index: 0,
        }
    }

    fn generate_inputs(&self, min_tokens: u32, max_tokens: u32, variance: u32) -> Vec<TextGenerationRequest> {
        let mut requests = Vec::new();
        for entry in self.conversations.iter() {
            let prompt = entry.conversations[0].value.clone();
            requests.push(TextGenerationRequest {
                prompt,
                max_tokens,
            });
        }
        requests
    }
}

impl Iterator for ShareGPTTextRequestGenerator {
    type Item = TextGenerationRequest;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.conversations.len() as u64 {
            let entry = self.conversations[self.current_index].clone();
            self.current_index += 1;
            let prompt = entry.conversations[0].value.clone();
            let max_tokens = 50;
            Some(entry)
        } else {
            None
        }
    }
}

fn tokenize_prompt(prompt: String, tokenizer: Tokenizer, num_tokens: u32) -> anyhow::Result<String> {
    // let's do a binary search to find the right number of tokens
    let mut low = 0;
    let mut high = prompt.len() as u32;
    while low < high {
        let mid = (low + high) / 2;
        let tokenized = tokenizer.encode(prompt[low..high], false)?;
        if tokenized.len() > num_tokens as usize {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    prompt[0..low].to_string()
}