#![feature(f16)]

use std::f16;
use std::net::SocketAddr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::{
    extract::State,
    http::StatusCode,
    response::sse::Event,
    response::{IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use ellm::memory::allocator::allocate_init;
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

const MODEL_ID: &str = "ellm-qwen36-a3b-smoke";

#[derive(Clone)]
struct AppState {
    model_id: String,
}

#[derive(Debug, Deserialize, Clone)]
struct ChatCompletionRequest {
    #[serde(default = "default_model")]
    model: String,
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
}

#[derive(Debug, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: MessageContent,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<MessageContentPart>),
    Null(()),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Deserialize, Clone)]
struct MessageContentPart {
    text: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChoice {
    index: u32,
    message: ChatCompletionMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct StreamResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
struct StreamChoice {
    index: u32,
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

fn default_model() -> String {
    MODEL_ID.to_string()
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "model": MODEL_ID,
        "note": "OpenAI-compatible eLLM smoke endpoint for coding-agent integration tests"
    }))
}

async fn models(State(state): State<AppState>) -> impl IntoResponse {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_id,
            object: "model".to_string(),
            created: unix_time(),
            owned_by: "ellm".to_string(),
        }],
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if request.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "messages must not be empty"})),
        )
            .into_response();
    }

    let request_id = format!("chatcmpl-{}", unix_time_nanos());
    let is_stream = request.stream.unwrap_or(false);
    let model = if request.model.is_empty() {
        state.model_id.clone()
    } else {
        request.model.clone()
    };
    let _client_options = (request.max_tokens, request.temperature);
    let prompt_tokens = estimate_tokens(&request.messages);
    let runtime_request = request.clone();
    let content = match tokio::task::spawn_blocking(move || run_smoke_completion(&runtime_request))
        .await
    {
        Ok(content) => content,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("eLLM smoke runtime failed: {error}")})),
            )
                .into_response();
        }
    };

    if is_stream {
        let stream_model = model.clone();
        let stream_request_id = request_id.clone();
        let completion_tokens = estimate_text_tokens(&content);
        let chunks = content
            .split_whitespace()
            .map(|word| format!("{word} "))
            .collect::<Vec<_>>();

        let response_stream = stream! {
            let first = StreamResponse {
                id: stream_request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: unix_time(),
                model: stream_model.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            yield Ok::<Event, axum::Error>(Event::default().data(serde_json::to_string(&first).unwrap()));

            for chunk in chunks {
                let response = StreamResponse {
                    id: stream_request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: unix_time(),
                    model: stream_model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: StreamDelta {
                            role: None,
                            content: Some(chunk),
                        },
                        finish_reason: None,
                    }],
                };
                yield Ok(Event::default().data(serde_json::to_string(&response).unwrap()));
            }

            let final_response = StreamResponse {
                id: stream_request_id,
                object: "chat.completion.chunk".to_string(),
                created: unix_time(),
                model: stream_model,
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta { role: None, content: None },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            yield Ok(Event::default().data(serde_json::to_string(&final_response).unwrap()));
            yield Ok(Event::default().data("[DONE]"));
            let _ = completion_tokens;
        };

        Sse::new(response_stream).into_response()
    } else {
        let completion_tokens = estimate_text_tokens(&content);
        Json(ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: unix_time(),
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
        .into_response()
    }
}

fn run_smoke_completion(request: &ChatCompletionRequest) -> String {
    let prompt = request
        .messages
        .iter()
        .map(|message| format!("{}: {}", message.role, message.content.as_text()))
        .collect::<Vec<_>>()
        .join("\n");
    let started = Instant::now();
    let operators = run_qwen36_a3b_runtime_once();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    format!(
        "eLLM local endpoint is running. I executed the Qwen3.6-35B-A3B-shaped CPU runtime smoke path for this request ({operators} operators, {:.2} ms). Prompt chars: {}. This endpoint is OpenAI-compatible for opencode/coding-agent integration tests. Real code-generation quality requires loading official Qwen3.6-35B-A3B weights and tokenizer; this repo currently exposes a validated runtime smoke model.",
        elapsed_ms,
        prompt.len()
    )
}

fn run_qwen36_a3b_runtime_once() -> usize {
    let sequence_length = 4usize;
    let batch_size = 3usize;
    let config = tiny_qwen36_a3b_config();
    let mut model = Model::<f16>::new(&config, sequence_length, sequence_length, batch_size, 4);
    let sequences = allocate_init::<usize>((sequence_length + 1) * batch_size, 0);
    let _ = model.forward(sequences);
    let queue = model.operator_queue.borrow();

    for operator in queue.iter() {
        operator.run(0, sequence_length, batch_size, 1, 0);
    }

    queue.len()
}

fn tiny_qwen36_a3b_config() -> Config {
    Config::from_json_str(
        r#"{
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "eos_token_id": 248044,
                "full_attention_interval": 2,
                "head_dim": 16,
                "hidden_size": 64,
                "intermediate_size": 64,
                "layer_types": ["linear_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_value_head_dim": 16,
                "max_position_embeddings": 128,
                "model_type": "qwen3_5_moe_text",
                "moe_intermediate_size": 64,
                "norm_topk_prob": true,
                "num_attention_heads": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000000,
                "shared_expert_intermediate_size": 64,
                "vocab_size": 64
            }
        }"#,
    )
    .unwrap()
}

fn estimate_tokens(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|message| estimate_text_tokens(&message.content.as_text()) + 4)
        .sum()
}

impl MessageContent {
    fn as_text(&self) -> String {
        match self {
            Self::Text(content) => content.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n"),
            Self::Null(()) => String::new(),
        }
    }
}

fn estimate_text_tokens(content: &str) -> usize {
    content.split_whitespace().count().max(1)
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn unix_time_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = std::env::var("ELLM_AGENT_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("ELLM_AGENT_PORT")
        .ok()
        .and_then(|port| port.parse::<u16>().ok())
        .unwrap_or(8000);
    let addr = format!("{host}:{port}").parse::<SocketAddr>()?;

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(AppState {
            model_id: MODEL_ID.to_string(),
        });

    let listener = TcpListener::bind(addr).await?;
    println!("eLLM OpenAI-compatible agent endpoint listening on http://{addr}");
    println!("base_url: http://{addr}/v1");
    println!("model: {MODEL_ID}");

    axum::serve(listener, app).await?;
    Ok(())
}
