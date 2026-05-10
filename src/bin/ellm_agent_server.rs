#![feature(f16)]

use std::f16;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use async_stream::stream;
use axum::{
    extract::State,
    http::StatusCode,
    response::sse::Event,
    response::{IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use ellm::compiler::operator::Operator;
use ellm::memory::allocator::allocate_init;
use ellm::qwen3_moe::artifacts::{ChatTemplateMessage, Qwen36Artifacts};
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::net::TcpListener;

const MODEL_ID: &str = "ellm-qwen36-a3b-smoke";
const OFFICIAL_MODEL_ID: &str = "Qwen/Qwen3.6-35B-A3B";

#[derive(Clone)]
struct AppState {
    model_id: String,
    backend: Arc<Backend>,
}

enum Backend {
    Smoke,
    Official(Mutex<OfficialRuntime>),
}

struct OfficialRuntime {
    artifacts: Qwen36Artifacts,
    operator_queue: Vec<Operator<f16>>,
    sequences: *mut usize,
    max_context: usize,
    max_generation_tokens: usize,
}

unsafe impl Send for OfficialRuntime {}

struct CompletionResult {
    content: String,
    prompt_tokens: usize,
    completion_tokens: usize,
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

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "model": state.model_id,
        "mode": state.backend.mode(),
        "note": state.backend.status_note()
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
    let backend = state.backend.clone();
    let runtime_request = request.clone();
    let completion = match tokio::task::spawn_blocking(move || backend.complete(&runtime_request))
        .await
    {
        Ok(Ok(completion)) => completion,
        Ok(Err(error)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("eLLM runtime failed: {error}")})),
            )
                .into_response();
        }
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("eLLM runtime task failed: {error}")})),
            )
                .into_response();
        }
    };
    let content = completion.content;
    let prompt_tokens = completion.prompt_tokens;

    if is_stream {
        let stream_model = model.clone();
        let stream_request_id = request_id.clone();
        let completion_tokens = completion.completion_tokens;
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
        let completion_tokens = completion.completion_tokens;
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

impl Backend {
    fn mode(&self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Official(_) => "official-qwen3.6",
        }
    }

    fn status_note(&self) -> String {
        match self {
            Self::Smoke => {
                "OpenAI-compatible eLLM smoke endpoint for coding-agent integration tests"
                    .to_string()
            }
            Self::Official(runtime) => {
                let runtime = runtime.lock().unwrap();
                format!(
                    "Official Qwen3.6 artifacts loaded from {} with {} operators and max_context {}",
                    runtime.artifacts.model_dir.display(),
                    runtime.operator_queue.len(),
                    runtime.max_context
                )
            }
        }
    }

    fn complete(&self, request: &ChatCompletionRequest) -> Result<CompletionResult> {
        match self {
            Self::Smoke => {
                let content = run_smoke_completion(request);
                Ok(CompletionResult {
                    prompt_tokens: estimate_tokens(&request.messages),
                    completion_tokens: estimate_text_tokens(&content),
                    content,
                })
            }
            Self::Official(runtime) => runtime.lock().unwrap().complete(request),
        }
    }
}

impl OfficialRuntime {
    fn from_dir(
        model_dir: &Path,
        max_context: usize,
        topk_size: usize,
        max_generation_tokens: usize,
    ) -> Result<Self> {
        let artifacts = Qwen36Artifacts::from_dir(model_dir)?;
        let max_context = max_context
            .max(1)
            .min(artifacts.config.max_position_embeddings.max(1));
        let weights = artifacts.load_normalized_weights_f16()?;
        let mut model = Model::<f16>::new_with_parameters(
            &artifacts.config,
            max_context,
            max_context,
            1,
            topk_size.max(1),
            weights,
        );
        let sequences = allocate_init::<usize>(max_context + 1, 0);
        let _ = model.forward(sequences);
        let operator_queue = model.operator_queue.take();

        Ok(Self {
            artifacts,
            operator_queue,
            sequences,
            max_context,
            max_generation_tokens: max_generation_tokens.max(1),
        })
    }

    fn complete(&mut self, request: &ChatCompletionRequest) -> Result<CompletionResult> {
        let messages = request
            .messages
            .iter()
            .map(|message| ChatTemplateMessage {
                role: message.role.clone(),
                content: message.content.as_value(),
            })
            .collect::<Vec<_>>();
        let encoded = self.artifacts.encode_chat(&messages, true, false)?;
        if encoded.is_empty() {
            return Err(anyhow!("tokenizer produced an empty prompt"));
        }

        let prompt_tokens = encoded.len();
        let start = encoded.len().saturating_sub(self.max_context);
        let prompt = &encoded[start..];
        let max_new_tokens = request
            .max_tokens
            .unwrap_or(1)
            .max(1)
            .min(self.max_generation_tokens)
            .min(self.max_context.saturating_sub(prompt.len()).max(1));
        let generated = self.generate(prompt, max_new_tokens)?;
        let content = if generated.is_empty() {
            String::new()
        } else {
            self.artifacts.decode_tokens(&generated, true)?
        };

        Ok(CompletionResult {
            content,
            prompt_tokens,
            completion_tokens: generated.len(),
        })
    }

    fn generate(&mut self, prompt: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(anyhow!("prompt must contain at least one token"));
        }
        if prompt.len() > self.max_context {
            return Err(anyhow!(
                "prompt length {} exceeds max_context {}",
                prompt.len(),
                self.max_context
            ));
        }

        unsafe {
            for offset in 0..=self.max_context {
                self.sequences.add(offset).write(0);
            }
            for (offset, token_id) in prompt.iter().enumerate() {
                let token_id = *token_id as usize;
                if token_id >= self.artifacts.config.vocab_size {
                    return Err(anyhow!(
                        "token id {token_id} is outside configured vocab_size {}",
                        self.artifacts.config.vocab_size
                    ));
                }
                self.sequences.add(offset).write(token_id);
            }
        }

        for position in 0..prompt.len() {
            for operator in &self.operator_queue {
                operator.run(position, 1, 1, 1, 0);
            }
            if position + 1 < prompt.len() {
                unsafe {
                    self.sequences
                        .add(position + 1)
                        .write(prompt[position + 1] as usize);
                }
            }
        }

        let mut generated = Vec::new();
        for _ in 0..max_new_tokens {
            let next_position = prompt.len() + generated.len();
            if next_position >= self.max_context {
                break;
            }

            let next_token = unsafe { *self.sequences.add(next_position) as u32 };
            if next_token as usize >= self.artifacts.config.vocab_size {
                return Err(anyhow!(
                    "generated token id {next_token} is outside configured vocab_size {}",
                    self.artifacts.config.vocab_size
                ));
            }
            generated.push(next_token);
            if next_token as usize == self.artifacts.config.eos_token_id {
                break;
            }

            let position = next_position;
            if position >= self.max_context {
                break;
            }
            for operator in &self.operator_queue {
                operator.run(position, 1, 1, 1, 0);
            }
        }

        Ok(generated)
    }
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

    fn as_value(&self) -> Value {
        match self {
            Self::Text(content) => Value::String(content.clone()),
            Self::Parts(parts) => Value::Array(
                parts
                    .iter()
                    .filter_map(|part| {
                        part.text
                            .as_ref()
                            .map(|text| serde_json::json!({"type": "text", "text": text}))
                    })
                    .collect(),
            ),
            Self::Null(()) => Value::Null,
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
    let (model_id, backend) = load_backend()?;

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(AppState {
            model_id: model_id.clone(),
            backend,
        });

    let listener = TcpListener::bind(addr).await?;
    println!("eLLM OpenAI-compatible agent endpoint listening on http://{addr}");
    println!("base_url: http://{addr}/v1");
    println!("model: {model_id}");

    axum::serve(listener, app).await?;
    Ok(())
}

fn load_backend() -> Result<(String, Arc<Backend>)> {
    let Some(model_dir) = std::env::var_os("ELLM_MODEL_DIR") else {
        return Ok((MODEL_ID.to_string(), Arc::new(Backend::Smoke)));
    };

    let model_dir = PathBuf::from(model_dir);
    let max_context = read_env_usize("ELLM_MAX_CONTEXT").unwrap_or(128);
    let topk_size = read_env_usize("ELLM_TOPK").unwrap_or(8);
    let max_generation_tokens = read_env_usize("ELLM_MAX_GENERATION_TOKENS").unwrap_or(16);
    let runtime =
        OfficialRuntime::from_dir(&model_dir, max_context, topk_size, max_generation_tokens)?;
    let model_id = std::env::var("ELLM_MODEL_ID").unwrap_or_else(|_| OFFICIAL_MODEL_ID.to_string());
    Ok((model_id, Arc::new(Backend::Official(Mutex::new(runtime)))))
}

fn read_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ellm::qwen3_moe::artifacts::expected_qwen36_parameter_specs;
    use safetensors::tensor::{serialize_to_file, TensorView};
    use safetensors::Dtype;
    use std::collections::HashMap;

    #[test]
    fn official_runtime_loads_tiny_artifacts_and_completes() {
        let dir = temp_model_dir("endpoint-official");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), tiny_official_config_json()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tiny_tokenizer_json()).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "{%- for message in messages %}{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}{%- endfor %}{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}"
            })
            .to_string(),
        )
        .unwrap();

        let config = Config::load_from_file(dir.join("config.json")).unwrap();
        let specs = expected_qwen36_parameter_specs(&config);
        let tensor_storage = specs
            .iter()
            .map(|spec| {
                let official_name = to_official_test_name(&spec.name);
                let bytes = vec![0u8; spec.elements * 4];
                (official_name, vec![spec.elements], bytes)
            })
            .collect::<Vec<_>>();
        let tensor_views = tensor_storage
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            tensor_views,
            &None,
            &dir.join("model-00001-of-00001.safetensors"),
        )
        .unwrap();
        let weight_map = tensor_storage
            .iter()
            .map(|(name, _, _)| (name, "model-00001-of-00001.safetensors"))
            .collect::<HashMap<_, _>>();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({
                "metadata": {"total_size": 0},
                "weight_map": weight_map
            })
            .to_string(),
        )
        .unwrap();

        let mut runtime = OfficialRuntime::from_dir(&dir, 64, 4, 1).unwrap();
        let response = runtime
            .complete(&ChatCompletionRequest {
                model: OFFICIAL_MODEL_ID.to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text("hello".to_string()),
                }],
                stream: None,
                max_tokens: Some(1),
                temperature: None,
            })
            .unwrap();

        assert!(response.prompt_tokens > 0);
        assert_eq!(response.completion_tokens, 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    fn to_official_test_name(name: &str) -> String {
        let prefixed = if let Some(rest) = name.strip_prefix("model.") {
            format!("model.language_model.{rest}")
        } else {
            name.to_string()
        };
        prefixed
            .strip_suffix(".mlp.experts.down_proj.weight")
            .map(|base| format!("{base}.mlp.experts.down_proj"))
            .unwrap_or(prefixed)
    }

    fn tiny_official_config_json() -> &'static str {
        r#"{
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "dtype": "bfloat16",
                "eos_token_id": 2,
                "full_attention_interval": 4,
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
                "num_attention_heads": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default"
                },
                "shared_expert_intermediate_size": 64,
                "tie_word_embeddings": false,
                "vocab_size": 64
            },
            "tie_word_embeddings": false,
            "transformers_version": "4.57.1"
        }"#
    }

    fn tiny_tokenizer_json() -> &'static str {
        r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0,
                    "<|im_start|>": 1,
                    "<|im_end|>": 2,
                    "assistant": 3,
                    "hello": 4,
                    "user": 5
                },
                "unk_token": "<unk>"
            }
        }"#
    }

    fn temp_model_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ellm-{name}-{nanos}"))
    }
}
