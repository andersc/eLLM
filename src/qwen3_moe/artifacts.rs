use std::collections::HashMap;
use std::f16;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::memory::model_loader::SafeTensorsLoader;

use super::config::Config;

#[derive(Debug)]
pub struct Qwen36Artifacts {
    pub model_dir: PathBuf,
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub chat_template: String,
    safetensors: SafeTensorsLoader,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatTemplateMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterSpec {
    pub name: String,
    pub elements: usize,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
}

impl Qwen36Artifacts {
    pub fn from_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let config = Config::load_from_file(&config_path)
            .map_err(|error| anyhow!("failed to load {}: {error}", config_path.display()))?;
        validate_qwen36_config(&config)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| anyhow!("failed to load {}: {error}", tokenizer_path.display()))?;
        let chat_template = load_chat_template(&model_dir)?;
        let safetensors = SafeTensorsLoader::new(&model_dir)?;

        Ok(Self {
            model_dir,
            config,
            tokenizer,
            chat_template,
            safetensors,
        })
    }

    pub fn render_chat_prompt(
        &self,
        messages: &[ChatTemplateMessage],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String> {
        let mut env = Environment::new();
        env.add_template("chat", &self.chat_template)
            .context("failed to compile chat template")?;
        let template = env.get_template("chat")?;
        let tools: Vec<Value> = Vec::new();
        template
            .render(context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => add_generation_prompt,
                add_vision_id => false,
                enable_thinking => enable_thinking,
                preserve_thinking => false,
            })
            .context("failed to render chat template")
    }

    pub fn encode_chat(
        &self,
        messages: &[ChatTemplateMessage],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<Vec<u32>> {
        let prompt = self.render_chat_prompt(messages, add_generation_prompt, enable_thinking)?;
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| anyhow!("tokenizer encode failed: {error}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode_tokens(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| anyhow!("tokenizer decode failed: {error}"))
    }

    pub fn load_normalized_weights_f16(&self) -> Result<HashMap<String, Vec<f16>>> {
        let weights = self.safetensors.load_all_weights_f16()?;
        let mut normalized = HashMap::with_capacity(weights.len());

        for (name, value) in weights {
            let normalized_name = normalize_qwen36_weight_name(&name);
            if normalized.insert(normalized_name.clone(), value).is_some() {
                return Err(anyhow!(
                    "duplicate tensor after Qwen3.6 name normalization: {normalized_name}"
                ));
            }
        }

        if self.config.tie_word_embeddings && !normalized.contains_key("lm_head.weight") {
            let embedding = normalized
                .get("model.embed_tokens.weight")
                .ok_or_else(|| anyhow!("missing tied embedding tensor model.embed_tokens.weight"))?
                .clone();
            normalized.insert("lm_head.weight".to_string(), embedding);
        }

        validate_required_parameters(&self.config, &normalized)?;
        Ok(normalized)
    }

    pub fn expected_parameter_specs(&self) -> Vec<ParameterSpec> {
        expected_qwen36_parameter_specs(&self.config)
    }

    pub fn safetensors_files(&self) -> &[String] {
        self.safetensors.model_files()
    }
}

pub fn validate_qwen36_config(config: &Config) -> Result<()> {
    if config.model_type != "qwen3_5_text" && config.model_type != "qwen3_5_moe_text" {
        return Err(anyhow!(
            "unsupported Qwen3.6 text model_type after config normalization: {}",
            config.model_type
        ));
    }
    if config.hidden_size == 0 || config.vocab_size == 0 || config.num_hidden_layers == 0 {
        return Err(anyhow!(
            "invalid Qwen3.6 config: hidden_size, vocab_size, and num_hidden_layers must be non-zero"
        ));
    }
    if config.layer_types.len() != config.num_hidden_layers {
        return Err(anyhow!(
            "invalid Qwen3.6 config: layer_types has {} entries but num_hidden_layers is {}",
            config.layer_types.len(),
            config.num_hidden_layers
        ));
    }
    Ok(())
}

pub fn normalize_qwen36_weight_name(name: &str) -> String {
    let mut normalized = if let Some(rest) = name.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = name.strip_prefix("language_model.") {
        format!("model.{rest}")
    } else {
        name.to_string()
    };

    if normalized.contains(".mlp.experts.down_proj") && !normalized.ends_with(".weight") {
        normalized.push_str(".weight");
    }

    normalized
}

pub fn validate_required_parameters(
    config: &Config,
    weights: &HashMap<String, Vec<f16>>,
) -> Result<()> {
    let specs = expected_qwen36_parameter_specs(config);
    let mut missing = Vec::new();
    let mut wrong_shape = Vec::new();

    for spec in &specs {
        match weights.get(&spec.name) {
            Some(values) if values.len() == spec.elements => {}
            Some(values) => wrong_shape.push(format!(
                "{} has {} elements, expected {}",
                spec.name,
                values.len(),
                spec.elements
            )),
            None => missing.push(spec.name.clone()),
        }
    }

    if !missing.is_empty() || !wrong_shape.is_empty() {
        let mut parts = Vec::new();
        if !missing.is_empty() {
            parts.push(format!("missing: {}", missing.join(", ")));
        }
        if !wrong_shape.is_empty() {
            parts.push(format!("shape mismatches: {}", wrong_shape.join("; ")));
        }
        return Err(anyhow!(
            "Qwen3.6 weights do not match the runtime-required tensor set ({})",
            parts.join("; ")
        ));
    }

    Ok(())
}

pub fn expected_qwen36_parameter_specs(config: &Config) -> Vec<ParameterSpec> {
    let mut specs = Vec::new();
    specs.push(spec(
        "model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
    ));
    specs.push(spec(
        "lm_head.weight",
        &[config.vocab_size, config.hidden_size],
    ));

    for layer_idx in 0..config.num_hidden_layers {
        let layer = format!("model.layers.{layer_idx}");
        match config.layer_type(layer_idx) {
            Some("linear_attention") => {
                let key_dim = config.linear_key_dim();
                let value_dim = config.linear_value_dim();
                let conv_dim = key_dim * 2 + value_dim;
                specs.push(spec(
                    &format!("{layer}.linear_attn.in_proj_qkv.weight"),
                    &[conv_dim, config.hidden_size],
                ));
                specs.push(spec(
                    &format!("{layer}.linear_attn.in_proj_z.weight"),
                    &[value_dim, config.hidden_size],
                ));
                specs.push(spec(
                    &format!("{layer}.linear_attn.in_proj_b.weight"),
                    &[config.linear_num_value_heads, config.hidden_size],
                ));
                specs.push(spec(
                    &format!("{layer}.linear_attn.in_proj_a.weight"),
                    &[config.linear_num_value_heads, config.hidden_size],
                ));
                specs.push(spec(
                    &format!("{layer}.linear_attn.conv1d.weight"),
                    &[conv_dim, 1, config.linear_conv_kernel_dim],
                ));
                specs.push(spec(
                    &format!("{layer}.linear_attn.A_log"),
                    &[config.linear_num_value_heads],
                ));
                specs.push(spec(
                    &format!("{layer}.linear_attn.dt_bias"),
                    &[config.linear_num_value_heads],
                ));
                specs.push(spec(
                    &format!("{layer}.linear_attn.norm.weight"),
                    &[config.linear_value_head_dim],
                ));
                specs.push(spec(
                    &format!("{layer}.linear_attn.out_proj.weight"),
                    &[config.hidden_size, value_dim],
                ));
            }
            Some("full_attention") => {
                specs.push(spec(
                    &format!("{layer}.self_attn.q_proj.weight"),
                    &[
                        config.num_attention_heads * config.head_dim * 2,
                        config.hidden_size,
                    ],
                ));
                specs.push(spec(
                    &format!("{layer}.self_attn.k_proj.weight"),
                    &[
                        config.num_key_value_heads * config.head_dim,
                        config.hidden_size,
                    ],
                ));
                specs.push(spec(
                    &format!("{layer}.self_attn.v_proj.weight"),
                    &[
                        config.num_key_value_heads * config.head_dim,
                        config.hidden_size,
                    ],
                ));
                specs.push(spec(
                    &format!("{layer}.self_attn.o_proj.weight"),
                    &[
                        config.hidden_size,
                        config.num_attention_heads * config.head_dim,
                    ],
                ));
                specs.push(spec(
                    &format!("{layer}.self_attn.q_norm.weight"),
                    &[config.head_dim],
                ));
                specs.push(spec(
                    &format!("{layer}.self_attn.k_norm.weight"),
                    &[config.head_dim],
                ));
            }
            _ => {}
        }

        if config.model_type == "qwen3_5_moe_text"
            && config.num_experts > 0
            && config.num_experts_per_tok > 0
        {
            specs.push(spec(
                &format!("{layer}.mlp.gate.weight"),
                &[config.num_experts, config.hidden_size],
            ));
            specs.push(spec(
                &format!("{layer}.mlp.experts.gate_up_proj"),
                &[
                    config.num_experts,
                    2 * config.moe_intermediate_size,
                    config.hidden_size,
                ],
            ));
            specs.push(spec(
                &format!("{layer}.mlp.experts.down_proj.weight"),
                &[
                    config.num_experts,
                    config.hidden_size,
                    config.moe_intermediate_size,
                ],
            ));
            specs.push(spec(
                &format!("{layer}.mlp.shared_expert.gate_proj.weight"),
                &[config.shared_experts_intermediate_size, config.hidden_size],
            ));
            specs.push(spec(
                &format!("{layer}.mlp.shared_expert.up_proj.weight"),
                &[config.shared_experts_intermediate_size, config.hidden_size],
            ));
            specs.push(spec(
                &format!("{layer}.mlp.shared_expert.down_proj.weight"),
                &[config.hidden_size, config.shared_experts_intermediate_size],
            ));
            specs.push(spec(
                &format!("{layer}.mlp.shared_expert_gate.weight"),
                &[1, config.hidden_size],
            ));
        } else {
            specs.push(spec(
                &format!("{layer}.mlp.gate_proj.weight"),
                &[config.intermediate_size, config.hidden_size],
            ));
            specs.push(spec(
                &format!("{layer}.mlp.up_proj.weight"),
                &[config.intermediate_size, config.hidden_size],
            ));
            specs.push(spec(
                &format!("{layer}.mlp.down_proj.weight"),
                &[config.hidden_size, config.intermediate_size],
            ));
        }
    }

    specs
}

fn spec(name: &str, shape: &[usize]) -> ParameterSpec {
    ParameterSpec {
        name: name.to_string(),
        elements: shape.iter().product(),
    }
}

fn load_chat_template(model_dir: &Path) -> Result<String> {
    let jinja_path = model_dir.join("chat_template.jinja");
    if jinja_path.exists() {
        return std::fs::read_to_string(&jinja_path)
            .with_context(|| format!("failed to read {}", jinja_path.display()));
    }

    let tokenizer_config_path = model_dir.join("tokenizer_config.json");
    let file = File::open(&tokenizer_config_path)
        .with_context(|| format!("failed to open {}", tokenizer_config_path.display()))?;
    let tokenizer_config: TokenizerConfig = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse {}", tokenizer_config_path.display()))?;
    tokenizer_config
        .chat_template
        .ok_or_else(|| anyhow!("tokenizer_config.json does not contain chat_template"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, TensorView};
    use safetensors::Dtype;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn normalizes_official_qwen36_weight_names() {
        assert_eq!(
            normalize_qwen36_weight_name("model.language_model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            normalize_qwen36_weight_name("model.language_model.layers.0.mlp.experts.down_proj"),
            "model.layers.0.mlp.experts.down_proj.weight"
        );
        assert_eq!(
            normalize_qwen36_weight_name("model.language_model.layers.0.mlp.experts.gate_up_proj"),
            "model.layers.0.mlp.experts.gate_up_proj"
        );
    }

    #[test]
    fn loads_config_tokenizer_template_and_normalized_weights() {
        let dir = temp_model_dir("qwen36-artifacts");
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

        let artifacts = Qwen36Artifacts::from_dir(&dir).unwrap();
        let prompt = artifacts
            .render_chat_prompt(
                &[ChatTemplateMessage {
                    role: "user".to_string(),
                    content: Value::String("hello".to_string()),
                }],
                true,
                false,
            )
            .unwrap();
        assert!(prompt.contains("<|im_start|>assistant"));

        let weights = artifacts.load_normalized_weights_f16().unwrap();
        assert!(weights.contains_key("model.embed_tokens.weight"));
        assert!(weights.contains_key("model.layers.0.mlp.experts.down_proj.weight"));

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
                "eos_token_id": 248044,
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
