use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Config {
    pub architectures: Vec<String>,
    pub attention_dropout: f32,
    pub attn_output_gate: bool,
    pub decoder_sparse_step: usize,
    #[serde(default, deserialize_with = "deserialize_token_id")]
    pub eos_token_id: usize,
    pub full_attention_interval: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub mlp_only_layers: Vec<usize>,
    pub model_type: String,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub output_router_logits: bool,
    pub qkv_bias: bool,
    pub partial_rotary_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_scaling: Option<HashMap<String, Value>>,
    pub rope_theta: f64,
    pub router_aux_loss_coef: f32,
    pub shared_experts_intermediate_size: usize,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub use_qk_norm: bool,
    pub use_sliding_window: bool,
    pub vocab_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            architectures: Vec::new(),
            attention_dropout: 0.0,
            attn_output_gate: false,
            decoder_sparse_step: 1,
            eos_token_id: 0,
            full_attention_interval: 0,
            head_dim: 0,
            hidden_act: "silu".to_string(),
            hidden_size: 0,
            initializer_range: 0.02,
            intermediate_size: 0,
            layer_types: Vec::new(),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            max_position_embeddings: 0,
            max_window_layers: 0,
            mlp_only_layers: Vec::new(),
            model_type: String::new(),
            moe_intermediate_size: 0,
            norm_topk_prob: true,
            num_attention_heads: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            num_hidden_layers: 0,
            num_key_value_heads: 0,
            output_router_logits: false,
            qkv_bias: false,
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            rope_scaling: None,
            rope_theta: 1_000_000.0,
            router_aux_loss_coef: 0.0,
            shared_experts_intermediate_size: 0,
            sliding_window: None,
            tie_word_embeddings: false,
            torch_dtype: "bfloat16".to_string(),
            transformers_version: String::new(),
            use_cache: true,
            use_qk_norm: false,
            use_sliding_window: false,
            vocab_size: 0,
        }
    }
}

impl Config {
    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let value: Value = serde_json::from_reader(reader)?;
        Self::from_json_value(value)
    }

    pub fn from_json_str(contents: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let value: Value = serde_json::from_str(contents)?;
        Self::from_json_value(value)
    }

    fn from_json_value(mut value: Value) -> Result<Self, Box<dyn std::error::Error>> {
        value = flatten_text_config(value);
        normalize_qwen_aliases(&mut value);

        let mut config: Config = serde_json::from_value(value)?;
        config.normalize();
        Ok(config)
    }

    fn normalize(&mut self) {
        if self.head_dim == 0 && self.num_attention_heads > 0 {
            self.head_dim = self.hidden_size / self.num_attention_heads;
        }
        if self.num_key_value_heads == 0 {
            self.num_key_value_heads = self.num_attention_heads;
        }
        if self.max_window_layers == 0 {
            self.max_window_layers = self.num_hidden_layers;
        }
        if self.decoder_sparse_step == 0 {
            self.decoder_sparse_step = 1;
        }
        if self.full_attention_interval == 0 {
            self.full_attention_interval = 1;
        }
        if self.layer_types.is_empty()
            && (self.model_type == "qwen3_5_text" || self.model_type == "qwen3_5_moe_text")
        {
            self.layer_types = (0..self.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % self.full_attention_interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect();
        }
        if self.num_experts == 0 {
            self.num_experts_per_tok = 0;
            self.moe_intermediate_size = 0;
        } else if self.moe_intermediate_size == 0 {
            self.moe_intermediate_size = self.intermediate_size;
        }
        if self.intermediate_size == 0 {
            self.intermediate_size = self.shared_experts_intermediate_size;
        }
        if self.intermediate_size == 0 {
            self.intermediate_size = self.moe_intermediate_size;
        }
    }

    pub fn layer_type(&self, layer_idx: usize) -> Option<&str> {
        self.layer_types.get(layer_idx).map(String::as_str)
    }

    pub fn has_qwen36_linear_attention(&self) -> bool {
        self.layer_types.iter().any(|layer_type| layer_type == "linear_attention")
    }

    pub fn unsupported_runtime_reason(&self) -> Option<&'static str> {
        if self.model_type == "qwen3_5_moe_text" {
            Some(
                "Qwen3.6 MoE shared-expert execution is not implemented in eLLM yet; dense Qwen3.6 text runtime is supported.",
            )
        } else {
            None
        }
    }

    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
}

fn flatten_text_config(value: Value) -> Value {
    let mut root = match value {
        Value::Object(root) => root,
        other => return other,
    };

    let Some(Value::Object(mut text_config)) = root.remove("text_config") else {
        return Value::Object(root);
    };

    for key in ["architectures", "tie_word_embeddings", "transformers_version"] {
        if !text_config.contains_key(key) {
            if let Some(value) = root.get(key) {
                text_config.insert(key.to_string(), value.clone());
            }
        }
    }

    Value::Object(text_config)
}

fn normalize_qwen_aliases(value: &mut Value) {
    let Some(object) = value.as_object_mut() else {
        return;
    };

    copy_alias(object, "moe_num_experts", "num_experts");
    copy_alias(object, "moe_top_k", "num_experts_per_tok");
    copy_alias(
        object,
        "shared_expert_intermediate_size",
        "shared_experts_intermediate_size",
    );
    copy_alias(object, "dtype", "torch_dtype");
    copy_alias(object, "layer_norm_eps", "rms_norm_eps");

    let rope_parameters = object.get("rope_parameters").and_then(Value::as_object).cloned();
    if let Some(rope_parameters) = rope_parameters {
        if !object.contains_key("rope_theta") {
            if let Some(rope_theta) = rope_parameters.get("rope_theta") {
                object.insert("rope_theta".to_string(), rope_theta.clone());
            }
        }
        if !object.contains_key("rope_scaling") {
            object.insert(
                "rope_scaling".to_string(),
                Value::Object(rope_parameters.clone()),
            );
        }
        if !object.contains_key("partial_rotary_factor") {
            if let Some(partial_rotary_factor) = rope_parameters.get("partial_rotary_factor") {
                object.insert(
                    "partial_rotary_factor".to_string(),
                    partial_rotary_factor.clone(),
                );
            }
        }
    }
}

fn copy_alias(object: &mut serde_json::Map<String, Value>, source: &str, target: &str) {
    if object.contains_key(target) {
        return;
    }
    if let Some(value) = object.get(source) {
        object.insert(target.to_string(), value.clone());
    }
}

fn deserialize_token_id<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Number(n) => n
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| serde::de::Error::custom("token id must be a non-negative integer")),
        Value::Array(values) => values
            .into_iter()
            .find_map(|value| value.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| serde::de::Error::custom("token id array must contain an integer")),
        Value::Null => Ok(0),
        _ => Err(serde::de::Error::custom("token id must be an integer or integer array")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_file() {
        let config = Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json");
        match config {
            Ok(cfg) => println!("{:?}", cfg),
            Err(e) => println!("Error loading config: {}", e),
        }
    }

    #[test]
    fn test_legacy_qwen3_dense_config_defaults() {
        let config = Config::from_json_str(
            r#"{
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_attention_heads": 32,
                "num_hidden_layers": 2,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
                "eos_token_id": [151645, 151643],
                "rms_norm_eps": 1e-6
            }"#,
        )
        .unwrap();

        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_key_value_heads, 32);
        assert_eq!(config.num_experts, 0);
        assert_eq!(config.num_experts_per_tok, 0);
        assert_eq!(config.eos_token_id, 151645);
    }

    #[test]
    fn test_qwen36_moe_text_config_is_flattened() {
        let config = Config::from_json_str(
            r#"{
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "image_token_id": 248056,
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "attention_dropout": 0.0,
                    "attn_output_gate": true,
                    "dtype": "bfloat16",
                    "eos_token_id": 248044,
                    "full_attention_interval": 4,
                    "head_dim": 256,
                    "hidden_act": "silu",
                    "hidden_size": 2048,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention"
                    ],
                    "max_position_embeddings": 262144,
                    "model_type": "qwen3_5_moe_text",
                    "moe_intermediate_size": 512,
                    "num_attention_heads": 16,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "num_hidden_layers": 40,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "mrope_interleaved": true,
                        "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000,
                        "rope_type": "default"
                    },
                    "router_aux_loss_coef": 0.001,
                    "shared_expert_intermediate_size": 512,
                    "tie_word_embeddings": false,
                    "use_cache": true,
                    "vocab_size": 248320
                },
                "transformers_version": "4.57.1"
            }"#,
        )
        .unwrap();

        assert_eq!(config.architectures, vec!["Qwen3_5MoeForConditionalGeneration"]);
        assert_eq!(config.model_type, "qwen3_5_moe_text");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.num_hidden_layers, 40);
        assert_eq!(config.num_experts, 256);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.moe_intermediate_size, 512);
        assert_eq!(config.shared_experts_intermediate_size, 512);
        assert_eq!(config.intermediate_size, 512);
        assert_eq!(config.linear_conv_kernel_dim, 4);
        assert_eq!(config.linear_key_head_dim, 128);
        assert_eq!(config.linear_value_head_dim, 128);
        assert_eq!(config.linear_num_key_heads, 16);
        assert_eq!(config.linear_num_value_heads, 32);
        assert_eq!(config.eos_token_id, 248044);
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert_eq!(config.partial_rotary_factor, 0.25);
        assert!(config.attn_output_gate);
        assert!(config.has_qwen36_linear_attention());
    }

    #[test]
    fn test_qwen36_dense_text_config_is_flattened() {
        let config = Config::from_json_str(
            r#"{
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "attention_dropout": 0.0,
                    "attn_output_gate": true,
                    "dtype": "bfloat16",
                    "eos_token_id": 248044,
                    "full_attention_interval": 4,
                    "head_dim": 256,
                    "hidden_act": "silu",
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention"
                    ],
                    "max_position_embeddings": 262144,
                    "model_type": "qwen3_5_text",
                    "num_attention_heads": 24,
                    "num_hidden_layers": 64,
                    "num_key_value_heads": 4,
                    "rms_norm_eps": 1e-6,
                    "tie_word_embeddings": false,
                    "use_cache": true,
                    "vocab_size": 248320
                },
                "transformers_version": "4.57.1"
            }"#,
        )
        .unwrap();

        assert_eq!(config.architectures, vec!["Qwen3_5ForConditionalGeneration"]);
        assert_eq!(config.model_type, "qwen3_5_text");
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.intermediate_size, 17408);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.linear_conv_kernel_dim, 4);
        assert_eq!(config.linear_key_head_dim, 128);
        assert_eq!(config.linear_value_head_dim, 128);
        assert_eq!(config.linear_num_key_heads, 16);
        assert_eq!(config.linear_num_value_heads, 32);
        assert_eq!(config.num_experts, 0);
        assert_eq!(config.num_experts_per_tok, 0);
        assert_eq!(config.eos_token_id, 248044);
        assert!(config.has_qwen36_linear_attention());
    }
}
