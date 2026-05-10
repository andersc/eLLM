#![feature(f16)]

use std::collections::HashMap;
use std::f16;
use std::time::Instant;

use ellm::memory::allocator::allocate_init;
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;

fn main() {
    let sequence_length = 16usize;
    let batch_size = 3usize;
    let iterations = std::env::var("QWEN36_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);

    let config = tiny_qwen36_config();
    let parameters = synthetic_parameters(&config);
    let mut model = Model::<f16>::new_with_parameters(
        &config,
        sequence_length,
        sequence_length,
        batch_size,
        4,
        parameters,
    );

    let sequences = allocate_init::<usize>((sequence_length + 1) * batch_size, 0);
    let _ = model.forward(sequences);
    let queue = model.operator_queue.borrow();

    let started = Instant::now();
    for _ in 0..iterations {
        for operator in queue.iter() {
            operator.run(0, sequence_length, batch_size, 1, 0);
        }
    }
    let elapsed = started.elapsed();
    let tokens = (sequence_length * batch_size * iterations) as f64;
    let tokens_per_second = tokens / elapsed.as_secs_f64();

    println!("Qwen3.6 dense scalar benchmark (synthetic tiny config)");
    println!("layers: {}", config.num_hidden_layers);
    println!("sequence_length: {}", sequence_length);
    println!("batch_size: {}", batch_size);
    println!("iterations: {}", iterations);
    println!("elapsed_s: {:.6}", elapsed.as_secs_f64());
    println!("tokens_per_second: {:.2}", tokens_per_second);
    println!("note: this measures the implemented scalar Qwen3.6-shaped runtime, not the official 27B/35B checkpoint.");
}

fn tiny_qwen36_config() -> Config {
    Config::from_json_str(
        r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "eos_token_id": 248044,
                "full_attention_interval": 2,
                "head_dim": 16,
                "hidden_size": 64,
                "intermediate_size": 64,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_value_head_dim": 16,
                "max_position_embeddings": 128,
                "model_type": "qwen3_5_text",
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 2,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000000,
                "vocab_size": 64
            }
        }"#,
    )
    .unwrap()
}

fn synthetic_parameters(config: &Config) -> HashMap<String, Vec<f16>> {
    let mut parameters = HashMap::new();
    insert(
        &mut parameters,
        "model.embed_tokens.weight",
        config.vocab_size * config.hidden_size,
        0.01,
    );
    insert(
        &mut parameters,
        "lm_head.weight",
        config.vocab_size * config.hidden_size,
        0.01,
    );

    insert_linear_attention(&mut parameters, config, 0);
    insert_full_attention(&mut parameters, config, 1);

    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}.mlp", layer);
        insert(
            &mut parameters,
            &format!("{}.gate_proj.weight", prefix),
            config.intermediate_size * config.hidden_size,
            0.008,
        );
        insert(
            &mut parameters,
            &format!("{}.up_proj.weight", prefix),
            config.intermediate_size * config.hidden_size,
            0.007,
        );
        insert(
            &mut parameters,
            &format!("{}.down_proj.weight", prefix),
            config.hidden_size * config.intermediate_size,
            0.006,
        );
    }

    parameters
}

fn insert_linear_attention(parameters: &mut HashMap<String, Vec<f16>>, config: &Config, layer: usize) {
    let prefix = format!("model.layers.{}.linear_attn", layer);
    let key_dim = config.linear_key_dim();
    let value_dim = config.linear_value_dim();
    let conv_dim = key_dim * 2 + value_dim;
    insert(
        parameters,
        &format!("{}.in_proj_qkv.weight", prefix),
        conv_dim * config.hidden_size,
        0.005,
    );
    insert(
        parameters,
        &format!("{}.in_proj_z.weight", prefix),
        value_dim * config.hidden_size,
        0.004,
    );
    insert(
        parameters,
        &format!("{}.in_proj_b.weight", prefix),
        config.linear_num_value_heads * config.hidden_size,
        0.003,
    );
    insert(
        parameters,
        &format!("{}.in_proj_a.weight", prefix),
        config.linear_num_value_heads * config.hidden_size,
        0.002,
    );
    insert(
        parameters,
        &format!("{}.conv1d.weight", prefix),
        conv_dim * config.linear_conv_kernel_dim,
        0.02,
    );
    insert(
        parameters,
        &format!("{}.A_log", prefix),
        config.linear_num_value_heads,
        0.0,
    );
    insert(
        parameters,
        &format!("{}.dt_bias", prefix),
        config.linear_num_value_heads,
        1.0,
    );
    insert(
        parameters,
        &format!("{}.norm.weight", prefix),
        config.linear_value_head_dim,
        1.0,
    );
    insert(
        parameters,
        &format!("{}.out_proj.weight", prefix),
        config.hidden_size * value_dim,
        0.005,
    );
}

fn insert_full_attention(parameters: &mut HashMap<String, Vec<f16>>, config: &Config, layer: usize) {
    let prefix = format!("model.layers.{}.self_attn", layer);
    insert(
        parameters,
        &format!("{}.q_proj.weight", prefix),
        config.num_attention_heads * config.head_dim * 2 * config.hidden_size,
        0.005,
    );
    insert(
        parameters,
        &format!("{}.k_proj.weight", prefix),
        config.num_key_value_heads * config.head_dim * config.hidden_size,
        0.004,
    );
    insert(
        parameters,
        &format!("{}.v_proj.weight", prefix),
        config.num_key_value_heads * config.head_dim * config.hidden_size,
        0.003,
    );
    insert(
        parameters,
        &format!("{}.o_proj.weight", prefix),
        config.hidden_size * config.num_attention_heads * config.head_dim,
        0.005,
    );
    insert(
        parameters,
        &format!("{}.q_norm.weight", prefix),
        config.head_dim,
        0.0,
    );
    insert(
        parameters,
        &format!("{}.k_norm.weight", prefix),
        config.head_dim,
        0.0,
    );
}

fn insert(parameters: &mut HashMap<String, Vec<f16>>, name: &str, len: usize, scale: f32) {
    let values = (0..len)
        .map(|i| {
            let value = if scale == 0.0 {
                0.0
            } else {
                ((i % 17) as f32 - 8.0) * scale
            };
            value as f16
        })
        .collect();
    parameters.insert(name.to_string(), values);
}
