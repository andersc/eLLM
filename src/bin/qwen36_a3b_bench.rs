#![feature(f16)]

use std::f16;
use std::time::Instant;

use ellm::memory::allocator::allocate_init;
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;

fn main() {
    let sequence_length = 16usize;
    let batch_size = 3usize;
    let iterations = std::env::var("QWEN36_A3B_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);

    let config = tiny_qwen36_a3b_config();
    let mut model = Model::<f16>::new(&config, sequence_length, sequence_length, batch_size, 4);
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

    println!("Qwen3.6 35B-A3B MoE scalar benchmark (synthetic tiny config)");
    println!("layers: {}", config.num_hidden_layers);
    println!("experts: {}", config.num_experts);
    println!("experts_per_token: {}", config.num_experts_per_tok);
    println!("sequence_length: {}", sequence_length);
    println!("batch_size: {}", batch_size);
    println!("iterations: {}", iterations);
    println!("elapsed_s: {:.6}", elapsed.as_secs_f64());
    println!("tokens_per_second: {:.2}", tokens_per_second);
    println!("note: this measures the implemented Qwen3.6 A3B-shaped runtime path, not an official 35B checkpoint.");
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
                "layer_types": ["linear_attention", "full_attention"],
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
                "num_hidden_layers": 2,
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
