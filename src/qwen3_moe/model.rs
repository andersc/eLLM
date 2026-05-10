use core_affinity;
use std::cell::RefCell;
use std::cell::SyncUnsafeCell;
use std::collections::HashMap;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::ptr::null;
use std::rc::Rc;

use std::sync::Barrier;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;

// use serde::{Deserialize, Serialize};
// use hurdles::Barrier;
// use super::barrier::Barrier;
// use serde::{Deserialize, Serialize};

use super::config::Config;
use crate::kernel::generic::from_f32::FromF32;
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::compiler::map::rms_map::RMSMap;
use super::super::compiler::operator::Operator;
use super::super::init::matmul_params::MatMulParams;
use super::super::memory::cache::Cache;
// use super::super::memory::model_loader::SafeTensorsLoader;
// use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;
use super::decoder_layer::DecoderLayer;

// use super::rope::precompute_freqs_cis;

// #[derive(Clone)]
pub struct Model<T>
where
    T: Copy + PartialOrd,
{
    // config: Config,
    // sequences: Vec<usize>,
    word_embedding: Rc<Tensor<T>>,
    position_embedding: Rc<Tensor<T>>,
    lm_head_weight: Tensor<T>,
    pub layers: Vec<DecoderLayer<T>>,
    rms_norm_eps: T,
    pub sequence_chunk_size: usize,
    pub batch_size: usize,
    pub hidden_size: usize,
    pub topk_size: usize,
    scope_name: String,
    pub cache: Rc<RefCell<Cache<T>>>,
    pub operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> Model<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid<T>
        + Sqrt
        + FromF32
        + AddAssign
        + Send
        + Sync,
{
    pub fn new(
        config: &Config,
        sequence_length: usize,
        sequence_chunk_size: usize,
        batch_size: usize,
        topk_size: usize,
    ) -> Self {
        Self::new_with_parameters(
            config,
            sequence_length,
            sequence_chunk_size,
            batch_size,
            topk_size,
            HashMap::new(),
        )
    }

    pub fn new_with_parameters(
        config: &Config,
        sequence_length: usize,
        sequence_chunk_size: usize,
        batch_size: usize,
        topk_size: usize,
        parameter_tensors: HashMap<String, Vec<T>>,
    ) -> Self {
        if let Some(reason) = config.unsupported_runtime_reason() {
            panic!("{}", reason);
        }

        let scope_name = String::from("model");

        // let torch_file = String::from("D:/llama-3-chinese-8b-instruct-v3");
        // let loader = SafeTensorsLoader::new(&torch_file).unwrap();
        // let tensors = loader.load_all_weights_f16().unwrap();
        let cache = Rc::new(RefCell::new(Cache::new(parameter_tensors)));
        let operator_queue: Rc<RefCell<Vec<Operator<T>>>> = Rc::new(RefCell::new(Vec::new()));

        // Create default tensors
        let word_embedding = Rc::new(Tensor::zeros(
            vec![config.vocab_size, config.hidden_size],
            String::from("model.embed_tokens.weight"),
            cache.clone(),
            operator_queue.clone(),
        ));

        let position_embedding = Rc::new(Tensor::zeros(
            vec![config.max_position_embeddings, 1, 1, config.head_dim],
            String::from("model.position_embedding.weight"),
            cache.clone(),
            operator_queue.clone(),
        ));

        let mut layers: Vec<DecoderLayer<T>> = Vec::new();
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::<T>::new(
                &config,
                i,
                sequence_length,
                sequence_chunk_size,
                batch_size,
                word_embedding.clone(),
                position_embedding.clone(),
                &scope_name.clone(),
                cache.clone(),
                operator_queue.clone(),
            ));
        }

        Self {
            // sequences: vec![0; (config.max_position_embeddings + 1) * batch_size],
            word_embedding: word_embedding.clone(),
            position_embedding: position_embedding.clone(),
            lm_head_weight: Tensor::zeros(
                vec![config.vocab_size, config.hidden_size],
                String::from("lm_head.weight"),
                cache.clone(),
                operator_queue.clone(),
            ),
            layers: layers,
            batch_size: batch_size,
            hidden_size: config.hidden_size,
            sequence_chunk_size: sequence_chunk_size,
            topk_size: topk_size,
            rms_norm_eps: T::from_f32(config.rms_norm_eps),
            scope_name: scope_name,
            cache: cache,
            operator_queue: operator_queue,
        }
    }

    pub fn forward(&mut self, sequences: *mut usize) -> (*const usize, Tensor<T>) {
        // -> Tensor<T> {
        // let sequences = vec![0; (self.config.max_position_embeddings + 1) * self.config.batch_size].into_boxed_slice();

        let mut hidden_state = Tensor::<T>::zeros(
            vec![self.batch_size, self.hidden_size],
            format!("{}.hidden_state.output", self.scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        for (i, layer_module) in self.layers.iter().enumerate() {
            hidden_state = layer_module.forward(
                &hidden_state,
                sequences,
                format!("{}.hidden_states.{}.output", self.scope_name, i),
            );
            // all_hidden_states.push(hidden_states);
        }

        let norm_state = hidden_state.rms(
            self.rms_norm_eps,
            format!("{}.norm_hidden", self.scope_name),
        );
   
        let (indices_ptr, values_tensor) = norm_state.matmul_local_topk(
            &self.lm_head_weight,
           MatMulParams {
                    a_row_step_macro: 3,
                    b_row_step_macro: 64,
                    column_step_macro: 64,
                    a_row_step_micro: 3,
                    b_row_step_micro: 32,
                },
            self.topk_size,
            format!("{}.lm_head", self.scope_name),
        );

        let (topk_indice, topk_value) = values_tensor.topk_softmax(
            indices_ptr,

            unsafe { sequences.add(self.batch_size) },
            self.topk_size,
            format!("{}.softmax", self.scope_name),
        );

        (topk_indice, topk_value)
        // (null(), values_tensor)
    }
}

// unsafe impl<T: Copy + Default + Send + Sync> Send for Transformer<T> {}
// unsafe impl<T: Copy + Default + Send + Sync> Sync for Transformer<T> {}

#[cfg(test)]
mod test {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::thread;

    use super::*;
    // use crate::init::config::Config;
    // use crate::llama::model_loader::SafeTensorsLoader;
    use crate::memory::allocator::allocate_init;
    use crate::memory::cache::Cache;
    use crate::ptensor::tensor::Tensor;

    #[test]
    fn test_model_forward() {
        // let cpu_num =  thread::available_parallelism().unwrap().get();
        let sequence_length = 128;
        let sequence_chunk_size = 1;
        let batch_size = 3;
        let topk_size = 8;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let mut model = Model::<f32>::new(
            &config,
            sequence_length,
            sequence_chunk_size,
            batch_size,
            topk_size, // word_embedding,
                       // position_embedding,
                       // norm_weight,
                       // cpu_num,
                       // cache.clone(),
                       // operator_queue.clone(),
        );

        // let mut sequences: Vec<usize> = vec![0; (config.max_position_embeddings + 1)*config.batch_size];
        let mut sequences = allocate_init::<usize>((sequence_length + 1) * batch_size, 0);
        let output_tensor = unsafe { model.forward(sequences) };

        
        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        for operator in model.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(0, 1, batch_size, thread_num, i);
            }
        }

        // Add assertions to verify the output_tensor
        // For example:
        // assert_eq!(output_tensor.shape, vec![config.batch_size, config.hidden_size]);
    }

    #[test]
    fn test_model_forward_f16() {
        let sequence_length = 128;
        let sequence_chunk_size = 1;
        let batch_size = 3;
        let topk_size = 8;

        let config =
            Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

        let mut model = Model::<f16>::new(
            &config,
            sequence_length,
            sequence_chunk_size,
            batch_size,
            topk_size,
        );

        let mut sequences =
            allocate_init::<usize>((config.max_position_embeddings + 1) * batch_size, 0);
        let output_tensor = unsafe { model.forward(sequences) };

        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        for operator in model.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(0, 1, batch_size, thread_num, i);
            }
        }
    }

    #[test]
    fn test_qwen3_dense_model_uses_mlp_path() {
        let config = Config::from_json_str(
            r#"{
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "vocab_size": 32000,
                "max_position_embeddings": 16,
                "eos_token_id": [151645, 151643],
                "rms_norm_eps": 1e-6
            }"#,
        )
        .unwrap();

        let mut model = Model::<f16>::new(&config, 16, 1, 3, 8);
        let sequences = allocate_init::<usize>(17 * 3, 0);
        let _ = model.forward(sequences);
        let queue = model.operator_queue.borrow();

        assert!(queue.iter().any(|op| matches!(op, Operator::SiluMulZipMap(_))));
        assert!(!queue.iter().any(|op| matches!(
            op,
            Operator::ExpertsSoftmaxNorm(_)
                | Operator::ExpertsMatMulSilu(_)
                | Operator::ExpertsMatMulDown(_)
        )));
    }

    #[test]
    fn test_qwen36_linear_attention_runtime_queues_gated_delta() {
        let config = Config::from_json_str(
            r#"{
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "eos_token_id": 248044,
                    "head_dim": 4,
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "layer_types": ["linear_attention"],
                    "linear_conv_kernel_dim": 2,
                    "linear_key_head_dim": 2,
                    "linear_num_key_heads": 1,
                    "linear_num_value_heads": 1,
                    "linear_value_head_dim": 2,
                    "model_type": "qwen3_5_text",
                    "num_attention_heads": 1,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 1,
                    "vocab_size": 8
                }
            }"#,
        )
        .unwrap();

        let mut model = Model::<f16>::new(&config, 2, 2, 1, 2);
        let sequences = allocate_init::<usize>(3, 0);
        let _ = model.forward(sequences);
        let queue = model.operator_queue.borrow();

        assert!(queue
            .iter()
            .any(|op| matches!(op, Operator::Qwen36GatedDelta(_))));
    }

    #[test]
    fn test_qwen36_full_attention_runtime_queues_output_gated_attention() {
        let config = Config::from_json_str(
            r#"{
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "eos_token_id": 248044,
                    "head_dim": 4,
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "layer_types": ["full_attention"],
                    "model_type": "qwen3_5_text",
                    "num_attention_heads": 1,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 1,
                    "vocab_size": 8
                }
            }"#,
        )
        .unwrap();

        let mut model = Model::<f16>::new(&config, 2, 2, 1, 2);
        let sequences = allocate_init::<usize>(3, 0);
        let _ = model.forward(sequences);
        let queue = model.operator_queue.borrow();

        assert!(queue
            .iter()
            .any(|op| matches!(op, Operator::Qwen36FullAttention(_))));
    }

    #[test]
    fn test_qwen36_dense_runtime_executes_operator_queue() {
        let config = Config::from_json_str(
            r#"{
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "eos_token_id": 248044,
                    "head_dim": 16,
                    "hidden_size": 64,
                    "intermediate_size": 64,
                    "layer_types": ["linear_attention", "full_attention"],
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 16,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_value_head_dim": 16,
                    "max_position_embeddings": 32,
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
        .unwrap();

        let sequence_length = 4;
        let batch_size = 3;
        let mut model = Model::<f16>::new(&config, sequence_length, sequence_length, batch_size, 4);
        let sequences = allocate_init::<usize>((sequence_length + 1) * batch_size, 0);
        let _ = model.forward(sequences);
        let queue = model.operator_queue.borrow();

        assert!(queue
            .iter()
            .any(|op| matches!(op, Operator::Qwen36GatedDelta(_))));
        assert!(queue
            .iter()
            .any(|op| matches!(op, Operator::Qwen36FullAttention(_))));

        for operator in queue.iter() {
            operator.run(0, sequence_length, batch_size, 1, 0);
        }
    }

    #[test]
    fn test_qwen36_a3b_moe_runtime_executes_operator_queue() {
        let config = Config::from_json_str(
            r#"{
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "eos_token_id": 248044,
                    "head_dim": 16,
                    "hidden_size": 64,
                    "intermediate_size": 64,
                    "layer_types": ["linear_attention"],
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 16,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_value_head_dim": 16,
                    "max_position_embeddings": 32,
                    "model_type": "qwen3_5_moe_text",
                    "moe_intermediate_size": 64,
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
        .unwrap();

        let sequence_length = 4;
        let batch_size = 3;
        let mut model = Model::<f16>::new(&config, sequence_length, sequence_length, batch_size, 4);
        let sequences = allocate_init::<usize>((sequence_length + 1) * batch_size, 0);
        let _ = model.forward(sequences);
        let queue = model.operator_queue.borrow();

        assert!(queue
            .iter()
            .any(|op| matches!(op, Operator::Qwen36GatedDelta(_))));
        assert!(queue.iter().any(|op| matches!(op, Operator::Qwen36Moe(_))));

        for operator in queue.iter() {
            operator.run(0, sequence_length, batch_size, 1, 0);
        }
    }
}
