use std::cell::RefCell;
use std::ops::{AddAssign, Neg, Sub};
use std::rc::Rc;

use crate::compiler::mul::qwen36_full_attention::Qwen36FullAttention;
use crate::compiler::operator::Operator;
use crate::kernel::generic::from_f32::FromF32;
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};
use crate::memory::cache::Cache;
use crate::ptensor::tensor::Tensor;

use super::config::Config;

#[derive(Clone)]
pub struct Qwen36Attention<T>
where
    T: Copy + PartialOrd,
{
    q_weight: Tensor<T>,
    k_weight: Tensor<T>,
    v_weight: Tensor<T>,
    o_weight: Tensor<T>,
    q_norm_weight: Tensor<T>,
    k_norm_weight: Tensor<T>,
    sequence_length: usize,
    batch_size: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    partial_rotary_factor: f32,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> Qwen36Attention<T>
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
        + AddAssign,
{
    pub fn new(
        config: &Config,
        sequence_length: usize,
        batch_size: usize,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = format!("{}.self_attn", parent_scope_name);
        Self {
            q_weight: Tensor::zeros(
                vec![config.num_attention_heads * config.head_dim * 2, config.hidden_size],
                format!("{}.q_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            k_weight: Tensor::zeros(
                vec![config.num_key_value_heads * config.head_dim, config.hidden_size],
                format!("{}.k_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            v_weight: Tensor::zeros(
                vec![config.num_key_value_heads * config.head_dim, config.hidden_size],
                format!("{}.v_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            o_weight: Tensor::zeros(
                vec![config.hidden_size, config.num_attention_heads * config.head_dim],
                format!("{}.o_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            q_norm_weight: Tensor::zeros(
                vec![config.head_dim],
                format!("{}.q_norm.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            k_norm_weight: Tensor::zeros(
                vec![config.head_dim],
                format!("{}.k_norm.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            sequence_length,
            batch_size,
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta as f32,
            partial_rotary_factor: config.partial_rotary_factor,
            cache,
            operator_queue,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        tensor_name: String,
    ) -> Tensor<T> {
        let output_tensor = Tensor::from_cache(
            hidden_states.shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::Qwen36FullAttention(Qwen36FullAttention::new(
            hidden_states.data,
            residual.data,
            self.q_weight.data,
            self.k_weight.data,
            self.v_weight.data,
            self.o_weight.data,
            self.q_norm_weight.data,
            self.k_norm_weight.data,
            output_tensor.data,
            self.sequence_length,
            self.batch_size,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.rms_norm_eps,
            self.rope_theta,
            self.partial_rotary_factor,
        ));
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }
}
