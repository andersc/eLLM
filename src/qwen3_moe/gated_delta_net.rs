use std::cell::RefCell;
use std::ops::{AddAssign, Neg, Sub};
use std::rc::Rc;

use crate::compiler::mul::qwen36_gated_delta::Qwen36GatedDelta;
use crate::compiler::operator::Operator;
use crate::kernel::generic::from_f32::FromF32;
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};
use crate::memory::cache::Cache;
use crate::ptensor::tensor::Tensor;

use super::config::Config;

#[derive(Clone)]
pub struct GatedDeltaNet<T>
where
    T: Copy + PartialOrd,
{
    in_proj_qkv: Tensor<T>,
    in_proj_z: Tensor<T>,
    in_proj_b: Tensor<T>,
    in_proj_a: Tensor<T>,
    conv1d: Tensor<T>,
    a_log: Tensor<T>,
    dt_bias: Tensor<T>,
    norm_weight: Tensor<T>,
    out_proj: Tensor<T>,
    sequence_length: usize,
    batch_size: usize,
    hidden_size: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel_size: usize,
    rms_norm_eps: f32,
    scope_name: String,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> GatedDeltaNet<T>
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
        _layer_idx: usize,
        sequence_length: usize,
        batch_size: usize,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = format!("{}.linear_attn", parent_scope_name);
        let key_dim = config.linear_key_dim();
        let value_dim = config.linear_value_dim();
        let conv_dim = key_dim * 2 + value_dim;

        Self {
            in_proj_qkv: Tensor::zeros(
                vec![conv_dim, config.hidden_size],
                format!("{}.in_proj_qkv.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            in_proj_z: Tensor::zeros(
                vec![value_dim, config.hidden_size],
                format!("{}.in_proj_z.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            in_proj_b: Tensor::zeros(
                vec![config.linear_num_value_heads, config.hidden_size],
                format!("{}.in_proj_b.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            in_proj_a: Tensor::zeros(
                vec![config.linear_num_value_heads, config.hidden_size],
                format!("{}.in_proj_a.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            conv1d: Tensor::zeros(
                vec![conv_dim, 1, config.linear_conv_kernel_dim],
                format!("{}.conv1d.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            a_log: Tensor::zeros(
                vec![config.linear_num_value_heads],
                format!("{}.A_log", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            dt_bias: Tensor::zeros(
                vec![config.linear_num_value_heads],
                format!("{}.dt_bias", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            norm_weight: Tensor::zeros(
                vec![config.linear_value_head_dim],
                format!("{}.norm.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            out_proj: Tensor::zeros(
                vec![config.hidden_size, value_dim],
                format!("{}.out_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            sequence_length,
            batch_size,
            hidden_size: config.hidden_size,
            num_key_heads: config.linear_num_key_heads,
            num_value_heads: config.linear_num_value_heads,
            key_head_dim: config.linear_key_head_dim,
            value_head_dim: config.linear_value_head_dim,
            conv_kernel_size: config.linear_conv_kernel_dim,
            rms_norm_eps: config.rms_norm_eps,
            scope_name,
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

        let operator = Operator::Qwen36GatedDelta(Qwen36GatedDelta::new(
            hidden_states.data,
            residual.data,
            self.in_proj_qkv.data,
            self.in_proj_z.data,
            self.in_proj_b.data,
            self.in_proj_a.data,
            self.conv1d.data,
            self.a_log.data,
            self.dt_bias.data,
            self.norm_weight.data,
            self.out_proj.data,
            output_tensor.data,
            self.sequence_length,
            self.batch_size,
            self.hidden_size,
            self.num_key_heads,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            self.conv_kernel_size,
            self.rms_norm_eps,
        ));
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }
}
