use std::cell::RefCell;
use std::ops::{AddAssign, Neg, Sub};
use std::rc::Rc;

use crate::compiler::mul::qwen36_moe::Qwen36Moe;
use crate::compiler::operator::Operator;
use crate::kernel::generic::from_f32::FromF32;
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};
use crate::memory::cache::Cache;
use crate::ptensor::tensor::Tensor;

use super::config::Config;

#[derive(Clone)]
pub struct Qwen36MoeBlock<T>
where
    T: Copy + PartialOrd,
{
    gate_weight: Tensor<T>,
    experts_gate_up_weight: Tensor<T>,
    experts_down_weight: Tensor<T>,
    shared_gate_weight: Tensor<T>,
    shared_up_weight: Tensor<T>,
    shared_down_weight: Tensor<T>,
    shared_expert_gate_weight: Tensor<T>,
    sequence_length: usize,
    batch_size: usize,
    hidden_size: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    expert_intermediate_size: usize,
    shared_intermediate_size: usize,
    norm_topk_prob: bool,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> Qwen36MoeBlock<T>
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
        let scope_name = format!("{}.mlp", parent_scope_name);
        let shared_intermediate_size = config.shared_experts_intermediate_size;
        Self {
            gate_weight: Tensor::zeros(
                vec![config.num_experts, config.hidden_size],
                format!("{}.gate.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            experts_gate_up_weight: Tensor::zeros(
                vec![
                    config.num_experts,
                    2 * config.moe_intermediate_size,
                    config.hidden_size,
                ],
                format!("{}.experts.gate_up_proj", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            experts_down_weight: Tensor::zeros(
                vec![
                    config.num_experts,
                    config.hidden_size,
                    config.moe_intermediate_size,
                ],
                format!("{}.experts.down_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            shared_gate_weight: Tensor::zeros(
                vec![shared_intermediate_size, config.hidden_size],
                format!("{}.shared_expert.gate_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            shared_up_weight: Tensor::zeros(
                vec![shared_intermediate_size, config.hidden_size],
                format!("{}.shared_expert.up_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            shared_down_weight: Tensor::zeros(
                vec![config.hidden_size, shared_intermediate_size],
                format!("{}.shared_expert.down_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            shared_expert_gate_weight: Tensor::zeros(
                vec![1, config.hidden_size],
                format!("{}.shared_expert_gate.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            sequence_length,
            batch_size,
            hidden_size: config.hidden_size,
            num_experts: config.num_experts,
            num_experts_per_token: config.num_experts_per_tok,
            expert_intermediate_size: config.moe_intermediate_size,
            shared_intermediate_size,
            norm_topk_prob: config.norm_topk_prob,
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

        let operator = Operator::Qwen36Moe(Qwen36Moe::new(
            hidden_states.data,
            residual.data,
            self.gate_weight.data,
            self.experts_gate_up_weight.data,
            self.experts_down_weight.data,
            self.shared_gate_weight.data,
            self.shared_up_weight.data,
            self.shared_down_weight.data,
            self.shared_expert_gate_weight.data,
            output_tensor.data,
            self.sequence_length,
            self.batch_size,
            self.hidden_size,
            self.num_experts,
            self.num_experts_per_token,
            self.expert_intermediate_size,
            self.shared_intermediate_size,
            self.norm_topk_prob,
        ));
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }
}
