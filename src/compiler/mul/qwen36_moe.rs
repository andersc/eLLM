use crate::init::send_sync_ptr::{ConstPtr, MutPtr};

use super::qwen36_matvec::{matvec, matvec_f32_input, Scalar};

#[derive(Clone)]
pub struct Qwen36Moe<T> {
    input_ptr: ConstPtr<T>,
    residual_ptr: ConstPtr<T>,
    gate_ptr: ConstPtr<T>,
    experts_gate_up_ptr: ConstPtr<T>,
    experts_down_ptr: ConstPtr<T>,
    shared_gate_ptr: ConstPtr<T>,
    shared_up_ptr: ConstPtr<T>,
    shared_down_ptr: ConstPtr<T>,
    shared_expert_gate_ptr: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    sequence_length: usize,
    batch_size: usize,
    hidden_size: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    expert_intermediate_size: usize,
    shared_intermediate_size: usize,
    norm_topk_prob: bool,
}

impl<T> Qwen36Moe<T>
where
    T: Copy + Default,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_ptr: *const T,
        residual_ptr: *const T,
        gate_ptr: *const T,
        experts_gate_up_ptr: *const T,
        experts_down_ptr: *const T,
        shared_gate_ptr: *const T,
        shared_up_ptr: *const T,
        shared_down_ptr: *const T,
        shared_expert_gate_ptr: *const T,
        output_ptr: *mut T,
        sequence_length: usize,
        batch_size: usize,
        hidden_size: usize,
        num_experts: usize,
        num_experts_per_token: usize,
        expert_intermediate_size: usize,
        shared_intermediate_size: usize,
        norm_topk_prob: bool,
    ) -> Self {
        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            residual_ptr: ConstPtr { ptr: residual_ptr },
            gate_ptr: ConstPtr { ptr: gate_ptr },
            experts_gate_up_ptr: ConstPtr {
                ptr: experts_gate_up_ptr,
            },
            experts_down_ptr: ConstPtr {
                ptr: experts_down_ptr,
            },
            shared_gate_ptr: ConstPtr {
                ptr: shared_gate_ptr,
            },
            shared_up_ptr: ConstPtr { ptr: shared_up_ptr },
            shared_down_ptr: ConstPtr {
                ptr: shared_down_ptr,
            },
            shared_expert_gate_ptr: ConstPtr {
                ptr: shared_expert_gate_ptr,
            },
            output_ptr: MutPtr { ptr: output_ptr },
            sequence_length,
            batch_size,
            hidden_size,
            num_experts,
            num_experts_per_token,
            expert_intermediate_size,
            shared_intermediate_size,
            norm_topk_prob,
        }
    }
}

impl<T> Qwen36Moe<T>
where
    T: Scalar,
{
    pub fn run(
        &self,
        position_index: usize,
        position_interval: usize,
        batch_size: usize,
        cpu_num: usize,
        thread_id: usize,
    ) {
        let Some(active_tokens) =
            active_prefix_len(position_index, position_interval, self.sequence_length)
        else {
            return;
        };
        let batch_size = batch_size.min(self.batch_size);
        let num_tokens = active_tokens * batch_size;
        let Some((begin, end)) = crate::compiler::assign::assign(num_tokens, cpu_num, thread_id)
        else {
            return;
        };

        for task in begin..end {
            let position = task / batch_size;
            let batch = task % batch_size;
            let token = position * self.batch_size + batch;
            self.run_token(token);
        }
    }

    fn run_token(&self, token: usize) {
        let input_offset = token * self.hidden_size;
        let output_offset = input_offset;
        let input = unsafe {
            std::slice::from_raw_parts(self.input_ptr.ptr.add(input_offset), self.hidden_size)
        };

        let topk = self.route_topk(input);
        let mut output = vec![0.0f32; self.hidden_size];
        unsafe {
            for h in 0..self.hidden_size {
                output[h] = (*self.residual_ptr.ptr.add(output_offset + h)).to_f32();
            }
        }

        let mut expert_hidden = vec![0.0f32; self.expert_intermediate_size];
        let mut expert_down = vec![0.0f32; self.hidden_size];
        for (expert_id, router_weight) in topk {
            self.compute_expert(input, expert_id, &mut expert_hidden, &mut expert_down);
            for h in 0..self.hidden_size {
                output[h] += router_weight * expert_down[h];
            }
        }

        if self.shared_intermediate_size > 0 {
            let mut shared_hidden = vec![0.0f32; self.shared_intermediate_size];
            let mut shared_down = vec![0.0f32; self.hidden_size];
            let shared_gate = self.compute_shared_expert_gate(input);
            self.compute_shared_expert(input, &mut shared_hidden, &mut shared_down);
            for h in 0..self.hidden_size {
                output[h] += shared_gate * shared_down[h];
            }
        }

        unsafe {
            for h in 0..self.hidden_size {
                *self.output_ptr.ptr.add(output_offset + h) = T::from_f32(output[h]);
            }
        }
    }

    fn route_topk(&self, input: &[T]) -> Vec<(usize, f32)> {
        let mut logits = vec![0.0f32; self.num_experts];
        unsafe {
            matvec(
                input.as_ptr(),
                self.gate_ptr.ptr,
                &mut logits,
                self.num_experts,
                self.hidden_size,
            );
        }

        let mut indices = (0..self.num_experts).collect::<Vec<_>>();
        indices.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        indices.truncate(self.num_experts_per_token.min(self.num_experts));

        let max_logit = indices
            .iter()
            .map(|&idx| logits[idx])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights = indices
            .iter()
            .map(|&idx| (logits[idx] - max_logit).exp())
            .collect::<Vec<_>>();
        let sum = weights.iter().sum::<f32>();
        if self.norm_topk_prob && sum > 0.0 {
            for weight in &mut weights {
                *weight /= sum;
            }
        }

        indices.into_iter().zip(weights).collect()
    }

    fn compute_expert(
        &self,
        input: &[T],
        expert_id: usize,
        expert_hidden: &mut [f32],
        expert_down: &mut [f32],
    ) {
        let expert_weight_stride = 2 * self.expert_intermediate_size * self.hidden_size;
        let expert_down_stride = self.hidden_size * self.expert_intermediate_size;
        let mut gate = vec![0.0f32; self.expert_intermediate_size];
        let mut up = vec![0.0f32; self.expert_intermediate_size];

        unsafe {
            matvec(
                input.as_ptr(),
                self.experts_gate_up_ptr
                    .ptr
                    .add(expert_id * expert_weight_stride),
                &mut gate,
                self.expert_intermediate_size,
                self.hidden_size,
            );
            matvec(
                input.as_ptr(),
                self.experts_gate_up_ptr.ptr.add(
                    expert_id * expert_weight_stride
                        + self.expert_intermediate_size * self.hidden_size,
                ),
                &mut up,
                self.expert_intermediate_size,
                self.hidden_size,
            );
        }

        for i in 0..self.expert_intermediate_size {
            expert_hidden[i] = silu(gate[i]) * up[i];
        }

        unsafe {
            matvec_f32_input(
                expert_hidden,
                self.experts_down_ptr
                    .ptr
                    .add(expert_id * expert_down_stride),
                expert_down,
                self.hidden_size,
                self.expert_intermediate_size,
            );
        }
    }

    fn compute_shared_expert_gate(&self, input: &[T]) -> f32 {
        let mut gate = [0.0f32; 1];
        unsafe {
            matvec(
                input.as_ptr(),
                self.shared_expert_gate_ptr.ptr,
                &mut gate,
                1,
                self.hidden_size,
            );
        }
        sigmoid(gate[0])
    }

    fn compute_shared_expert(
        &self,
        input: &[T],
        shared_hidden: &mut [f32],
        shared_down: &mut [f32],
    ) {
        let mut gate = vec![0.0f32; self.shared_intermediate_size];
        let mut up = vec![0.0f32; self.shared_intermediate_size];

        unsafe {
            matvec(
                input.as_ptr(),
                self.shared_gate_ptr.ptr,
                &mut gate,
                self.shared_intermediate_size,
                self.hidden_size,
            );
            matvec(
                input.as_ptr(),
                self.shared_up_ptr.ptr,
                &mut up,
                self.shared_intermediate_size,
                self.hidden_size,
            );
        }

        for i in 0..self.shared_intermediate_size {
            shared_hidden[i] = silu(gate[i]) * up[i];
        }

        unsafe {
            matvec_f32_input(
                shared_hidden,
                self.shared_down_ptr.ptr,
                shared_down,
                self.hidden_size,
                self.shared_intermediate_size,
            );
        }
    }
}

fn active_prefix_len(
    position_index: usize,
    position_interval: usize,
    sequence_length: usize,
) -> Option<usize> {
    if position_interval == 0 || sequence_length == 0 {
        return None;
    }

    let active_len = position_index
        .saturating_add(position_interval)
        .min(sequence_length);
    (active_len > 0).then_some(active_len)
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
