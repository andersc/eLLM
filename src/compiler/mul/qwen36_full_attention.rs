use std::f16;

use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel::generic::from_f32::FromF32;

pub trait Scalar: Copy + Default + FromF32 {
    fn to_f32(self) -> f32;
}

impl Scalar for f16 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Scalar for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl Scalar for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

#[derive(Clone)]
pub struct Qwen36FullAttention<T> {
    input_ptr: ConstPtr<T>,
    residual_ptr: ConstPtr<T>,
    q_proj_ptr: ConstPtr<T>,
    k_proj_ptr: ConstPtr<T>,
    v_proj_ptr: ConstPtr<T>,
    o_proj_ptr: ConstPtr<T>,
    q_norm_ptr: ConstPtr<T>,
    k_norm_ptr: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    sequence_length: usize,
    batch_size: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    partial_rotary_factor: f32,
}

impl<T> Qwen36FullAttention<T>
where
    T: Copy + Default,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_ptr: *const T,
        residual_ptr: *const T,
        q_proj_ptr: *const T,
        k_proj_ptr: *const T,
        v_proj_ptr: *const T,
        o_proj_ptr: *const T,
        q_norm_ptr: *const T,
        k_norm_ptr: *const T,
        output_ptr: *mut T,
        sequence_length: usize,
        batch_size: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f32,
        rope_theta: f32,
        partial_rotary_factor: f32,
    ) -> Self {
        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            residual_ptr: ConstPtr { ptr: residual_ptr },
            q_proj_ptr: ConstPtr { ptr: q_proj_ptr },
            k_proj_ptr: ConstPtr { ptr: k_proj_ptr },
            v_proj_ptr: ConstPtr { ptr: v_proj_ptr },
            o_proj_ptr: ConstPtr { ptr: o_proj_ptr },
            q_norm_ptr: ConstPtr { ptr: q_norm_ptr },
            k_norm_ptr: ConstPtr { ptr: k_norm_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            sequence_length,
            batch_size,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            partial_rotary_factor,
        }
    }
}

impl<T> Qwen36FullAttention<T>
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
        let Some(active_len) =
            active_prefix_len(position_index, position_interval, self.sequence_length)
        else {
            return;
        };
        let batch_size = batch_size.min(self.batch_size);
        let Some((begin, end)) = crate::compiler::assign::assign(batch_size, cpu_num, thread_id)
        else {
            return;
        };

        for batch in begin..end {
            self.run_batch(batch, active_len);
        }
    }

    fn run_batch(&self, batch: usize, active_len: usize) {
        let q_dim = self.num_attention_heads * self.head_dim;
        let kv_dim = self.num_key_value_heads * self.head_dim;
        let groups = self.num_attention_heads / self.num_key_value_heads;
        let mut q = vec![0.0f32; active_len * q_dim];
        let mut gate = vec![0.0f32; active_len * q_dim];
        let mut k = vec![0.0f32; active_len * kv_dim];
        let mut v = vec![0.0f32; active_len * kv_dim];

        for t in 0..active_len {
            let input_offset = (t * self.batch_size + batch) * self.hidden_size;
            let mut q_raw = vec![0.0f32; q_dim * 2];
            let mut k_raw = vec![0.0f32; kv_dim];
            let mut v_raw = vec![0.0f32; kv_dim];
            unsafe {
                matvec(
                    self.input_ptr.ptr.add(input_offset),
                    self.q_proj_ptr.ptr,
                    &mut q_raw,
                    q_dim * 2,
                    self.hidden_size,
                );
                matvec(
                    self.input_ptr.ptr.add(input_offset),
                    self.k_proj_ptr.ptr,
                    &mut k_raw,
                    kv_dim,
                    self.hidden_size,
                );
                matvec(
                    self.input_ptr.ptr.add(input_offset),
                    self.v_proj_ptr.ptr,
                    &mut v_raw,
                    kv_dim,
                    self.hidden_size,
                );
            }

            for head in 0..self.num_attention_heads {
                let raw_offset = head * self.head_dim * 2;
                let out_offset = t * q_dim + head * self.head_dim;
                let mut q_head = q_raw[raw_offset..raw_offset + self.head_dim].to_vec();
                rms_norm_head(
                    &mut q_head,
                    unsafe { self.q_norm_ptr.ptr },
                    self.head_dim,
                    self.rms_norm_eps,
                );
                apply_rope(&mut q_head, t, self.rope_theta, self.partial_rotary_factor);
                q[out_offset..out_offset + self.head_dim].copy_from_slice(&q_head);
                gate[out_offset..out_offset + self.head_dim].copy_from_slice(
                    &q_raw[raw_offset + self.head_dim..raw_offset + self.head_dim * 2],
                );
            }

            for head in 0..self.num_key_value_heads {
                let out_offset = t * kv_dim + head * self.head_dim;
                let mut k_head = k_raw[head * self.head_dim..(head + 1) * self.head_dim].to_vec();
                rms_norm_head(
                    &mut k_head,
                    unsafe { self.k_norm_ptr.ptr },
                    self.head_dim,
                    self.rms_norm_eps,
                );
                apply_rope(&mut k_head, t, self.rope_theta, self.partial_rotary_factor);
                k[out_offset..out_offset + self.head_dim].copy_from_slice(&k_head);
                v[out_offset..out_offset + self.head_dim]
                    .copy_from_slice(&v_raw[head * self.head_dim..(head + 1) * self.head_dim]);
            }
        }

        let mut attention_out = vec![0.0f32; q_dim];
        let mut projected = vec![0.0f32; self.hidden_size];
        for t in 0..active_len {
            attention_out.fill(0.0);
            for head in 0..self.num_attention_heads {
                let kv_head = head / groups;
                let q_offset = t * q_dim + head * self.head_dim;
                let mut scores = vec![0.0f32; t + 1];
                for source_t in 0..=t {
                    let k_offset = source_t * kv_dim + kv_head * self.head_dim;
                    let mut score = 0.0f32;
                    for d in 0..self.head_dim {
                        score += q[q_offset + d] * k[k_offset + d];
                    }
                    scores[source_t] = score / (self.head_dim as f32).sqrt();
                }
                softmax_in_place(&mut scores);
                for source_t in 0..=t {
                    let v_offset = source_t * kv_dim + kv_head * self.head_dim;
                    for d in 0..self.head_dim {
                        attention_out[head * self.head_dim + d] +=
                            scores[source_t] * v[v_offset + d];
                    }
                }
                for d in 0..self.head_dim {
                    let idx = head * self.head_dim + d;
                    attention_out[idx] *= sigmoid(gate[t * q_dim + idx]);
                }
            }

            for out_dim in 0..self.hidden_size {
                let mut acc = 0.0f32;
                for col in 0..q_dim {
                    unsafe {
                        acc += attention_out[col]
                            * (*self.o_proj_ptr.ptr.add(out_dim * q_dim + col)).to_f32();
                    }
                }
                projected[out_dim] = acc;
            }

            let output_offset = (t * self.batch_size + batch) * self.hidden_size;
            unsafe {
                for h in 0..self.hidden_size {
                    let residual = (*self.residual_ptr.ptr.add(output_offset + h)).to_f32();
                    *self.output_ptr.ptr.add(output_offset + h) =
                        T::from_f32(residual + projected[h]);
                }
            }
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

unsafe fn matvec<T: Scalar>(
    input_ptr: *const T,
    weight_ptr: *const T,
    output: &mut [f32],
    output_dim: usize,
    input_dim: usize,
) {
    for row in 0..output_dim {
        let mut acc = 0.0f32;
        for col in 0..input_dim {
            acc +=
                (*input_ptr.add(col)).to_f32() * (*weight_ptr.add(row * input_dim + col)).to_f32();
        }
        output[row] = acc;
    }
}

fn rms_norm_head<T: Scalar>(values: &mut [f32], weight_ptr: *const T, head_dim: usize, eps: f32) {
    let rms = (values.iter().map(|v| v * v).sum::<f32>() / head_dim as f32 + eps).sqrt();
    for d in 0..head_dim {
        let weight = unsafe { (*weight_ptr.add(d)).to_f32() };
        values[d] = values[d] / rms * (1.0 + weight);
    }
}

fn apply_rope(values: &mut [f32], position: usize, theta: f32, partial_rotary_factor: f32) {
    let mut rotary_dim =
        ((values.len() as f32 * partial_rotary_factor).round() as usize).min(values.len());
    rotary_dim -= rotary_dim % 2;
    if rotary_dim == 0 {
        return;
    }

    let half = rotary_dim / 2;
    let original = values[..rotary_dim].to_vec();
    for d in 0..rotary_dim {
        let freq_idx = d % half;
        let inv_freq = 1.0 / theta.powf((2 * freq_idx) as f32 / rotary_dim as f32);
        let angle = position as f32 * inv_freq;
        let rotated = if d < half {
            -original[d + half]
        } else {
            original[d - half]
        };
        values[d] = original[d] * angle.cos() + rotated * angle.sin();
    }
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values.iter_mut() {
        *value /= sum;
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
