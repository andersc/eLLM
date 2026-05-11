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
pub struct Qwen36GatedDelta<T> {
    input_ptr: ConstPtr<T>,
    residual_ptr: ConstPtr<T>,
    in_proj_qkv_ptr: ConstPtr<T>,
    in_proj_z_ptr: ConstPtr<T>,
    in_proj_b_ptr: ConstPtr<T>,
    in_proj_a_ptr: ConstPtr<T>,
    conv1d_ptr: ConstPtr<T>,
    a_log_ptr: ConstPtr<T>,
    dt_bias_ptr: ConstPtr<T>,
    norm_weight_ptr: ConstPtr<T>,
    out_proj_ptr: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    sequence_length: usize,
    batch_size: usize,
    hidden_size: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel_size: usize,
    rms_norm_eps: f32,
}

impl<T> Qwen36GatedDelta<T>
where
    T: Copy + Default,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_ptr: *const T,
        residual_ptr: *const T,
        in_proj_qkv_ptr: *const T,
        in_proj_z_ptr: *const T,
        in_proj_b_ptr: *const T,
        in_proj_a_ptr: *const T,
        conv1d_ptr: *const T,
        a_log_ptr: *const T,
        dt_bias_ptr: *const T,
        norm_weight_ptr: *const T,
        out_proj_ptr: *const T,
        output_ptr: *mut T,
        sequence_length: usize,
        batch_size: usize,
        hidden_size: usize,
        num_key_heads: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        conv_kernel_size: usize,
        rms_norm_eps: f32,
    ) -> Self {
        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            residual_ptr: ConstPtr { ptr: residual_ptr },
            in_proj_qkv_ptr: ConstPtr {
                ptr: in_proj_qkv_ptr,
            },
            in_proj_z_ptr: ConstPtr { ptr: in_proj_z_ptr },
            in_proj_b_ptr: ConstPtr { ptr: in_proj_b_ptr },
            in_proj_a_ptr: ConstPtr { ptr: in_proj_a_ptr },
            conv1d_ptr: ConstPtr { ptr: conv1d_ptr },
            a_log_ptr: ConstPtr { ptr: a_log_ptr },
            dt_bias_ptr: ConstPtr { ptr: dt_bias_ptr },
            norm_weight_ptr: ConstPtr {
                ptr: norm_weight_ptr,
            },
            out_proj_ptr: ConstPtr { ptr: out_proj_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            sequence_length,
            batch_size,
            hidden_size,
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
            rms_norm_eps,
        }
    }
}

impl<T> Qwen36GatedDelta<T>
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
        let key_dim = self.num_key_heads * self.key_head_dim;
        let value_dim = self.num_value_heads * self.value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let head_ratio = self.num_value_heads / self.num_key_heads;

        let mut projected_qkv = vec![0.0f32; active_len * conv_dim];
        let mut projected_z = vec![0.0f32; active_len * value_dim];
        let mut projected_b = vec![0.0f32; active_len * self.num_value_heads];
        let mut projected_a = vec![0.0f32; active_len * self.num_value_heads];

        for t in 0..active_len {
            let input_offset = (t * self.batch_size + batch) * self.hidden_size;
            unsafe {
                matvec(
                    self.input_ptr.ptr.add(input_offset),
                    self.in_proj_qkv_ptr.ptr,
                    &mut projected_qkv[t * conv_dim..(t + 1) * conv_dim],
                    conv_dim,
                    self.hidden_size,
                );
                matvec(
                    self.input_ptr.ptr.add(input_offset),
                    self.in_proj_z_ptr.ptr,
                    &mut projected_z[t * value_dim..(t + 1) * value_dim],
                    value_dim,
                    self.hidden_size,
                );
                matvec(
                    self.input_ptr.ptr.add(input_offset),
                    self.in_proj_b_ptr.ptr,
                    &mut projected_b[t * self.num_value_heads..(t + 1) * self.num_value_heads],
                    self.num_value_heads,
                    self.hidden_size,
                );
                matvec(
                    self.input_ptr.ptr.add(input_offset),
                    self.in_proj_a_ptr.ptr,
                    &mut projected_a[t * self.num_value_heads..(t + 1) * self.num_value_heads],
                    self.num_value_heads,
                    self.hidden_size,
                );
            }
        }

        let mut mixed_qkv = vec![0.0f32; projected_qkv.len()];
        for t in 0..active_len {
            for c in 0..conv_dim {
                let mut acc = 0.0f32;
                for k in 0..self.conv_kernel_size {
                    let source_t = t as isize + k as isize + 1 - self.conv_kernel_size as isize;
                    if source_t < 0 {
                        continue;
                    }
                    let source_t = source_t as usize;
                    if source_t >= active_len {
                        continue;
                    }
                    unsafe {
                        let w = (*self.conv1d_ptr.ptr.add(c * self.conv_kernel_size + k)).to_f32();
                        acc += projected_qkv[source_t * conv_dim + c] * w;
                    }
                }
                mixed_qkv[t * conv_dim + c] = silu(acc);
            }
        }

        let mut recurrent_state =
            vec![0.0f32; self.num_value_heads * self.key_head_dim * self.value_head_dim];
        let mut core = vec![0.0f32; value_dim];
        let mut projected = vec![0.0f32; self.hidden_size];

        for t in 0..active_len {
            core.fill(0.0);

            for value_head in 0..self.num_value_heads {
                let key_head = value_head / head_ratio;
                let q_offset = t * conv_dim + key_head * self.key_head_dim;
                let k_offset = t * conv_dim + key_dim + key_head * self.key_head_dim;
                let v_offset = t * conv_dim + key_dim * 2 + value_head * self.value_head_dim;
                let z_offset = t * value_dim + value_head * self.value_head_dim;

                let q = l2_norm(&mixed_qkv[q_offset..q_offset + self.key_head_dim]);
                let k = l2_norm(&mixed_qkv[k_offset..k_offset + self.key_head_dim]);
                let q_scale = 1.0f32 / (self.key_head_dim as f32).sqrt();
                let beta = sigmoid(projected_b[t * self.num_value_heads + value_head]);
                let g = unsafe {
                    -(*self.a_log_ptr.ptr.add(value_head)).to_f32().exp()
                        * softplus(
                            projected_a[t * self.num_value_heads + value_head]
                                + (*self.dt_bias_ptr.ptr.add(value_head)).to_f32(),
                        )
                };
                let decay = g.exp();
                let state_head_offset = value_head * self.key_head_dim * self.value_head_dim;

                for kd in 0..self.key_head_dim {
                    for vd in 0..self.value_head_dim {
                        recurrent_state[state_head_offset + kd * self.value_head_dim + vd] *= decay;
                    }
                }

                let mut retrieved = vec![0.0f32; self.value_head_dim];
                for kd in 0..self.key_head_dim {
                    for vd in 0..self.value_head_dim {
                        retrieved[vd] += recurrent_state
                            [state_head_offset + kd * self.value_head_dim + vd]
                            * k[kd];
                    }
                }

                for kd in 0..self.key_head_dim {
                    for vd in 0..self.value_head_dim {
                        let value = mixed_qkv[v_offset + vd];
                        let delta = (value - retrieved[vd]) * beta;
                        recurrent_state[state_head_offset + kd * self.value_head_dim + vd] +=
                            k[kd] * delta;
                    }
                }

                let mut head_out = vec![0.0f32; self.value_head_dim];
                for kd in 0..self.key_head_dim {
                    for vd in 0..self.value_head_dim {
                        head_out[vd] += recurrent_state
                            [state_head_offset + kd * self.value_head_dim + vd]
                            * q[kd]
                            * q_scale;
                    }
                }

                let rms = (head_out.iter().map(|v| v * v).sum::<f32>()
                    / self.value_head_dim as f32
                    + self.rms_norm_eps)
                    .sqrt();
                for vd in 0..self.value_head_dim {
                    let norm_weight = unsafe { (*self.norm_weight_ptr.ptr.add(vd)).to_f32() };
                    core[value_head * self.value_head_dim + vd] =
                        head_out[vd] / rms * norm_weight * silu(projected_z[z_offset + vd]);
                }
            }

            for out_dim in 0..self.hidden_size {
                let mut acc = 0.0f32;
                for col in 0..value_dim {
                    unsafe {
                        acc += core[col]
                            * (*self.out_proj_ptr.ptr.add(out_dim * value_dim + col)).to_f32();
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

fn l2_norm(values: &[f32]) -> Vec<f32> {
    let norm = (values.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
    values.iter().map(|v| *v / norm).collect()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}
