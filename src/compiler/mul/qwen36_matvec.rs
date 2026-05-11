use std::f16;

use crate::compiler::assign::assign;
use crate::kernel::generic::from_f32::FromF32;

pub trait Scalar: Copy + Default + FromF32 + Send + Sync {
    fn to_f32(self) -> f32;

    unsafe fn matvec_rows(
        input_ptr: *const Self,
        weight_ptr: *const Self,
        output_ptr: *mut f32,
        row_begin: usize,
        row_end: usize,
        input_dim: usize,
    ) {
        scalar_matvec_rows(input_ptr, weight_ptr, output_ptr, row_begin, row_end, input_dim);
    }

    unsafe fn matvec_f32_input_rows(
        input_ptr: *const f32,
        weight_ptr: *const Self,
        output_ptr: *mut f32,
        row_begin: usize,
        row_end: usize,
        input_dim: usize,
    ) {
        scalar_matvec_f32_input_rows(
            input_ptr, weight_ptr, output_ptr, row_begin, row_end, input_dim,
        );
    }
}

impl Scalar for f16 {
    fn to_f32(self) -> f32 {
        self as f32
    }

    unsafe fn matvec_rows(
        input_ptr: *const Self,
        weight_ptr: *const Self,
        output_ptr: *mut f32,
        row_begin: usize,
        row_end: usize,
        input_dim: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("f16c")
                && std::is_x86_feature_detected!("fma")
            {
                let mut input = vec![0.0f32; input_dim];
                convert_f16_to_f32_avx2(input_ptr, input.as_mut_ptr(), input_dim);
                matvec_f32_input_f16_rows_avx2(
                    input.as_ptr(),
                    weight_ptr,
                    output_ptr,
                    row_begin,
                    row_end,
                    input_dim,
                );
                return;
            }
        }

        scalar_matvec_rows(input_ptr, weight_ptr, output_ptr, row_begin, row_end, input_dim);
    }

    unsafe fn matvec_f32_input_rows(
        input_ptr: *const f32,
        weight_ptr: *const Self,
        output_ptr: *mut f32,
        row_begin: usize,
        row_end: usize,
        input_dim: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("f16c")
                && std::is_x86_feature_detected!("fma")
            {
                matvec_f32_input_f16_rows_avx2(
                    input_ptr, weight_ptr, output_ptr, row_begin, row_end, input_dim,
                );
                return;
            }
        }

        scalar_matvec_f32_input_rows(
            input_ptr, weight_ptr, output_ptr, row_begin, row_end, input_dim,
        );
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

pub unsafe fn matvec<T: Scalar>(
    input_ptr: *const T,
    weight_ptr: *const T,
    output: &mut [f32],
    output_dim: usize,
    input_dim: usize,
) {
    T::matvec_rows(
        input_ptr,
        weight_ptr,
        output.as_mut_ptr(),
        0,
        output_dim,
        input_dim,
    );
}

pub unsafe fn matvec_parallel<T: Scalar>(
    input_ptr: *const T,
    weight_ptr: *const T,
    output: &mut [f32],
    output_dim: usize,
    input_dim: usize,
    cpu_num: usize,
) {
    let cpu_num = cpu_num.max(1).min(output_dim.max(1));
    if cpu_num == 1 || output_dim < 128 {
        matvec(input_ptr, weight_ptr, output, output_dim, input_dim);
        return;
    }

    let input_addr = input_ptr as usize;
    let weight_addr = weight_ptr as usize;
    let output_addr = output.as_mut_ptr() as usize;

    std::thread::scope(|scope| {
        for thread_id in 0..cpu_num {
            let Some((begin, end)) = assign(output_dim, cpu_num, thread_id) else {
                continue;
            };
            scope.spawn(move || unsafe {
                T::matvec_rows(
                    input_addr as *const T,
                    weight_addr as *const T,
                    output_addr as *mut f32,
                    begin,
                    end,
                    input_dim,
                );
            });
        }
    });
}

pub unsafe fn matvec_f32_input<T: Scalar>(
    input: &[f32],
    weight_ptr: *const T,
    output: &mut [f32],
    output_dim: usize,
    input_dim: usize,
) {
    T::matvec_f32_input_rows(
        input.as_ptr(),
        weight_ptr,
        output.as_mut_ptr(),
        0,
        output_dim,
        input_dim,
    );
}

pub unsafe fn matvec_f32_input_parallel<T: Scalar>(
    input: &[f32],
    weight_ptr: *const T,
    output: &mut [f32],
    output_dim: usize,
    input_dim: usize,
    cpu_num: usize,
) {
    let cpu_num = cpu_num.max(1).min(output_dim.max(1));
    if cpu_num == 1 || output_dim < 128 {
        matvec_f32_input(input, weight_ptr, output, output_dim, input_dim);
        return;
    }

    let input_addr = input.as_ptr() as usize;
    let weight_addr = weight_ptr as usize;
    let output_addr = output.as_mut_ptr() as usize;

    std::thread::scope(|scope| {
        for thread_id in 0..cpu_num {
            let Some((begin, end)) = assign(output_dim, cpu_num, thread_id) else {
                continue;
            };
            scope.spawn(move || unsafe {
                T::matvec_f32_input_rows(
                    input_addr as *const f32,
                    weight_addr as *const T,
                    output_addr as *mut f32,
                    begin,
                    end,
                    input_dim,
                );
            });
        }
    });
}

unsafe fn scalar_matvec_rows<T: Scalar>(
    input_ptr: *const T,
    weight_ptr: *const T,
    output_ptr: *mut f32,
    row_begin: usize,
    row_end: usize,
    input_dim: usize,
) {
    for row in row_begin..row_end {
        let mut acc = 0.0f32;
        for col in 0..input_dim {
            acc +=
                (*input_ptr.add(col)).to_f32() * (*weight_ptr.add(row * input_dim + col)).to_f32();
        }
        output_ptr.add(row).write(acc);
    }
}

unsafe fn scalar_matvec_f32_input_rows<T: Scalar>(
    input_ptr: *const f32,
    weight_ptr: *const T,
    output_ptr: *mut f32,
    row_begin: usize,
    row_end: usize,
    input_dim: usize,
) {
    for row in row_begin..row_end {
        let mut acc = 0.0f32;
        for col in 0..input_dim {
            acc += *input_ptr.add(col) * (*weight_ptr.add(row * input_dim + col)).to_f32();
        }
        output_ptr.add(row).write(acc);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "f16c")]
unsafe fn convert_f16_to_f32_avx2(input_ptr: *const f16, output_ptr: *mut f32, len: usize) {
    use std::arch::x86_64::{_mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps, __m128i};

    let mut i = 0usize;
    while i + 8 <= len {
        let half = _mm_loadu_si128(input_ptr.add(i) as *const __m128i);
        let value = _mm256_cvtph_ps(half);
        _mm256_storeu_ps(output_ptr.add(i), value);
        i += 8;
    }
    while i < len {
        output_ptr.add(i).write((*input_ptr.add(i)) as f32);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "f16c")]
#[target_feature(enable = "fma")]
unsafe fn matvec_f32_input_f16_rows_avx2(
    input_ptr: *const f32,
    weight_ptr: *const f16,
    output_ptr: *mut f32,
    row_begin: usize,
    row_end: usize,
    input_dim: usize,
) {
    use std::arch::x86_64::{
        _mm_loadu_si128, _mm256_cvtph_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps,
        _mm256_storeu_ps, __m128i,
    };

    let mut lanes = [0.0f32; 8];
    for row in row_begin..row_end {
        let row_ptr = weight_ptr.add(row * input_dim);
        let mut acc = _mm256_setzero_ps();
        let mut col = 0usize;
        while col + 8 <= input_dim {
            let input = _mm256_loadu_ps(input_ptr.add(col));
            let weight_half = _mm_loadu_si128(row_ptr.add(col) as *const __m128i);
            let weight = _mm256_cvtph_ps(weight_half);
            acc = _mm256_fmadd_ps(input, weight, acc);
            col += 8;
        }

        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum = lanes.iter().sum::<f32>();
        while col < input_dim {
            sum += *input_ptr.add(col) * (*row_ptr.add(col) as f32);
            col += 1;
        }
        output_ptr.add(row).write(sum);
    }
}
