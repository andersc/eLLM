use std::arch::aarch64::*;
use std::f16;

use crate::init::matmul_params::MatMulParams;

#[inline(always)]
pub unsafe fn matmul_block(a: *const f16, b_panel: *const f16, c: *mut f16, param: &MatMulParams) {
    if param.a_row_step_micro == 3 && param.b_row_step_micro == 32 {
        matmul_3x32_fp16(a, b_panel, c, param);
    } else {
        crate::kernel::generic::matmul_block::matmul_block(a, b_panel, c, param);
    }
}

#[target_feature(enable = "fp16")]
unsafe fn matmul_3x32_fp16(
    a: *const f16,
    b_panel: *const f16,
    c: *mut f16,
    param: &MatMulParams,
) {
    let lda = param.a_row_step_macro;
    let ldc = param.b_row_step_macro;
    let kc = param.column_step_macro;
    let b_stride = 32usize;

    let a0 = a;
    let a1 = a.add(lda);
    let a2 = a.add(2 * lda);

    let mut c00 = vld1q_f16(c);
    let mut c01 = vld1q_f16(c.add(8));
    let mut c02 = vld1q_f16(c.add(16));
    let mut c03 = vld1q_f16(c.add(24));

    let mut c10 = vld1q_f16(c.add(ldc));
    let mut c11 = vld1q_f16(c.add(ldc + 8));
    let mut c12 = vld1q_f16(c.add(ldc + 16));
    let mut c13 = vld1q_f16(c.add(ldc + 24));

    let mut c20 = vld1q_f16(c.add(2 * ldc));
    let mut c21 = vld1q_f16(c.add(2 * ldc + 8));
    let mut c22 = vld1q_f16(c.add(2 * ldc + 16));
    let mut c23 = vld1q_f16(c.add(2 * ldc + 24));

    for k in 0..kc {
        let b_row = b_panel.add(k * b_stride);
        let b0 = vld1q_f16(b_row);
        let b1 = vld1q_f16(b_row.add(8));
        let b2 = vld1q_f16(b_row.add(16));
        let b3 = vld1q_f16(b_row.add(24));

        let a0k = vdupq_n_f16(*a0.add(k));
        c00 = vfmaq_f16(c00, a0k, b0);
        c01 = vfmaq_f16(c01, a0k, b1);
        c02 = vfmaq_f16(c02, a0k, b2);
        c03 = vfmaq_f16(c03, a0k, b3);

        let a1k = vdupq_n_f16(*a1.add(k));
        c10 = vfmaq_f16(c10, a1k, b0);
        c11 = vfmaq_f16(c11, a1k, b1);
        c12 = vfmaq_f16(c12, a1k, b2);
        c13 = vfmaq_f16(c13, a1k, b3);

        let a2k = vdupq_n_f16(*a2.add(k));
        c20 = vfmaq_f16(c20, a2k, b0);
        c21 = vfmaq_f16(c21, a2k, b1);
        c22 = vfmaq_f16(c22, a2k, b2);
        c23 = vfmaq_f16(c23, a2k, b3);
    }

    vst1q_f16(c, c00);
    vst1q_f16(c.add(8), c01);
    vst1q_f16(c.add(16), c02);
    vst1q_f16(c.add(24), c03);

    vst1q_f16(c.add(ldc), c10);
    vst1q_f16(c.add(ldc + 8), c11);
    vst1q_f16(c.add(ldc + 16), c12);
    vst1q_f16(c.add(ldc + 24), c13);

    vst1q_f16(c.add(2 * ldc), c20);
    vst1q_f16(c.add(2 * ldc + 8), c21);
    vst1q_f16(c.add(2 * ldc + 16), c22);
    vst1q_f16(c.add(2 * ldc + 24), c23);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_neon_matmul_block_matches_reference() {
        let kc = 64;
        let lda = 64;
        let ldc = 32;
        let params = MatMulParams {
            a_row_step_macro: lda,
            b_row_step_macro: ldc,
            column_step_macro: kc,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let mut a = vec![0.0f16; 3 * lda];
        let mut b = vec![0.0f16; kc * 32];
        let mut got = vec![0.0f16; 3 * ldc];
        let mut expected = vec![0.0f16; 3 * ldc];

        for i in 0..3 {
            for k in 0..kc {
                a[i * lda + k] = (((i + k) % 17) as f32 * 0.01) as f16;
            }
        }
        for k in 0..kc {
            for j in 0..32 {
                b[k * 32 + j] = (((k + j) % 19) as f32 * 0.01) as f16;
            }
        }

        unsafe {
            matmul_block(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), &params);
        }
        crate::kernel::generic::matmul_block::matmul_block(
            a.as_ptr(),
            b.as_ptr(),
            expected.as_mut_ptr(),
            &params,
        );

        for (g, e) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*g as f32, *e as f32, epsilon = 0.02);
        }
    }
}
