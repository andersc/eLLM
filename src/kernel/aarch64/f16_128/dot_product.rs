use std::arch::aarch64::*;
use std::f16;
use std::ptr;

#[inline(always)]
pub fn dot_product(input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16, length: usize) {
    unsafe {
        let product = dot_product_accumulate(input_ptr1, input_ptr2, *output_ptr, length);
        ptr::write(output_ptr, product);
    }
}

#[target_feature(enable = "fp16")]
unsafe fn dot_product_accumulate(
    input_ptr1: *const f16,
    input_ptr2: *const f16,
    initial: f16,
    length: usize,
) -> f16 {
    let mut acc = vdupq_n_f16(0.0f16);
    let mut i = 0usize;

    while i + 8 <= length {
        let a = vld1q_f16(input_ptr1.add(i));
        let b = vld1q_f16(input_ptr2.add(i));
        acc = vfmaq_f16(acc, a, b);
        i += 8;
    }

    let mut lanes = [0.0f16; 8];
    vst1q_f16(lanes.as_mut_ptr(), acc);

    let mut sum = initial;
    for lane in lanes {
        sum = sum + lane;
    }
    while i < length {
        sum = sum + *input_ptr1.add(i) * *input_ptr2.add(i);
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_neon_dot_product_matches_reference() {
        let a: Vec<f16> = (0..19).map(|i| (i as f32 * 0.125) as f16).collect();
        let b: Vec<f16> = (0..19).map(|i| ((19 - i) as f32 * 0.0625) as f16).collect();
        let mut got = 1.0f16;
        let mut expected = 1.0f16;

        dot_product(a.as_ptr(), b.as_ptr(), &mut got, a.len());
        crate::kernel::generic::dot_product::dot_product(
            a.as_ptr(),
            b.as_ptr(),
            &mut expected,
            a.len(),
        );

        assert_abs_diff_eq!(got as f32, expected as f32, epsilon = 0.01);
    }
}
