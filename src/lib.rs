#![feature(test)]
#![feature(f16)]
#![feature(duration_millis_float)]
#![feature(sync_unsafe_cell)]
#![cfg_attr(all(target_arch = "aarch64", target_os = "macos"), feature(stdarch_neon_f16))]
#![feature(avx512_target_feature)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512_f16))]
#![feature(specialization)]
#![allow(incomplete_features)]
#![allow(unused_parens)]

pub mod compiler;
pub mod init;
pub mod kernel;
pub mod memory;
pub mod ptensor;
pub mod qwen3_moe;
pub mod runtime;
pub mod serving;
// pub mod llama;
