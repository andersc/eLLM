pub mod common;
pub mod generic;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
