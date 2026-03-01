//! jq-jit: JIT compiler for jq bytecode.
//!
//! This library exposes the core modules for use by benchmarks and tests.

#[allow(dead_code)]
pub mod bytecode;
pub mod codegen;
pub mod compiler;
pub mod cps_ir;
#[allow(dead_code)]
pub mod jq_ffi;
pub mod jq_runner;
pub mod runtime;
pub mod aot;
pub mod cache;
pub mod input;
pub mod output;
pub mod value;
