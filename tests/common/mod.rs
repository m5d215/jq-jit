//! Shared test helpers used across the integration test suite.
//!
//! Each integration test under `tests/` is a separate binary. Files in this
//! directory are only compiled when a test imports them via `mod common;`.
//!
//! - `diff_harness` — spawn `jq-jit` and reference `jq-1.8.x`, capture
//!   stdout/exit-code, resolve the reference binary on $JQ_BIN / Homebrew /
//!   $PATH. Used by every test that compares against external jq.
//! - `filter_strategy` — shared `FilterExpr` / `JsonShape` AST + proptest
//!   combinators used by `fuzz_restricted`, `metamorphic`, and (#686)
//!   `fuzz_axis_*` harnesses. Single source of truth for the AST and the
//!   conservative base; harnesses layer their own weights and exotics.
//! - `json_normalize` — value-level JSON normalisation (sort keys, fold
//!   integer-valued floats) so equality is semantic, not textual. Used by
//!   both diff tests and the official/regression compat suites.
//! - `jq_test_format` — parser + runner for the 3-line group format
//!   (`filter / input / expected_output`) shared by `tests/official/jq.test`
//!   and `tests/regression.test`.
//! - `parallel` — `par_map`, a dep-free order-preserving parallel map used to
//!   fan the spawn-bound differential harnesses across CPU cores.
//!
//! `#[allow(dead_code)]` is applied because each integration test imports
//! only the subset of helpers it needs; unused ones look dead from the
//! per-binary compilation unit.

#![allow(dead_code)]

pub mod diff_harness;
pub mod filter_strategy;
pub mod jq_test_format;
pub mod json_normalize;
pub mod parallel;
