//! Shared test helpers used across the integration test suite.
//!
//! Each integration test under `tests/` is a separate binary. Files in this
//! directory are only compiled when a test imports them via `mod common;`.
//!
//! - `diff_harness` — spawn `jq-jit` and reference `jq-1.8.x`, capture
//!   stdout/exit-code, resolve the reference binary on $JQ_BIN / Homebrew /
//!   $PATH. Used by every test that compares against external jq.
//! - `json_normalize` — value-level JSON normalisation (sort keys, fold
//!   integer-valued floats) so equality is semantic, not textual. Used by
//!   both diff tests and the official/regression compat suites.
//! - `jq_test_format` — parser + runner for the 3-line group format
//!   (`filter / input / expected_output`) shared by `tests/official/jq.test`
//!   and `tests/regression.test`.
//!
//! `#[allow(dead_code)]` is applied because each integration test imports
//! only the subset of helpers it needs; unused ones look dead from the
//! per-binary compilation unit.

#![allow(dead_code)]

pub mod diff_harness;
pub mod jq_test_format;
pub mod json_normalize;
