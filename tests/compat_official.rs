//! Upstream jq compatibility suite.
//!
//! Runs the verbatim test corpus from `tests/official/jq.test` (imported
//! from jq-1.8.x) against jq-jit. Each case is a 3-line group: filter,
//! input, expected output. See `tests/common/jq_test_format.rs` for the
//! parser and runner.
//!
//! Failures here mean jq-jit diverged from documented upstream behaviour
//! on a case the project of record cares about.

mod common;

#[test]
fn official_jq_test_suite() {
    let test_file = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/official/jq.test");
    common::jq_test_format::run_jq_test_suite("jq Official Test Suite Results", test_file);
}
