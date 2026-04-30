//! Issue-driven regression suite.
//!
//! Runs `tests/regression.test`, a project-curated corpus where each case
//! is anchored to a fixed expected output (and usually to a GitHub issue).
//! Same file format as the upstream suite — see
//! `tests/common/jq_test_format.rs`.
//!
//! Add a case here whenever a bug fix lands or a behaviour the project
//! deliberately relies on needs pinning, even if upstream has no equivalent.

mod common;

#[test]
fn regression_test_suite() {
    let test_file = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/regression.test");
    common::jq_test_format::run_jq_test_suite("Regression Test Suite Results", test_file);
}
