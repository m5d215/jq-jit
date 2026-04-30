# tests/

Integration tests for jq-jit. This README is the entry point — read it
before adding a new test or trying to figure out which existing test
should grow.

For deeper invariant rules (fast-path layout, dedup, env seeding, Empty
handling) see [`docs/maintenance.md`](../docs/maintenance.md).

## Roles

Every test under `tests/` belongs to one of five roles. The role
determines the file prefix and how the test is consumed.

| Role | Prefix | What it pins | When it runs |
|---|---|---|---|
| **compatibility** | `compat_` | jq-jit matches a fixed expected output for cases anchored to upstream jq or a project issue. No external jq needed at run time. | Every `cargo test --release`. |
| **differential** | `diff_` | jq-jit produces the same value-level output as reference `jq-1.8.x` on a hand-curated corpus. Errors-on-both-sides count as agreement; one-sided errors and value mismatches fail. | Every `cargo test --release` (skipped locally if no jq-1.8.x on $PATH or `$JQ_BIN`; CI panics in that case). |
| **fuzz** | `fuzz_` | proptest-driven property assertions against `jq-1.8.x`. Same skip policy as `diff_`. | `fuzz_restricted` and `fuzz_error_wrap` run every `cargo test --release`. `fuzz_full` is `#[ignore]` — opt in with `-- --ignored`. |
| **self-diff** | `selfdiff_` | Two internal execution paths inside jq-jit produce identical output on the same filter. No external jq needed. | Every `cargo test --release`. |
| **contract / coverage / enforcement** | `contract_`, `coverage_`, `enforce_` | Unit tests that pin one fast path / one factory invariant, or meta-tests that scan the source/corpus for structural rules (every fast path is exercised, no `Value::Obj` is built outside the factories). No external jq needed. | Every `cargo test --release`. |

## Test inventory

| File | Role | Target / what it tests | Data source | External dep |
|---|---|---|---|---|
| `compat_official.rs` | compat | Upstream jq behaviour on the verbatim `jq.test` corpus | `tests/official/jq.test` | none |
| `compat_regression.rs` | compat | Issue-driven regressions and project-pinned behaviour | `tests/regression.test` | none |
| `diff_corpus.rs` | diff | Value-level parity vs `jq-1.8.x` on a curated `(filter, input)` corpus | `tests/differential/corpus.test` | jq-1.8.x |
| `diff_scenarios.rs` | diff | Real-world script scenarios (`group_by`, `walk`, k8s manifests, `@csv`, …). Adding a case = "1 .jq + 1 .json, no harness changes" | `tests/corpus/<name>.{jq,json}` | jq-1.8.x |
| `fuzz_restricted.rs` | fuzz | Random `(filter, input)` pairs from a deliberately *narrow* grammar (no `try/catch`, no `..`, integer-only inputs). Default-on. | proptest generators | jq-1.8.x |
| `fuzz_full.rs` | fuzz | Same idea, *full* grammar. `#[ignore]` because it surfaces known type-dispatch divergences (#83) until the contract lands. | proptest generators | jq-1.8.x |
| `fuzz_error_wrap.rs` | fuzz | Property: `(filter)?` produces empty output whenever `filter` would error. Locks in the bug class behind the 2026-04-26 sweep (#172). | proptest generators | jq-1.8.x |
| `selfdiff_jit_interp.rs` | selfdiff | JIT/fast-path output equals generic-interpreter output, runs the regression corpus twice with `JQJIT_FORCE_INTERPRETER=1` toggled. | `tests/regression.test` | none |
| `contract_fast_path.rs` | contract | Per-fast-path unit tests: hit, bail, edge cases. ~424 `#[test]` fns, one file per fast path category. | inline | none |
| `contract_value_factories.rs` | contract | `Value::object_from_pairs` / `_from_normalized_pairs` / number factories: dedup, repr preservation. | inline | none |
| `coverage_fast_path.rs` | coverage | Every `detect_*` fast path in `src/bin/jq-jit.rs` is hit by at least one corpus case. Runs the corpus with `JQJIT_TRACE=1` and asserts the trace coverage. | scrapes `src/`, runs `tests/differential/corpus.test`, allowlist `tests/coverage_fast_path.allowlist` | none |
| `enforce_value_factories.rs` | enforce | Greps `src/` and refuses new direct `Value::Obj(Rc::new(…))` construction sites. Allowlist enforces an exact upper *and* lower bound per file (shrink-only). | scrapes `src/`, allowlist `tests/enforce_value_factories.allowlist` | none |

`src/jit.rs` also contains an inline `#[cfg(test)] mod` for JIT internals
that don't fit any of the above roles.

## Shared helpers (`tests/common/`)

Files in `tests/common/` are not integration test binaries; they are
pulled in via `mod common;` from per-test files.

- `diff_harness.rs` — `RunOutput`, `run_filter`, `run_script_file`,
  `resolve_jq`, `require_jq`, `jq_jit_path`. Single source of truth for
  spawning `jq-jit` / reference `jq` and capturing their output.
- `json_normalize.rs` — `normalize`, `normalize_value`,
  `serialize_sorted`. Value-level JSON normalisation (sort keys, fold
  integer-valued floats to integers).
- `jq_test_format.rs` — `parse_test_file`, `run_test`, `run_jq_test_suite`,
  `TestCase`, `TestStatus`, `TestResult`. Parser + runner for the 3-line
  group format shared by `compat_official` and `compat_regression`.

When adding a new diff or compat test, reuse these — do **not** copy
`run_once` / `normalize` / `resolve_jq` into a new file.

## Data files

| Path | Format | Consumed by |
|---|---|---|
| `tests/regression.test` | 3-line group: filter / input / expected output | `compat_regression`, `selfdiff_jit_interp` |
| `tests/official/jq.test` | Same format, plus `%%FAIL` blocks (skipped) | `compat_official` |
| `tests/differential/corpus.test` | 3-line group: filter / input / (no expected; reference `jq` is the oracle) | `diff_corpus`, `coverage_fast_path` |
| `tests/corpus/<name>.{jq,json}` | filter file + input file pair | `diff_scenarios` |
| `tests/modules/` | jq library files (`a.jq`, `b/`, `c/`, `lib/`, shadow + bind-order test fixtures) | `compat_official`, `compat_regression` (via `-L`) |

The 3-line group format is documented in `tests/common/jq_test_format.rs`.

## Environment variables

| Variable | Default | Honoured by | Purpose |
|---|---|---|---|
| `JQ_BIN` | unset | every `diff_*` and `fuzz_*` test | Override the reference jq binary. Must be jq-1.8.x. |
| `CI` | (set by GitHub Actions) | every `diff_*` and `fuzz_*` test | When set and no jq-1.8.x is found, the test panics instead of skipping. |
| `JQJIT_PROPTEST_CASES` | 256 (`fuzz_restricted`) / 200 (`fuzz_error_wrap`) / 500 (`fuzz_full`) | `fuzz_*` | proptest case budget. Crank to 100k+ for nightly sweeps. |
| `JQJIT_PROPTEST_TIMEOUT_SECS` | 3 | `fuzz_*` | Per-subprocess wall-clock cap. |
| `JQJIT_TRACE` | unset | the binary, used by `coverage_fast_path` | Print `[trace] filter='…' matched=<name>` to stderr for each invocation. |
| `JQJIT_FORCE_INTERPRETER` | unset | the binary, used by `selfdiff_jit_interp` | Disable all raw-byte fast paths and JIT compilation; route through the generic interpreter. |
| `JIT_INTERP_DIFF_LIMIT` | unset | `selfdiff_jit_interp` | Truncate the regression corpus to the first N cases (local dev). |

## Decision tree: where do I put a new test case?

The same question, answered three ways depending on what you have.

### "I just fixed a bug. Where do I pin it?"

1. **Always**: add the minimal repro to `tests/regression.test`. The case
   should fail before the fix and pass after. Comment with the issue
   number.
2. If the fix touched a fast path: add a contract test in
   `tests/contract_fast_path.rs` covering hit + bail + the failure mode.
3. If a fuzz harness (`fuzz_restricted` / `fuzz_full` / `fuzz_error_wrap`)
   shrunk the failure: also widen the generator if the bug shape was
   filtered out, so the same class won't escape next time.

### "I'm adding a new fast path."

1. **Required**: at least one corpus probe in
   `tests/differential/corpus.test` that hits the path.
   `coverage_fast_path` will fail otherwise.
2. **Required**: per-shape unit tests in `tests/contract_fast_path.rs`
   (hit, every bail condition, every type-mismatch path).
3. **Recommended**: extend the `fuzz_restricted` generators in
   `tests/fuzz_restricted.rs` so random shapes can hit the new path.
4. If the path interacts with `Value::Obj` construction: route through
   the factories in `src/value.rs`; `enforce_value_factories` will fail
   if you build `Value::Obj(Rc::new(…))` directly outside the existing
   allowlist.

### "I'm adding a new behaviour the project deliberately diverges from upstream on."

1. Pin it in `tests/regression.test` (with expected output).
2. Make sure no `tests/official/jq.test` case is broken — if upstream
   has a conflicting expectation, document why this project differs in
   the regression case comment.
3. If reference jq would error and jq-jit would now succeed, add a
   negative case to `tests/differential/corpus.test` only if both sides
   error or both succeed; **do not** add cases that intentionally
   diverge to the differential corpus (it would just be a permanent
   FAIL).

### "Random fuzz found a divergence."

1. Shrink, paste the minimal `(filter, input)` into
   `tests/regression.test` with expected output set to the *correct*
   answer (jq's, in almost all cases).
2. Fix the bug.
3. If the generator produced the case via a fluke (it shouldn't have
   reached this shape in the first place), fix the generator. If the
   shape is a known-divergent class, document the filter inline in the
   appropriate `fuzz_*.rs` so future contributors don't accidentally
   widen the generator into the same trap.

## CI vs local execution

CI workflows are at `.github/workflows/ci.yml` (push/PR) and `release.yml`
(tag push). Both install jq-1.8.1 unconditionally and run
`cargo test --release` — no extra flags, no env-var overrides.

| Test | Local `cargo test --release` | CI |
|---|---|---|
| `compat_official`, `compat_regression` | always | always |
| `contract_*`, `coverage_fast_path`, `enforce_value_factories`, `selfdiff_jit_interp` | always | always |
| `diff_corpus`, `diff_scenarios`, `fuzz_restricted`, `fuzz_error_wrap` | runs if jq-1.8.x is found, else skipped (eprintln) | always (CI installs jq-1.8.1) |
| `fuzz_full` | `#[ignore]` — needs `-- --ignored` | not run |

To run a single test: `cargo test --release --test <file_stem>`. Examples:

```bash
cargo test --release --test compat_regression
cargo test --release --test fuzz_restricted -- --nocapture
JQJIT_PROPTEST_CASES=100000 cargo test --release --test fuzz_restricted
cargo test --release --test fuzz_full -- --ignored --nocapture
```

## Glossary

- **fast path** — `detect_*` / `is_*` raw-byte optimisation in
  `src/bin/jq-jit.rs` that skips JSON parsing entirely. Detailed in
  `docs/maintenance.md` § Fast path map.
- **interpreter path** — generic tree-walking `Filter::execute_cb` in
  `src/interpreter.rs`. Used as the oracle in `selfdiff_jit_interp`.
- **JIT path** — Cranelift-compiled filters. Self-diff'd against the
  interpreter on every regression case.
