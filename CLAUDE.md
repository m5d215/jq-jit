# jq-jit

JIT compilation implementation of jq.

## Commands

```bash
cargo build --release            # Build (must have zero warnings)
cargo test --release             # Test (official 509 + regression)
./bench/run.sh                   # Benchmark
./bench/comprehensive.sh --quick # Benchmark (vs past results, see docs/benchmark-history.md)
```

## Issue Fix Workflow

Follow this workflow for bug fixes:

1. **Create branch**: Branch from main with `fix/<brief-description>`
2. **Fix + add tests**: Add regression test cases to `tests/regression.test`
3. **Verify build**: Zero warnings with `cargo build --release`, all tests pass with `cargo test --release`
4. **Verify performance**: Run `./bench/comprehensive.sh` and compare against the latest results in `docs/benchmark-history.md` to ensure no regression
5. **Commit**: Use Conventional Commits format (in English)
6. **Push + create PR**: Link the issue with `Closes #N`
7. **Wait for Actions**: Confirm all CI checks pass before merging
8. **Merge**: Use merge commit
9. **Close issue**: Rely on automatic closing via `Closes` keyword
10. **Update local**: `git checkout main && git pull`

## Test Format

`tests/regression.test` uses a 3-line group format:

```
# Comment (issue number and description)
filter_expression
input_json
expected_output
```
