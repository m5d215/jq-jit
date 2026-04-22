# jq-jit

JIT compilation implementation of jq.

## Commands

```bash
cargo build --release            # Build (must have zero warnings)
cargo test --release             # Test (official 509 + regression)
./bench/run.sh                   # Benchmark
./bench/comprehensive.sh --quick # Benchmark (vs past results, see docs/benchmark-history.md)
```

## Language Policy

Write all GitHub artifacts in English:

- Issue titles and bodies
- Pull request titles and descriptions
- Commit messages (Conventional Commits)
- Code comments and documentation in the repository

This applies regardless of the conversation language used with the assistant.

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

## Maintenance Notes

When hunting jq compatibility bugs or adding new optimizations, read
`docs/maintenance.md` first — it catalogs the fast-path layout, the
invariants each must preserve (dedup, env seeding, Empty handling, path
guards), and the diff-loop method that reliably surfaces divergences.

## Test Format

`tests/regression.test` uses a 3-line group format:

```
# Comment (issue number and description)
filter_expression
input_json
expected_output
```
