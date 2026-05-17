---
name: release
description: Cut a new jq-jit release — pre-flight, bench, version bump, tag, push. Single PR.
---

# release — jq-jit release workflow

Cut a release from the current branch in **one PR**: benchmark capture +
`Cargo.toml` bump + `docs/benchmark-history.{tsv,md}`. Tag & push after
auto-merge; `.github/workflows/release.yml` then builds binaries, creates
the GitHub Release, and bumps the Homebrew tap.

Run **autonomously** end-to-end. Only pause if the regression gate
(step 4) exceeds 1.30.

## Version

Extract the target from user input:

| User says | Behavior |
|---|---|
| "release" / "リリース" (no arg) | Auto-detect bump from commits since last tag |
| "release patch/minor/major" | Force that bump |
| "release v1.5.0" | Explicit version |

Auto-detect scans `git log v<latest>..HEAD --pretty=%s`:
`feat!:` / `<type>!:` / `BREAKING CHANGE:` → **major**; `feat:` →
**minor**; otherwise → **patch**.

## Workflow

**1. Pre-flight.** cwd = repo root (`git rev-parse --show-toplevel`).
`cargo build --release` → **zero warnings** (hard project rule; fix or
stop, never suppress). `cargo test --release` → pass.

**2. Version + branch.** Compute `vX.Y.Z`; ensure `git tag -l vX.Y.Z`
is empty. On `main` → `git checkout -b release/vX.Y.Z`. On any other
branch → stay (release bundles with current WIP, which is intentional);
commit/stash unrelated dirty edits first (ask if intent is unclear).

**3. Benchmark.** Run in the **background** (`run_in_background: true`):

```
python3 .claude/skills/release/scripts/update_history.py vX.Y.Z
```

It runs `bench/comprehensive.sh` (~10 min), appends to
`docs/benchmark-history.tsv`, and regenerates the slim `.md` (last 5
columns). Never poll — wait for the completion notification.

**4. Regression gate.** Per benchmark, `ratio = new / previous` (skip
rows with an empty previous column — new patterns, no baseline).

| Max ratio | Action |
|---|---|
| ≤ 1.05 | Continue. |
| 1.05–1.30 | Re-run bench once (this appends a *second* `vX.Y.Z` column — dedup the TSV or use a temp label first). Still in range → confirmed regression: investigate (`git log v<prev>..HEAD --oneline`, inspect the regressed benchmarks), fix + re-bench. Too involved → `gh issue create` for the regression, then continue the release on record. |
| > 1.30 | **Stop.** Show the user which benchmarks regressed and by how much; don't bump or PR. |

`FAIL/TIMEOUT` entries → suspect bench-data drift before code regression
(e.g. #316: `ltrim+tonum+arith` failed because the data prefix didn't
match the filter).

**5. Bump.** `sed -i '' -E 's/^version = "[^"]+"/version = "X.Y.Z"/'
Cargo.toml`, then `cargo check --release` to verify it parses.

**6. Commit.** One commit `chore: release vX.Y.Z` staging `Cargo.toml`
+ `docs/benchmark-history.tsv` + `docs/benchmark-history.md` (on top of
any pre-existing WIP commits).

**7. PR + auto-merge.**

```
gh pr create --title "release vX.Y.Z" --body "<summary>"
gh pr merge --auto --merge
```

Body: one-line release summary, a note that history columns were
appended for vX.Y.Z, and a bench line ("no regression > 5%" or
"regressed: <list>, see #issue"). Wait for auto-merge in the
**background** — don't poll.

**8. Tag + push.** After auto-merge completes:

```
git checkout main && git pull --ff-only
git tag -a vX.Y.Z -m "release vX.Y.Z"
git push origin vX.Y.Z
```

Tag push triggers `release.yml`: binaries for `linux-x86_64` /
`macos-arm64`, GitHub Release with auto-generated notes, and Homebrew
tap formula bump — all automatic, no manual edits.

**9. Wrap-up.** Report: PR # + merge SHA, tag pushed, release Actions
run (`gh run list --workflow=release.yml -L 1`), bench delta (best
speedup / worst regression vs the previous version).

## Constraints

- Never amend or force-push. If a step fails after commit, fix forward
  with a new commit.
- Never poll the bench with `sleep`; rely on the background completion
  notification.
- **Post-`git bisect` (e.g. step 4 investigation): `git bisect reset`
  does NOT rebuild.** The leftover `target/release/jq-jit` belongs to
  whatever commit bisect evaluated last. Always `cargo build --release`
  before any re-bench or perf comparison — a stale binary once nearly
  shipped a wrong v1.5.0 regression analysis (`67a28be`→`5d80c22`).
  Verify with `md5 -q target/release/jq-jit` if unsure.
