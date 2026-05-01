---
name: release
description: Cut a new jq-jit release — pre-flight, bench, version bump, tag, push. Single PR.
---

# release — jq-jit release workflow

Cut a new release from the current branch. Bundles benchmark capture,
`Cargo.toml` bump, and `docs/benchmark-history.{tsv,md}` updates into a
single PR; tags & pushes after auto-merge so the existing release
workflow (`.github/workflows/release.yml`) builds binaries, creates the
GitHub Release, and bumps the Homebrew tap formula.

Run **autonomously** end-to-end. Do not pause for confirmation unless
the regression gate triggers (see step 4).

## Invocation

This skill is **not bound to a slash command** — `/release` resolves to
the user-global aeonnext release skill at `~/.claude/skills/release/`,
not this one. Invoke it via the `Skill` tool (the description above
matches user intent like "リリースして" / "release" / "release v1.5.0"
in the jq-jit repo).

## Determining the version

Extract from user input:

| User says | Behavior |
|---|---|
| "release" / "リリース" (no other arg) | Auto-detect bump type from commits since last tag |
| "release patch" / "patch でリリース" | Force patch bump (e.g. v1.4.4 → v1.4.5) |
| "release minor" / "minor リリース" | Force minor bump (v1.4.4 → v1.5.0) |
| "release major" / "major リリース" | Force major bump (v1.4.4 → v2.0.0) |
| "release v1.5.0" / "v1.5.0 でリリース" | Explicit version |

Auto-detection scans `git log v<latest>..HEAD --pretty=%s` for
Conventional Commits prefixes:

- Any `feat!:`, `fix!:`, `<type>!:` or `BREAKING CHANGE:` body → **major**
- Any `feat:` → **minor**
- Otherwise → **patch**

## Workflow

### 1. Pre-flight

- Confirm cwd is repo root (`git rev-parse --show-toplevel`).
- `cargo build --release` — must succeed with **zero warnings**. If
  warnings appear, fix them (or stop and surface to user) before
  continuing. Zero warnings is a hard project rule (see CLAUDE.md).
- `cargo test --release` — must pass.

### 2. Determine version

Apply the rule from the Args table to compute target `vX.Y.Z`. Then:

- Verify no tag named `vX.Y.Z` already exists: `git tag -l vX.Y.Z`.
- Branch handling:
  - On `main` → create `git checkout -b release/vX.Y.Z`.
  - On any other branch → stay (release will bundle with current WIP).
    Note any uncommitted changes; if working tree is dirty with edits
    unrelated to the release, prefer to commit/stash them first
    (ask the user if intent is unclear).

### 3. Benchmark + history update

```
python3 .claude/skills/release/scripts/update_history.py vX.Y.Z
```

Run it in the **background** (`run_in_background: true`). The script
runs `bench/comprehensive.sh` (~10 min), parses output, appends rows to
`docs/benchmark-history.tsv`, and regenerates the slim
`docs/benchmark-history.md` (last 5 columns).

Do not poll — wait for the completion notification.

### 4. Regression gate

After the bench step, compare the new `vX.Y.Z` column against the
previous version's column in `docs/benchmark-history.tsv`. Compute
`ratio = new / previous` per benchmark (skip rows where the previous
column is empty — they're new patterns with no baseline).

| Max ratio | Action |
|---|---|
| ≤ 1.05 | Continue to step 5. |
| 1.05–1.30 | **Re-run** the bench once (`update_history.py` again — note this appends a *second* `vX.Y.Z` column; remove duplicates from TSV before re-running, or run with a temp version label and replace). If still in this range, treat as confirmed regression: investigate the cause (`git log v<previous>..HEAD --oneline` for suspects, code-level inspection of the regressed benchmarks). Apply the fix and re-run bench. If the fix is too involved, open a GitHub issue with `gh issue create` describing the regressed benchmarks + suspected cause, then continue the release with the regression on record. |
| > 1.30 | **Stop.** Show the user which benchmarks regressed and by how much, and ask how to proceed. Do not bump or PR. |

If the bench produced any `FAIL/TIMEOUT` entries, suspect bench-data
drift before code regression — historically this has been the cause
(e.g. #316: `ltrim+tonum+arith` failing because the data prefix didn't
match the filter expectation).

### 5. Cargo.toml bump

```
sed -i '' -E 's/^version = "[^"]+"/version = "X.Y.Z"/' Cargo.toml
cargo check --release   # fast verify it parses
```

### 6. Single commit

Stage and commit all release-related changes together:

- `Cargo.toml`
- `docs/benchmark-history.tsv`
- `docs/benchmark-history.md`

Conventional Commits message: `chore: release vX.Y.Z`. If the branch
already had unrelated WIP commits, this is the final commit on top of
them.

### 7. PR + auto-merge

```
gh pr create --title "release vX.Y.Z" --body "<short summary>"
gh pr merge --auto --merge
```

Body should include:
- One line summary of what's in the release (high-level, not file diff)
- A note that history columns were appended for vX.Y.Z
- Bench summary line (e.g. "no regression > 5%" or "regressed: <list>, see #issue")

Wait for auto-merge in the **background** — do not poll synchronously.

### 8. Tag + push

After auto-merge completes:

```
git checkout main && git pull --ff-only
git tag -a vX.Y.Z -m "release vX.Y.Z"
git push origin vX.Y.Z
```

Tag push triggers `.github/workflows/release.yml`, which builds
binaries for `linux-x86_64` and `macos-arm64`, creates the GitHub
Release with auto-generated notes, and bumps the
[homebrew-tap](https://github.com/m5d215/homebrew-tap) formula. No
manual edits to either are needed.

### 9. Wrap-up

Report to the user:
- PR number and merge SHA
- Tag pushed
- Link to release Actions run (`gh run list --workflow=release.yml -L 1`)
- Bench delta summary (best speedup / worst regression vs previous version)

## Constraints

- The `bench/comprehensive.sh` run is the long pole (~10 min). Use
  `run_in_background: true` and let the notification fire — never poll
  with `sleep`.
- Never amend or force-push. If a step fails after commit, fix forward
  with a new commit.
- Pre-flight `cargo build --release` warnings are blocking — investigate
  and fix, never `--allow warnings` or similar.
- If on a non-main branch with WIP, the release commits ride on top of
  whatever's there. The user invoking `/release` from such a branch is
  asserting that bundling is intentional.

## Files this skill writes to

- `Cargo.toml` (version field only)
- `docs/benchmark-history.tsv` (append rows)
- `docs/benchmark-history.md` (regenerate from TSV)
- `.git/` (commit, tag, push)
