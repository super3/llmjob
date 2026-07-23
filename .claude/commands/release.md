---
description: Cut a new LLMJob Earn release — rebase, draft the release PR, launch the build locally to test; then auto-tag + publish once the founder merges
---

Run the LLMJob Earn release workflow. The founder wants: **you rebase, draft the release PR, and launch the build on this machine to test. He merges the PR. Then you publish** (tag → CI builds & publishes the installers).

Optional argument in `$ARGUMENTS`: an explicit version (e.g. `0.3.0` for a minor bump). With no argument, bump the **patch** of the latest release tag.

## Environment

Prepend the portable toolchain to PATH in every shell:
```bash
export PATH="$HOME/AppData/Local/nodejs:$HOME/AppData/Local/gh/bin:$PATH"
```
Repo root: `C:\Users\template\Code\llmjob`. Run earn tests from `earn/`, server tests from the repo root, both with `node node_modules/jest/bin/jest.js` (the `.bin/jest` shim fails under node).

## Phase 1 — rebase, draft PR, launch (do this now)

1. **Sync to main.** `git fetch origin`. If on a stale/merged branch, `git checkout main && git pull origin main --ff-only`. If already on a `release/*` branch with the bump, `git rebase origin/main` and force-push with `--force-with-lease` instead of re-cutting.

2. **Pick the version.** `NEW` = `$ARGUMENTS` if given, else the latest `vX.Y.Z` git tag (`git tag | sort -V | tail -1`) with the patch incremented. Never reuse an existing tag.

3. **Create the release branch:** `git checkout -b release/v$NEW` off the up-to-date main.

4. **Bump versions:**
   - `earn/package.json` → `"version": "$NEW"`.
   - `earn.html` download links — these have gone **stale before** (they lagged two releases at v0.2.7). Detect the version actually in the download URLs (`grep -oE 'releases/download/v[0-9.]+' earn.html | head -1`) and replace **that** version string with `$NEW` everywhere (6 refs: 2×main .exe + 2×menu .exe + 2×.AppImage). Confirm `grep -c "$NEW" earn.html` is 6 and no old version remains.

5. **Run tests — must be green.** If a suite errors on a missing module (e.g. `jest-environment-jsdom`), the local `node_modules` is stale: run `npm install` in `earn/` then retry. Earn and server suites must both pass at the 100% coverage gate before proceeding.

6. **Commit & push:** commit `Release v$NEW` (mention if it's catching the download links up), `git push -u origin release/v$NEW`. End the commit body with the standard `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.

7. **Open the PR** titled `Release v$NEW`. Body: summarize everything merged since the previous tag — `git log --oneline vPREV..HEAD` — grouped into meaningful buckets (features/fixes/tooling), with the earn + server test counts. Keep it accurate (per the repo's PR rule). Link the PR back to the user.

8. **Launch the build on this machine to test.** Stop any old instances (`electron.exe`, `LLMJob Earn.exe`, `llama-server.exe`, `alpha-miner*`), then `cd earn && npm start -- --remote-debugging-port=9223 &`. Wait for the CDP page target, then bring the window to the foreground (PowerShell `SetForegroundWindow`). Tell the user it's up and note anything worth checking. Leave it at the idle/ready state — don't auto-start mining unless asked.

## Phase 2 — publish (after the founder merges)

9. **Arm the auto-tag poll now** (background) so publishing fires the moment he merges — don't wait for him to say "merged". Poll `gh pr view <N> --json state,mergeCommit` every 30s; on `MERGED`:
   - Read the merged version **robustly** — `node -e "JSON.parse(require('child_process').execSync('git show <sha>:earn/package.json').toString()).version"`. (A prior poll used `readFileSync(0)` from a pipe, which returned empty and falsely bailed.)
   - If it isn't `$NEW`, stop and report — never tag an unverified version.
   - Otherwise `git tag v$NEW <mergeCommit> && git push origin v$NEW` (the `v` prefix is required). The `v*` tag triggers `.github/workflows/miner-build.yml`, which builds and publishes the installers.
   - If the poll dies on a flaky read, just tag manually — the merge + version are what matter.

10. **Confirm the publish.** Watch the tag's `miner-build.yml` run to completion, then verify the GitHub Release `v$NEW` is published (not a draft) with the Windows `.exe` + blockmap, Linux `.AppImage`, CLI, HiveOS packages, and `latest.yml` / `latest-linux.yml`. Report the release URL. Existing installs auto-update via `latest.yml`.

## Notes

- Only the founder merges; you never merge the release PR yourself.
- If `git push` is rejected for lacking the `workflow` scope, keep `.github/workflows/*` changes out of the release commit (the release bump shouldn't touch them anyway).
- This mirrors the established flow and the `cut-release-poll-merge` guidance: prepare + push, poll every 30s, auto-tag on merge, confirm the release published.
