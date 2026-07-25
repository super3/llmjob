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
Repo root: `C:\Users\template\Code\llmjob`. Earn tests run from `earn/`, server tests from the repo root. `npm test` and `npm --prefix earn test` work from the Bash tool; if the `.bin/jest` shim fails under node, fall back to `node node_modules/jest/bin/jest.js`.

`eslint` has been missing from the installed `node_modules` before, so `npm run lint` dies with "not recognized" while the suites pass. It's a declared devDependency — `npm install --no-save eslint@^9 @eslint/js@^9 globals@^17` fixes it without touching `package.json`.

## Phase 1 — rebase, draft PR, launch (do this now)

1. **Sync to main.** `git fetch origin`. If on a stale/merged branch, `git checkout main && git pull origin main --ff-only`. If already on a `release/*` branch with the bump, `git rebase origin/main` and force-push with `--force-with-lease` instead of re-cutting.

2. **Pick the version.** `NEW` = `$ARGUMENTS` if given, else the latest `vX.Y.Z` git tag (`git tag | sort -V | tail -1`) with the patch incremented. Never reuse an existing tag. If the range since the last tag is large or carries headline features (check `git log --oneline --no-merges vPREV..HEAD` before committing to a number), still follow the patch default — but say so in your report so the founder can redirect to a minor. Re-cutting is cheap before the tag and awkward after it.

3. **Create the release branch:** `git checkout -b release/v$NEW` off the up-to-date main.

4. **Bump versions — two files, both required:**
   - `earn/package.json` → `"version": "$NEW"` (the app and its installers).
   - `site/config.json` → `"appVersion": "$NEW"` (the site's download links).

   Then verify by building rather than by grepping the source: `npm run build:site`, and confirm `dist/earn.html` carries six `v$NEW` download URLs (4× `.exe`, 2× `.AppImage`) with no previous version anywhere in the file.

   Do **not** go looking for literal version strings in the page to hand-edit — there are none. The links live in `site/pages/earn.html` (moved out of the repo root) and are templated as `{{!appVersion}}`, so the build substitutes the single `site/config.json` value into all six URLs. This replaced an earlier hand-edited arrangement that went stale twice, lagging at v0.2.7 through two releases; the templating fixes that structurally. If you find yourself editing six URLs by hand, you are on a stale checkout.

5. **Run tests — must be green.** If a suite errors on a missing module (e.g. `jest-environment-jsdom`), the local `node_modules` is stale: run `npm install` in `earn/` then retry. Earn and server suites must both pass at the 100% coverage gate before proceeding.

6. **Commit & push:** commit `Release v$NEW`, `git push -u origin release/v$NEW`. End the commit body with the standard `Co-Authored-By:` trailer naming the model you are actually running (e.g. `Claude Opus 5 <noreply@anthropic.com>`) — don't copy a model version out of this file, it goes stale.

7. **Open the PR** titled `Release v$NEW`. Body: summarize everything merged since the previous tag — `git log --oneline vPREV..HEAD` — grouped into meaningful buckets (features/fixes/tooling), with the earn + server test counts. Keep it accurate (per the repo's PR rule), and include the Railway preview URL alongside the PR link as CLAUDE.md requires: `https://llmjob-llmjob-pr-<N>.up.railway.app`. The PR number isn't known until `gh pr create` returns, so if you put the preview URL in the body up front, check the number you guessed and `gh pr edit` it if it differs.

8. **Launch the build on this machine to test.** Stop any old instances (`electron.exe`, `LLMJob Earn.exe`, `llama-server.exe`, `alpha-miner*`), then `cd earn && npm start -- --remote-debugging-port=9223 &`. Wait for the CDP page target, then bring the window to the foreground (PowerShell `SetForegroundWindow`).

   Don't click Start yourself. But **expect the app to resume mining on its own** — with a valid address and `mode: auto` persisted in `%APPDATA%\llmjob-earn\settings.json`, it starts the engine and the LLM at launch. That's the shipped behaviour, so report it rather than fighting it; a user's machine will do the same.

   Then verify the build rather than just reporting that a window appeared. Screenshot it and look at the image — a blank frame is a failed launch. Read the spawned processes' real arguments back off the OS (`(Get-CimInstance Win32_Process -Filter "ProcessId=<PID>").CommandLine`) and confirm `llama-server` carries a real `--main-gpu <index>`; check `curl -s http://127.0.0.1:8080/health`; and once a report lands (~60s), fetch `https://llmjob-production.up.railway.app/api/miners` and confirm the row's `version` is `$NEW`. That last check is the one that proves you launched the build you think you did, rather than a stale artifact.

## Phase 2 — publish (after the founder merges)

9. **Arm the auto-tag poll now** (background) so publishing fires the moment he merges — don't wait for him to say "merged". Poll `gh pr view <N> --json state,mergeCommit` every 30s; on `MERGED`:
   - Read the merged version **robustly** — `node -e "JSON.parse(require('child_process').execSync('git show <sha>:earn/package.json').toString()).version"`. (A prior poll used `readFileSync(0)` from a pipe, which returned empty and falsely bailed.)
   - If it isn't `$NEW`, stop and report — never tag an unverified version.
   - Otherwise `git tag v$NEW <mergeCommit> && git push origin v$NEW` (the `v` prefix is required). The `v*` tag triggers `.github/workflows/miner-build.yml`, which builds and publishes the installers.
   - If the poll dies on a flaky read, just tag manually — the merge + version are what matter.
   - **If arming the poll is denied by the permission classifier**, don't try to route around it. Say plainly that publishing is not armed, finish and report Phase 1, and tag manually once the founder says he's merged (or once he grants the permission and you re-arm). A release that publishes itself through a worked-around denial is worse than one that waits for a word.

10. **Confirm the publish.** Watch the tag's `miner-build.yml` run to completion, then verify the GitHub Release `v$NEW` is published (not a draft) with the Windows `.exe` + blockmap, Linux `.AppImage`, CLI, HiveOS packages, and `latest.yml` / `latest-linux.yml`. Report the release URL. Existing installs auto-update via `latest.yml`.

## Notes

- Only the founder merges; you never merge the release PR yourself.
- If `git push` is rejected for lacking the `workflow` scope, keep `.github/workflows/*` changes out of the release commit (the release bump shouldn't touch them anyway).
- This mirrors the established flow and the `cut-release-poll-merge` guidance: prepare + push, poll every 30s, auto-tag on merge, confirm the release published.
