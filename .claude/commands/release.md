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

Both toolchain directories are portable, hand-extracted installs, and those PATH entries persist in the user environment even when the directories are gone — so "command not found" does **not** mean the PATH is wrong; check whether the directory actually exists. Rebuild by extracting the official zips (`nodejs.org/dist/vX/node-vX-win-x64.zip`, `github.com/cli/cli/releases/.../gh_X_windows_amd64.zip`) into those exact paths.

Install **Node 22**, not the newest LTS. CI (`test.yml`, `miner-build.yml`, `deploy.yml`) pins 22, and both `package.json` files declare `engines.node >= 22`. Node 24 also ships npm 11, which **blocks package install scripts by default** and silently skips native postinstalls.

`gh` may not be authenticated — check `gh auth status` before Phase 2 rather than at the point you need it. `gh auth login` needs a browser and cannot run non-interactively, so ask the founder to run it; **don't** try to read the token back out of Git Credential Manager, which is blocked by the permission classifier and should stay that way. `git push` works regardless, since Credential Manager already holds a credential — so tagging and publishing never depend on `gh` auth, only the PR edit and the merge poll do.

`eslint` has been missing from the installed `node_modules` before, so `npm run lint` dies with "not recognized" while the suites pass. It's a declared devDependency — `npm install --no-save eslint@^9 @eslint/js@^9 globals@^17` fixes it without touching `package.json`.

## Phase 1 — rebase, draft PR, launch (do this now)

1. **Sync to main.** `git fetch origin`. If on a stale/merged branch, `git checkout main && git pull origin main --ff-only`. If already on a `release/*` branch with the bump, `git rebase origin/main` and force-push with `--force-with-lease` instead of re-cutting.

2. **Pick the version.** `NEW` = `$ARGUMENTS` if given, else the latest `vX.Y.Z` git tag (`git tag | sort -V | tail -1`) with the patch incremented. Never reuse an existing tag. If the range since the last tag is large or carries headline features (check `git log --oneline --no-merges vPREV..HEAD` before committing to a number), still follow the patch default — but say so in your report so the founder can redirect to a minor. Re-cutting is cheap before the tag and awkward after it.

3. **Create the release branch:** `git checkout -b release/v$NEW` off the up-to-date main.

4. **Bump versions — three files:**
   - `earn/package.json` → `"version": "$NEW"` (the app and its installers).
   - `earn/package-lock.json` → the **two** `version` fields near the top (the root object and `packages[""]`). Running `npm install` in `earn/` syncs them for you; editing both by hand is fine when npm isn't available. Don't skip it: v0.3.9 bumped the lock and v0.3.8 didn't, so the lock has silently disagreed with the manifest before.
   - `site/config.json` → `"appVersion": "$NEW"` (the site's download links).

   Then verify by building rather than by grepping the source: `npm run build:site`, and confirm `dist/earn.html` carries six `v$NEW` download URLs (4× `.exe`, 2× `.AppImage`) with no previous version anywhere in the file.

   Do **not** go looking for literal version strings in the page to hand-edit — there are none. The links live in `site/pages/earn.html` (moved out of the repo root) and are templated as `{{!appVersion}}`, so the build substitutes the single `site/config.json` value into all six URLs. This replaced an earlier hand-edited arrangement that went stale twice, lagging at v0.2.7 through two releases; the templating fixes that structurally. If you find yourself editing six URLs by hand, you are on a stale checkout.

5. **Run tests — must be green.** If a suite errors on a missing module (e.g. `jest-environment-jsdom`), the local `node_modules` is stale: run `npm install` in `earn/` then retry. Earn and server suites must both pass at the 100% coverage gate before proceeding.

   But not every missing-module error is stale `node_modules`. If jest names a file that demonstrably exists on disk —

   ```
   Validation Error: Module ./server/tests/setup.js in the setupFilesAfterEnv option was not found.
   ```

   — check the C runtime before you touch the repo. Jest 30's default resolver is the native `unrs-resolver`, which cannot `dlopen` without the MSVC runtime, and it fails by reporting a *missing setup file* rather than a missing DLL. `npm ci`, reinstalling, and changing Node version all leave it broken, and plain `node`/`fs` resolves the same path fine, so everything points at the repo instead of the box. Confirm with `node -e "require('unrs-resolver')"`. The fix needs no admin: copy `vcruntime140.dll`, `vcruntime140_1.dll` and `msvcp140.dll` from `earn/vendor/llm-runtime/` — the repo already vendors them, for exactly this reason, for `llama-server` — into `$HOME/AppData/Local/nodejs`, which is on PATH and survives `npm ci`.

6. **Commit & push:** commit `Release v$NEW`, `git push -u origin release/v$NEW`. End the commit body with the standard `Co-Authored-By:` trailer naming the model you are actually running (e.g. `Claude Opus 5 <noreply@anthropic.com>`) — don't copy a model version out of this file, it goes stale.

7. **Open the PR** titled `Release v$NEW`. Body: summarize everything merged since the previous tag — `git log --oneline vPREV..HEAD` — grouped into meaningful buckets (features/fixes/tooling), with the earn + server test counts. Keep it accurate (per the repo's PR rule), and include the Railway preview URL alongside the PR link as CLAUDE.md requires: `https://llmjob-llmjob-pr-<N>.up.railway.app`. The PR number isn't known until `gh pr create` returns, so if you put the preview URL in the body up front, check the number you guessed and `gh pr edit` it if it differs.

8. **Launch the build on this machine to test.** Stop any old instances (`electron.exe`, `LLMJob Earn.exe`, `llama-server.exe`, `alpha-miner*`), then `cd earn && npm start -- --remote-debugging-port=9223 &`. Wait for the CDP page target, then bring the window to the foreground (PowerShell `SetForegroundWindow`).

   Leave it idle — don't click Start unless asked. The app does **not** auto-start: `applyPlan` runs only from the `miner:start` IPC handler, so a freshly launched instance sits at START with `0m 00s` uptime no matter what `settings.json` holds. If you find it mining, someone clicked it; don't record that as the app's own behaviour.

   Then verify the build rather than just reporting that a window appeared. Screenshot it and **look at the image** — a blank frame is a failed launch, and the uptime/START-vs-STOP state tells you whether anything is actually running.

   The deeper checks all require the app to be started, so they are gated on the founder actually testing it (or on him saying go ahead): read the spawned processes' real arguments back off the OS (`(Get-CimInstance Win32_Process -Filter "ProcessId=<PID>").CommandLine`) and confirm `llama-server` carries a real `--main-gpu <index>`; check `curl -s http://127.0.0.1:8080/health`; and once a report lands (~60s) fetch `https://llmjob-production.up.railway.app/api/miners` and confirm the row's `version` is `$NEW`. That last one is what proves you launched the build you think you did rather than a stale artifact — but note the client only reports to the board **while mining**, so an idle instance will never produce a row, and its absence means nothing.

## Phase 2 — publish (after the founder merges)

9. **Arm the auto-tag poll now** (background) so publishing fires the moment he merges — don't wait for him to say "merged". Poll `gh pr view <N> --json state,mergeCommit` every 30s; on `MERGED`:
   - Read the merged version **robustly** — `node -p "JSON.parse(require('child_process').execSync('git show <sha>:earn/package.json').toString()).version"`. Note `-p`, not `-e`: **`node -e` evaluates without printing**, so it returns an empty string, the "could not read the version" guard fires, and the poll hands the job back on every single release. That is exactly what happened on v0.3.7 and v0.3.8 — the guard was right to refuse a version it could not read, but it could never read one. Whatever form you use, check it actually prints before trusting it; an earlier attempt used `readFileSync(0)` from a pipe and failed the same silent way.
   - If it isn't `$NEW`, stop and report — never tag an unverified version.
   - Otherwise `git tag v$NEW <mergeCommit> && git push origin v$NEW` (the `v` prefix is required). The `v*` tag triggers `.github/workflows/miner-build.yml`, which builds and publishes the installers.
   - If the poll dies on a flaky read, just tag manually — the merge + version are what matter.
   - **If arming the poll is denied by the permission classifier**, don't try to route around it. Say plainly that publishing is not armed, finish and report Phase 1, and tag manually once the founder says he's merged (or once he grants the permission and you re-arm). A release that publishes itself through a worked-around denial is worse than one that waits for a word.

10. **Confirm the publish.** Watch the tag's `miner-build.yml` run to completion, then verify the GitHub Release `v$NEW` is published (not a draft) with the Windows `.exe` + blockmap, Linux `.AppImage`, CLI, HiveOS packages, and `latest.yml` / `latest-linux.yml`. Report the release URL. Existing installs auto-update via `latest.yml`.

11. **Write the release notes — the body is empty until you do.** The release is created by `electron-builder --publish always` (`miner-build.yml`), which publishes it with a **blank body**, titled just `$NEW`. Nothing in the workflow ever writes notes, so the changelog is empty unless you fill it in. Every release through v0.3.9 shipped that way. This is the last step of the release, not an optional extra.

    - Build the notes from `git log --oneline --no-merges vPREV..v$NEW`, grouped into the same buckets as the PR body — but write them for **someone running a rig**, not as raw commit titles. "Accepted shares and worker names stay stable when a card ages out" beats "Keep shares and worker names stable when a card ages out"; a bare `git log` dump is not release notes.
    - Lead with the install/update line: existing installs auto-update via `latest.yml`, new installs want the `LLMJob-Earn-Setup-$NEW.exe` / `LLMJob-Earn-$NEW.AppImage` names.
    - Close with the compare link: `https://github.com/super3/llmjob/compare/vPREV...v$NEW`.
    - Apply it and fix the bare-number title at the same time:
      ```bash
      gh release edit v$NEW --title "LLMJob Earn v$NEW" --notes-file <file>
      ```
    - Verify it took — `gh release view v$NEW --json name,isDraft,body` — rather than trusting the command's exit code.

    If you'd rather fix this at the source, it needs a workflow step after the Linux publish that runs `gh release edit` (with `--generate-notes`, or from a checked-in changelog). That's a separate PR off `main` — don't fold a `.github/workflows/` change into the release commit.

## Notes

- Only the founder merges; you never merge the release PR yourself.
- If `git push` is rejected for lacking the `workflow` scope, keep `.github/workflows/*` changes out of the release commit (the release bump shouldn't touch them anyway).
- This mirrors the established flow and the `cut-release-poll-merge` guidance: prepare + push, poll every 30s, auto-tag on merge, confirm the release published.
