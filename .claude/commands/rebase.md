---
description: Rebase the current branch onto the latest default branch, re-verify (build + tests + lint), force-push with lease, and keep the PR accurate
---

Rebase the current working branch onto the latest default branch (`main`), resolve
any conflicts, re-verify the result, and update the open PR. Optional
`$ARGUMENTS`: a base branch other than `main` (e.g. `develop`).

Do **not** run this on `main` itself. If the branch's PR is already **merged**,
stop and say so — follow-up work belongs on a fresh branch off `main`, not a
re-rebase of merged history.

## Steps

1. **Confirm the branch.** `git status` — note the current branch (call it
   `$BRANCH`) and make sure the working tree is clean. If there are uncommitted
   changes, stop and ask; don't rebase over them.

2. **Fetch the base.** `git fetch origin main` (or the branch named in
   `$ARGUMENTS`). Retry up to 4× with exponential backoff (2s, 4s, 8s, 16s) on
   network errors.

3. **Check divergence.** `git rev-list --left-right --count origin/main...HEAD`.
   If the branch is 0 behind, there's nothing to rebase — skip to step 7 (still
   worth re-verifying if `main` moved under a shared file).

4. **Rebase.** `git rebase origin/main`.
   - On conflicts: open each conflicted file, resolve by **keeping both intents**
     (this repo's `main` moves fast — pages were migrated from repo root into
     `site/pages/` with shared `site/partials/`, so a "deleted by them" on a root
     HTML file usually means your edit belongs in the moved `site/pages/` copy).
     Then `git add` the resolved files and `git rebase --continue`.
   - If the rebase is genuinely ambiguous (both sides changed the same logic),
     stop and ask before picking a side.
   - `git rebase --abort` returns to the pre-rebase state if you need a clean slate.

5. **Re-verify** the rebased tree — the base may have changed things your branch
   depends on:
   - `npm run build:site` — the marketing/dashboard pages build from `site/`
     into `dist/`; make sure nothing you moved left an unrendered `{{…}}` token or
     a broken include.
   - `npm test` — the suite enforces a **100% coverage** gate; it must stay green.
   - `npm run lint`.
   Fix anything the rebase broke before pushing.

6. **Push.** `git push --force-with-lease -u origin $BRANCH`. Use
   `--force-with-lease` (never a plain `--force`) so a concurrent push isn't
   clobbered. Retry with the same backoff on network errors.

7. **Keep the PR accurate.** If an open PR exists for `$BRANCH`, make sure its
   title and description still match what's on the branch (scope, test counts,
   anything the rebase added or dropped). Don't post a comment unless something
   genuinely needs calling out.

Report what happened: how many commits were replayed, any conflicts you resolved,
and the final CI-relevant state (build/tests/lint).
