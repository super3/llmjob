# Instructions for Claude

## Start of Conversation

Always read this CLAUDE.md file at the start of each conversation to ensure you follow project-specific rules and workflows.

## Testing Requirements

Always run tests before starting work and after completing tasks. A task is NOT complete unless all tests pass. Use `npm test` in the appropriate directory to verify code quality and functionality.

## Git Workflow Rules

- NEVER commit unless explicitly requested with words like "commit", "push", or "save"
- When user says "Push" - this means commit AND push
- When making edits, just edit and stop - don't commit
- After making changes, wait for user's next instruction
- If asked to commit, show what will be committed first (git status)
- Use descriptive commit messages that explain the "why" not just the "what"
- Never commit or push changes unless explicitly requested
- This maintains control over what gets pushed to the repository

## Pull Request Rules

- Always keep the PR title and description accurate. Whenever you push a commit
  to a branch with an open PR, update that PR's title and description so they
  reflect what's actually on the branch (scope, test counts, notable changes,
  and anything added, reverted, or rebased).
- Whenever you share a pull request link, always include its Railway preview URL
  right alongside the PR URL. The preview URL is deterministic from the PR
  number: `https://llmjob-llmjob-pr-<PR-number>.up.railway.app` (append a page
  path like `/chat.html` or `/network.html` when pointing at a specific page).
- One conversation, one PR. Don't widen a PR's scope on your own — if you spot
  something adjacent, say so and ask. But once it's asked for, it belongs on the
  branch that's already open, not on a new one. Follow-up work in the same
  conversation goes into the same open PR: push to that branch and update the
  title and description, rather than spinning up a second PR for what is really
  one piece of work. "Don't expand scope silently" is about asking first, not
  about splitting the answer across PRs.

### When a preview deploy is wedged by its own history

The one case where a second PR is right, and the exception to "one conversation,
one PR" above. Close the PR and open a new one from the same branch. The preview
environment is keyed by PR number, so a new PR provisions a clean one — including
a fresh database. Do not try to repair it in code.

This is the fix whenever a PR's preview keeps failing on state an *earlier deploy
of that same PR* wrote, which large changes hit most often. The usual trigger is
a migration renamed or renumbered during a rebase: `node-pg-migrate` compares the
database's applied-migration list positionally against the files on disk and
refuses to run when they diverge, so the preview fails forever on a migration
name only that environment ever recorded. GitHub Actions stays green throughout,
which is the tell — CI builds from the tree, the preview carries state.

Never resolve it by renaming the migration back to match the stale row, or by
adding `--no-check-order` to `npm start`. Both trade a broken preview for a
broken production deploy or a disabled safeguard; the ordering check is what
stops a migration from being skipped in production.

In the replacement PR, say that it supersedes the closed one and link them, so
the review history stays followable.
