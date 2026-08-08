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
- One conversation, one PR. Everything the founder asks for during a
  conversation goes on the branch that is already open — **including work that
  is unrelated to what the PR started as**. If nothing is open yet, the first
  piece of work opens the PR and everything after it joins that PR. Push to that
  branch and update the title and description to describe what it now contains.

  Don't widen a PR's scope on your own: if you *spot* something adjacent, say so
  and ask. But once it has been asked for, it belongs on the open branch.

  This holds even when the new work is obviously a different concern.
  "These are unrelated, so they're cleaner as separate PRs" is the reasoning
  that keeps producing the wrong answer, and it is not a judgment call to make
  here. If you genuinely believe something has to ship separately, **ask first
  and wait for an answer** — do not open the second PR and explain afterwards.

  Release PRs are not an exception. A `release/vX.Y.Z` branch is still the open
  branch: a fix asked for while it is open goes onto it and ships in that
  release. Whether the release should go out without that extra work is the
  founder's call, not a reason to start a new branch.

  Start a second branch only when there is no open PR, or when the founder says
  to.