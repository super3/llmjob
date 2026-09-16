# Instructions for Claude

Read this file at the start of every conversation and follow it.

## Writing

Write plainly. Short sentences, ordinary words, no marketing language. Say what
changed, why, and what it means for the user. If you can cut a sentence without
losing information, cut it.

This applies to commit messages, PR titles and descriptions, code comments, and
what you say in chat.

## Tests

Run the tests before you start work and again when you finish. Use `npm test` in
the directory you changed: the repo root for the server, `earn/` for the desktop
app. The job isn't done until every test passes.

## Git

- Don't commit unless I ask. "Commit", "push", and "save" mean I'm asking.
- "Push" means commit and push.
- Any other time: make the edit, stop, and wait for me.
- Before you commit, show me `git status` so I can see what's included.
- Write commit messages that explain why you made the change, not just what you
  changed.

I want to decide what goes into the repo, which is the point of all of the above.

## Pull requests

**Keep the title and description accurate.** Every time you push to a branch with
an open PR, update that PR's title and description to match what's on the branch
now: what it covers, the test counts, anything notable you added, reverted, or
rebased.

**Always include the preview URL** when you give me a PR link. It comes from the
PR number: `https://llmjob-llmjob-pr-<PR-number>.up.railway.app`. Add a page path
when you mean a specific page, like `/chat.html` or `/network.html`.

**One conversation, one PR.** Everything I ask for during a conversation goes on
the branch that is already open, even when it has nothing to do with what the PR
started as. If no PR is open, the first thing I ask for opens one and everything
after it joins that PR. Push to that branch and update the title and description.

Don't grow a PR on your own. If you notice something worth fixing nearby, tell me
and ask. Once I've asked for it, it goes on the open branch.

This still applies when the new work is clearly a different topic. Don't decide
that unrelated things are cleaner as separate PRs — that reasoning keeps giving
the wrong answer, and it isn't your call. If you really think something has to
ship on its own, ask me and wait for an answer. Don't open the second PR and
explain afterwards.

Release PRs work the same way. A `release/vX.Y.Z` branch is still the open
branch, so a fix I ask for while it's open goes on it and ships in that release.
Whether to hold the release for that extra work is my decision, not a reason to
start a new branch.

Only start a second branch when no PR is open, or when I tell you to.

### When a PR's preview is broken by its own old deploys

This is the one exception to "one conversation, one PR", and it isn't the
"they're unrelated" reasoning above. No work gets split. The same branch and the
same commits move to a new PR number because the old PR's preview environment is
unusable.

Close the PR and open a new one **from the same branch**. Preview environments
are tied to the PR number, so a new PR gets a clean one with a fresh database.
Don't try to fix it in code.

Use this whenever a PR's preview keeps failing because of data an earlier deploy
of that same PR wrote. Big changes hit it most. The usual cause is a migration
that got renamed or renumbered during a rebase: `node-pg-migrate` compares the
migrations the database has already run against the files on disk, in order, and
refuses to run when they don't match. The preview then fails forever on a
migration name that only that database ever recorded. GitHub Actions stays green
the whole time, which is how you can tell — CI builds from the files, the preview
carries the old database.

Never fix this by renaming the migration back to match the database, and never by
adding `--no-check-order` to `npm start`. The first breaks the production deploy
instead, and the second turns off the check that stops a migration being skipped
in production.

In the new PR, say that it replaces the closed one and link the two, so the
review history is still readable.
