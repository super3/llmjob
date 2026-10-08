---
description: Check the latest LLMJob Discord #general messages and draft replies for the founder to post
---

Check the LLMJob Discord `#general` channel for the latest messages and surface anything that needs a reply. Runs read-only — you draft, the founder (super3) posts.

## How to read the channel

The bot token is in the `DISCORD_BOT_TOKEN` environment variable (never print it). `#general` is channel id `1522322936554455245`.

Pull the most recent ~40 messages:

```bash
curl -s -H "Authorization: Bot $DISCORD_BOT_TOKEN" \
  -H "User-Agent: DiscordBot (llmjob, 1.0)" \
  "https://discord.com/api/v10/channels/1522322936554455245/messages?limit=40"
```

Parse the JSON (it comes newest-first). For each message look at `author.username`, `timestamp`, `content`, `type`, `attachments`, and `sticker_items`.

## What to skip

- `type: 7` messages — these are "joined the server" notifications, not posts.
- Sticker-only / empty-content messages.
- Messages authored by `super3` or `super3b` (the founder himself).
- Anything he has already replied to (a super3 message after it that answers it).

## What to surface

Flag messages that are a support question, bug report, error/screenshot, setup or config problem, or a product comparison / feature request. For each one, output:

1. A one-line summary (who + what).
2. A ready-to-post reply in the founder's plain voice — plain language, no em-dashes, not overly technical.
3. **Always pair the reply with its translation.** Much of the community is Spanish-speaking: if the message is in Spanish, give the Spanish reply to post **and** an English translation underneath. If it's in English, English is enough.

If a message carries an image/attachment, note it and read the attachment URL to see the screenshot before drafting.

If nothing needs a reply, say so in one line — do not pad.

**Never post to Discord.** Drafts only; the founder posts from his own account. The bot is read-only.

## Grounding facts (keep replies accurate)

- Pearl (PRL) is GPU-only (NoisyGEMM algorithm) — it cannot be mined on CPU. CPU mining in general nets almost nothing after electricity.
- Merge mining (MDL) is retired: the app no longer offers it, so don't suggest it. Old setups that still pass an `mdl1…` address keep mining PRL.
- The in-app earnings estimate pulls live prlscan data as of v0.2.9; older builds overstated it. Real payouts are always what lands in the wallet.
- HiveOS: the flight sheet's Installation URL must be the versioned `https://github.com/super3/llmjob/releases/download/v<version>/llmjob-earn-<version>.tar.gz`. HiveOS keeps the old build while the file name stays the same, so updating a rig means changing that URL. Releases v0.5.0 to v0.5.12 have no HiveOS package. It needs the HiveOS 0.6 jammy image (the focal image's glibc is too old for the mining core). Wallet template `%WAL%` (or `%WAL%.%WORKER_NAME%`); the Pool URL field is required by HiveOS but ignored. The package mines only unless Extra config has `--mode auto`. The README's "HiveOS (flight sheet)" section has the rest.
- LLM co-mining is live but early — no extra earnings yet; it's the direction, not a promise.
- For the current version, check https://github.com/super3/llmjob/releases/latest rather than assuming.
