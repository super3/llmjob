# LLMJob node telemetry + usage shipping (llama.cpp / NVIDIA)

Two small services make a llama.cpp node light up both dashboard tables:

1. **`llmjob-agent`** — the repo's `install.sh` ping loop. Every ping
   (`POST /api/nodes/ping`, every 5 min) now carries best-effort telemetry —
   `device`, `vramTotal`, `vramUsed` (from `nvidia-smi`), `model`, `quant`
   (from the local llama.cpp server), and `tps` (latest generation speed from
   the journal). This populates the dashboard **Nodes** table.
2. **`llmjob-usage`** — `scripts/llmjob-usage.sh` tails the llama.cpp journal
   and posts one `POST /api/usage` record per completed inference, with real
   input/output token counts and generation speed. This populates the
   dashboard **Logs** table and the 24h activity chart.

Telemetry is **best-effort and self-detecting**: on a machine without a GPU,
without llama.cpp, or without journald, the missing fields are simply omitted
from the ping and the agent behaves exactly like the original 4-field ping.
The Ed25519 signing scheme is unchanged — the signature still covers only
`nodeId:timestamp`, never the telemetry body.

## Prerequisites

- `curl`, `openssl` (agent); additionally `bash`, `journalctl`, `python3`
  (usage shipper). `nvidia-smi` is optional.
- llama.cpp `llama-server` running as a systemd unit (default `llama-qwen`)
  serving its OpenAI-compatible API (default `http://127.0.0.1:8000`).
- An **API key** for the usage shipper: dashboard → **API Keys** → create —
  the key starts with `lj-`. Copy it immediately; it is shown once.
- A **join token** for the agent: dashboard → **Add Node** (the token baked
  into your personal install command).

## Install

```sh
sudo git clone https://github.com/super3/llmjob /opt/llmjob

sudo mkdir -p /etc/llmjob
sudo cp /opt/llmjob/scripts/systemd/agent.env.example /etc/llmjob/agent.env
sudo chmod 600 /etc/llmjob/agent.env
sudoedit /etc/llmjob/agent.env        # fill in LLMJOB_TOKEN and LLMJOB_API_KEY

sudo cp /opt/llmjob/scripts/systemd/llmjob-agent.service \
        /opt/llmjob/scripts/systemd/llmjob-usage.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llmjob-agent llmjob-usage
```

The units run as `User=worker` (adjust to your box). The usage unit adds
`SupplementaryGroups=systemd-journal` so it can read the `llama-qwen` journal.

### Environment variables (`/etc/llmjob/agent.env`)

| Var | Used by | Default | Purpose |
|---|---|---|---|
| `LLMJOB_SERVER` | both | `https://llmjob-production.up.railway.app` | API base |
| `LLMJOB_TOKEN` | agent | — (required) | join token |
| `LLMJOB_API_KEY` | usage shipper | — (required) | Bearer `lj-…` key for `/api/usage` |
| `LLMJOB_NODE_NAME` | both | `node-<nodeId>` / hostname | `node` field + display name |
| `LLAMA_ENDPOINT` | both | `http://127.0.0.1:8000` | llama.cpp OpenAI API |
| `LLAMA_UNIT` | both | `llama-qwen` | journald unit to read |
| `LLMJOB_APP` | usage shipper | `hermes` | `app` field in usage logs |

If `LLMJOB_API_KEY` is unset, `llmjob-usage` exits non-zero with a clear error
(visible via `systemctl status llmjob-usage`); the agent is unaffected.

## Verifying (manual checklist)

1. **Nodes table** — with the agent running, the node row should show the GPU
   name, VRAM, model and a quant pill (e.g. `Q6_K`) within one ping interval:

   ```sh
   systemctl status llmjob-agent
   journalctl -u llmjob-agent -n 5        # expect "✓ ping"
   # API check (Clerk session cookie required, or just open the dashboard):
   # GET $LLMJOB_SERVER/api/nodes -> device/vramTotal/model/quant/tps non-null
   ```

2. **Logs table** — run one inference through llama.cpp, then check the
   dashboard Logs table for a new row with the correct Input/Output/Speed and
   the 24h activity chart incrementing:

   ```sh
   curl -s http://127.0.0.1:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
   journalctl -u llmjob-usage -n 5        # no warnings expected
   ```

3. **Non-GPU host** — remove `nvidia-smi` from `PATH` and restart the agent:
   pings still succeed with only the 4 base fields; the loop does not crash.

4. **Missing API key** — comment out `LLMJOB_API_KEY` and restart
   `llmjob-usage`: it exits non-zero with a clear message; `llmjob-agent`
   keeps running.

## Notes and known approximations

- **`finish` is always `"stop"`.** The llama.cpp journal does not expose the
  real finish reason, so truncation (`length`) is not distinguishable from a
  natural stop. Acceptable for v1.
- **Future "accurate mode":** a lightweight reverse proxy in front of `:8000`
  could read the OpenAI `usage` + `timings` fields off each response, giving
  exact finish reasons and token counts without journal parsing. Not built in
  v1 because it inserts a hop into the live inference path.
- The shipper deduplicates per llama.cpp `task` id and caps its in-flight map
  at 64 tasks, so a dropped journal line cannot leak memory or double-post.
