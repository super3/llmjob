---
name: release-testing
description: Run the Earn pre-release test plan on a real Windows GPU box — the checks CI structurally cannot make (real driver, real spawned processes, real VRAM contention, real round trip to the deployed board). Use before cutting a release, or when asked to verify a GPU/LLM/board change end-to-end on live hardware.
---

Execute the phases in [earn/RELEASE-TESTING.md](../../../earn/RELEASE-TESTING.md).
That document is the source of truth for what each phase checks and why — read it
first and follow it rather than working from memory. This file is only the
procedure for running it.

Optional `$ARGUMENTS`: a phase or range (`4`, `2-4`, `0-8`). Default is 0–8, plus
phase 9 for whichever entry points the release touches.

## Before starting

1. **Confirm the box has a GPU.** `nvidia-smi`. No driver means no run — say so
   and stop; every phase below assumes real hardware. Note the card count: with
   one GPU the fleet paths in phases 2/4/6 are only partially exercised (see the
   doc's "Known limitation" section) and the report must say so.
2. **Confirm what is under test.** `earn/package.json` → `version`, and
   `git log --oneline -1`. Every version-reporting check must show this build; a
   stale artifact serving old code is the easiest false pass there is.
3. **Launch from the right place.** Source: `npm --prefix earn start`. Packaged
   (phase 8): the installed app. These are different code paths — `resourcesPath`
   resolution differs — so never substitute one for the other.

## Running

Work phases in order. **Stop at the first failure in phases 0–4** and report:
those are foundational, and a failure there makes every later result
meaningless. A failure in 5–9 is worth recording and continuing past, since
those phases are independent of each other.

Prefer direct observation over the app's own display — the point of this plan is
to catch the app lying about itself:

- process arguments: `(Get-CimInstance Win32_Process -Filter "ProcessId=<PID>").CommandLine`
- LLM liveness: `curl -s http://127.0.0.1:8080/health` and `/v1/models`
- GPU truth: `nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader`
- board truth: fetch `https://llmjob-production.up.railway.app/api/miners` and
  assert the row field by field against what the box is actually doing
- settings: `%APPDATA%\llmjob-earn\settings.json`

To read the GUI, screenshot the window and **look at it** — a blank frame is a
failed launch, not a pass. Drive the UI (start, stop, switch modes, send a chat
message); launching alone proves only that the entrypoint resolves.

## Hand back to the human

These need a person and cannot be faked. Ask, wait, and mark the phase blocked
rather than skipping it silently:

- occupying VRAM to trip the gate (phase 1) if no easy allocator is at hand
- a pairing token from the dashboard (phase 7)
- installing on a **clean box without the dev toolchain** (phase 8) — a machine
  with Visual Studio hides exactly the DLL failure this phase exists to catch
- publishing a release to exercise auto-update (phase 8)
- a genuine multi-GPU rig for any fleet change (the doc's known limitation)

## Reporting

Give a phase-by-phase pass/fail, and for each failure the command run, what was
expected, and what actually happened. Then state plainly whether the build is
good to ship.

Two rules for the report:

- **A skipped phase is a decision, not an omission.** Name it and why.
- **Never report a phase as passing on evidence you did not gather.** "Not run"
  and "passed" are different outcomes, and on a release checklist the difference
  is the whole point.

If a phase fails, do not fix the bug as part of this run unless asked — report
it, since a fix mid-run invalidates the phases already completed.
