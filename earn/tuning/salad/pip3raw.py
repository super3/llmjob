"""pip3raw.py: the rate of Pearl's fp8-scheme four-op mining pipeline on this card, without vLLM (pip3box.sh's
"raw" stage). A measuring tool: nothing here ships.

Run from miner/pearl-gemm of a pearl fp8-scheme checkout, so `tests.helpers` imports:
    python pip3raw.py [seconds] [shape ...]
A shape is m x n x k: A is m x k (the activations), B is n x k (the weights). Work is counted as the reference
miner credits it, m*n*k a launch. One "RAW" line per shape: launches a second, T-MAC/s (1e12 m*n*k a second) for
the whole pipeline with a target no tile can meet, the FP8 GEMM alone, and nvidia-smi's power, use, clock and
temperature in the middle of the run; then a second line with the rate under the tests' easy target (a hit on
every launch).
"""
import subprocess
import sys
import time
import traceback

import torch
from miner_base.devices import local_device
from tests.helpers.pipeline import MinerPipeline
from tests.helpers.preprocess import default_config, preprocess

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
shapes = sys.argv[2:] or ["8192x8192x8192", "8192x4096x4096", "2048x4096x4096", "8192x16384x4096", "8192x6144x1024"]


def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,clocks.sm,temperature.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"smi failed: {e}"


def rate(pipe, a, seconds, work):
    """(launches a second, T-MAC/s, launches that left a hit, nvidia-smi mid-run) over `seconds`."""
    it = hits = 0
    mid = ""
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(8):
            pipe.run(a)
            it += 1
        torch.cuda.synchronize()
        if pipe.hit_signal.read_hit() is not None:
            hits += 1
        if not mid and time.time() - t0 > seconds / 2:
            mid = smi()
    dt = time.time() - t0
    return it / dt, it * work / dt / 1e12, hits, mid


dev = local_device()
print(f"RAW device {torch.cuda.get_device_name()} cc {torch.cuda.get_device_capability()} committed {dev} "
      f"torch {torch.__version__} cuda {torch.version.cuda}", flush=True)
for shape in shapes:
    m, n, k = (int(x) for x in shape.split("x"))
    try:
        torch.manual_seed(7)
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        t0 = time.time()
        ctx = preprocess(b, config=default_config(k, device=dev))
        pipe = MinerPipeline(ctx, m)
        pipe.run(a)
        torch.cuda.synchronize()
        compile_s = time.time() - t0
        _, easy, easy_hits, _ = rate(pipe, a, secs / 2, m * n * k)
        pipe.threshold.zero_()
        pipe.hit_signal.reset_hit()
        lps, hard, hard_hits, mid = rate(pipe, a, secs, m * n * k)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            pipe.stage_gemm()
        torch.cuda.synchronize()
        gemm = m * n * k / ((time.time() - t0) / 20) / 1e12
        # Two short lines a shape: Salad's log service dropped the one 330-character line this used to be.
        print(f"RAW {shape} tile {ctx.config.tile_rows}x{ctx.config.tile_cols}: {hard:.2f} T-MAC/s whole pipeline, "
              f"{gemm:.2f} gemm alone, {lps:.1f} launches/s, smi {mid}", flush=True)
        print(f"RAW {shape} more: {easy:.2f} T-MAC/s with a hit every launch, hits {hard_hits} and {easy_hits}, "
              f"compile {compile_s:.0f}s", flush=True)
        del pipe, ctx, a, b
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        print(f"RAW {shape} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        traceback.print_exc()
        torch.cuda.empty_cache()
