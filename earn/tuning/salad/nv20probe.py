"""nv20probe.py: runs Turing's price probe (earn/native/probes/turing/price75.cu) on a Salad PC's card, one run a
line, and samples the card meanwhile. For nv20box.sh (branch goal/nv-20): measuring tools, not the miner.

  python3 nv20probe.py <price75 exe> <list>

A line of the list (';' between lines when it comes from the environment): <kernel> <seconds> [tiles a launch].
Each run prints one line: the probe's own result (pipe = IMMA.8816 a clock an SM over the peak of one), then the
means of the card's readings from 8 s on (the last 20 at most): SM clock, watts, temperature, memory-controller
use, and in how many samples the power cap and a thermal slowdown were on. "count" is the rate by count: SMs x
1024 MACs a clock x the clock x pipe, in TH/s. No "%" or "|" in a line: Salad's log keeps such a line empty.
"""
import importlib.util
import os
import re
import subprocess
import sys
import threading
import time

# The SM count and the card's sampled readings, shared with the other probe. Loaded by path: under python3 -I
# this file's folder is not on sys.path.
_spec = importlib.util.spec_from_file_location("nvprobe", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                       "nvprobe.py"))
_nvprobe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nvprobe)
sms, sample = _nvprobe.sms, _nvprobe.sample


def summary(out, rows, n_sm):
    keep = [v for t, v in rows if t >= 8.0 and v[3] >= 90][-20:]
    if not keep:
        return out + " / no samples"
    n = len(keep)
    mean = [sum(v[i] for v in keep) / n for i in range(8)]
    m = re.search(r"pipe=([\d.]+)", out)
    count = " count %.2f" % (n_sm * 1024 * mean[0] * float(m[1]) / 1e6) if m and n_sm else ""
    return out + " / %.0f MHz %.1f W %.0f C mem %.0f cap %d of %d hwslow %d thermal %d%s" % (
        mean[0], mean[1], mean[2], mean[4], sum(v[5] for v in keep), n, sum(v[6] for v in keep),
        sum(v[7] for v in keep), count)


def main():
    exe, lst = sys.argv[1:3]
    n_sm = sms()
    print("probe card has %d SMs" % n_sm, flush=True)
    for line in re.split(r"[;\n]", lst):
        args = line.split()
        if not args or args[0].startswith("#"):
            continue
        secs = float(args[1]) if len(args) > 1 else 20.0
        rows, stop = [], threading.Event()
        th = threading.Thread(target=sample, args=(stop, rows))
        th.start()
        t0 = time.time()
        try:
            r = subprocess.run([exe] + args, capture_output=True, text=True, timeout=secs + 120)
            out = (r.stdout.strip() or r.stderr.strip() or "rc %d" % r.returncode).splitlines()[-1]
            if r.returncode != 0:
                out += " [rc %d]" % r.returncode
        except subprocess.TimeoutExpired:
            out = "%s TIMEOUT" % args[0]
        took = time.time() - t0
        stop.set()
        th.join(timeout=5)
        print("%s [%.0f s]" % (summary(out, rows, n_sm).replace("%", "pc").replace("|", "/"), took), flush=True)


if __name__ == "__main__":
    main()
