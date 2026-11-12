"""nv30probe.py: runs the rotation fold's probe (earn/native/probes/hand89/gprobe.cu) on a Salad PC's card, one
run a line, and samples the card meanwhile. For nv30box.sh (branch goal/nv-30): measuring tools, not the miner.

  python3 nv30probe.py <gprobe exe> <cubin dir> <list>

A line of the list (';' between lines when it comes from the environment):
  <cubin|-> <name> <k> <tiles a block> [secs] [check tiles] [order] [fake] [rot]
"-" runs the kernel as ptxas compiled it (the gather fold); otherwise <cubin dir>/<cubin>.cubin is launched
(PD_CUBIN). Each run prints one line: the probe's own result, then, for runs of 10 s or more, the means of the
card's readings from 8 s on (the last 20 at most): SM clock, watts, temperature, memory-controller use, and in how
many samples the power cap and a thermal slowdown were on. "count" is the rate by count: SMs x 1024 MACs a cycle x
the clock x the share of the pipe the probe measured, in TH/s (the 4090's factor 0.13107 is 128 SMs of this).
"""
import ctypes
import os
import re
import subprocess
import sys
import threading
import time

Q = ("clocks.sm,power.draw,temperature.gpu,utilization.gpu,utilization.memory,"
     "clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown")


def sms():
    try:
        cu = ctypes.CDLL("libcuda.so.1")
        n, dev = ctypes.c_int(0), ctypes.c_int(0)
        if cu.cuInit(0) == 0 and cu.cuDeviceGet(ctypes.byref(dev), 0) == 0 and \
                cu.cuDeviceGetAttribute(ctypes.byref(n), 16, dev) == 0:
            return n.value
    except OSError:
        pass
    return 0


def sample(stop, rows):
    p = subprocess.Popen(["nvidia-smi", "-i", "0", "--query-gpu=" + Q, "--format=csv,noheader,nounits", "-lms", "1000"],
                         stdout=subprocess.PIPE, text=True)
    t0 = time.time()
    while not stop.is_set():
        line = p.stdout.readline()
        if not line:
            break
        f = [x.strip() for x in line.split(",")]
        try:
            rows.append((time.time() - t0, [float(x) for x in f[:5]] + [1.0 if x == "Active" else 0.0 for x in f[5:8]]))
        except ValueError:
            pass
    p.kill()


def main():
    exe, cdir, lst = sys.argv[1:4]
    n_sm = sms()
    print("probe card has %d SMs" % n_sm, flush=True)
    for line in re.split(r"[;\n]", lst):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        env = dict(os.environ)
        env.pop("PD_CUBIN", None)
        if parts[0] != "-":
            env["PD_CUBIN"] = os.path.join(cdir, parts[0] + ".cubin")
            if not os.path.exists(env["PD_CUBIN"]):
                print("%s SKIPPED: no cubin %s" % (parts[1], parts[0]), flush=True)
                continue
        args = parts[1:]
        secs = float(args[3]) if len(args) > 3 else 0.0
        rows, stop = [], threading.Event()
        th = None
        if secs >= 10:
            th = threading.Thread(target=sample, args=(stop, rows))
            th.start()
        t0 = time.time()
        diag = []
        try:
            r = subprocess.run([exe] + args, capture_output=True, text=True, timeout=secs + 300, env=env)
            # gprobe's PD_DIAG lines ("diag ...") follow its result line; they are printed after it.
            lines = (r.stdout.strip() or r.stderr.strip() or "rc %d" % r.returncode).splitlines()
            diag = [x for x in lines if x.startswith("diag ")]
            out = ([x for x in lines if not x.startswith("diag ")] or ["rc %d" % r.returncode])[-1]
            if r.returncode != 0:
                out += " [rc %d] %s" % (r.returncode, " | ".join(r.stderr.strip().splitlines()[:2]))
        except subprocess.TimeoutExpired:
            out = "%s TIMEOUT" % args[0]
        took = time.time() - t0
        if th:
            stop.set()
            th.join(timeout=5)
            keep = [v for t, v in rows if t >= 8.0 and v[3] >= 90][-20:]
            if keep:
                n = len(keep)
                mean = [sum(v[i] for v in keep) / n for i in range(8)]
                m = re.search(r"pipe\s+([\d.]+)%", out)
                count = " count %.1f" % (n_sm * 1024 * mean[0] * 1e6 * float(m[1]) / 100 / 1e12) if m and n_sm else ""
                out += " | %.0f MHz %.1f W %.0f C mem %.0f%% cap %d/%d hwslow %d swthermal %d%s" % (
                    mean[0], mean[1], mean[2], mean[4], sum(v[5] for v in keep), n, sum(v[6] for v in keep),
                    sum(v[7] for v in keep), count)
        print("%s [%.0f s]" % (out, took), flush=True)
        for x in diag[:40]:
            print(x, flush=True)


if __name__ == "__main__":
    main()
