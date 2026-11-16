"""nvprobe.py: what nv20probe.py and nv30probe.py share: the card's SM count, and its nvidia-smi readings sampled
once a second while a probe runs. Measuring tools for the Salad boxes, not the miner.

The probes run under python3 -I, which leaves their own folder off sys.path, so they load this file by path.
"""
import ctypes
import subprocess
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
