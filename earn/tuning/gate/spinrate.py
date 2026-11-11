"""spinrate.py <th_per_s> <SPIN line...> : what share of a whole draw's rate the spin blocks kept up.

The spin kernel's SPIN line gives launches, sms, units_lane_sum (units a lane, summed over the blocks that have
ended) and secs since the first launch. With one launch running and one queued, launches - 2 have surely ended and
launches - 1 probably. A whole stage's draw is 56 units a lane; the fold does TH/s x 1e12 / (sms x 3,145,728) stages a
second on each SM (192 x 256 x 64 MACs a stage).
"""
import re
import sys

th = float(sys.argv[1])
line = " ".join(sys.argv[2:])
d = dict(re.findall(r"(\w+)=([\d.]+)", line))
n, sms, u, secs = int(float(d["launches"])), int(float(d["sms"])), float(d["units_lane_sum"]), float(d["secs"])
periods, units = int(float(d["periods"])), int(float(d["units"]))
stages = th * 1e12 / (sms * 3145728.0)
need = 56 * stages
for ended in (n - 2, n - 1):
    if ended <= 0:
        continue
    per_block_launch = u / (sms * ended)          # units a lane a launch
    rate = u / sms / (secs * ended / n)             # units a lane a second, over the time the ended launches took
    print(f"ended={ended}: {per_block_launch:,.0f} units a lane a launch (asked {periods * units:,}), "
          f"{rate / 1e6:.1f} M units a lane a second; a whole draw needs {need / 1e6:.1f} M: {rate / need * 100:.0f}%")
