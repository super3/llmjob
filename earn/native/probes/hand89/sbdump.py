"""sbdump.py [W] [key=value ...]: the gathers of rfold.py's loop whose IADD3 waits on a scoreboard that another
gather's load set less than W gaps before (default 6): a wait for a load that is not the waiter's own.

A scoreboard counts the loads that set it and are still out, and an instruction that waits on it waits for all of
them. The loop has three scoreboards for its gathers and up to six gathers under way, so two gathers share one,
and the later one's LDSM can go out just ahead of the earlier one's IADD3. rfold.py's sbo picks the scoreboards
so that this happens as little as possible; this lists what is left, for any keys:

  python sbdump.py              the default loop
  python sbdump.py 6 sbo=0      the same with the scoreboards handed out in turn
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
args = sys.argv[1:]
W = int(args.pop(0)) if args and "=" not in args[0] else 6
# rfold.chunk keeps its plans to itself: run a copy of it that hands them out.
src = open(os.path.join(HERE, "rfold.py"), encoding="utf-8").read()
hook = "    use, assign = {}, {}\n"
assert src.count(hook) == 1
src = src.replace(hook, "    DEBUG.append((order, plans, dict(sbs), tot))\n" + hook)
ns = {"__name__": "rfold_copy", "__file__": os.path.join(HERE, "rfold.py")}
exec(compile("DEBUG = []\n" + src, "rfold_copy", "exec"), ns)
cfg = dict(ns["CFG"])
for kv in args:
    k, v = kv.split("=")
    assert k in cfg, k
    cfg[k] = int(v, 0)
ns["chunk"](cfg, ns["layout"](cfg["mt"]))
order, plans, sbs, tot = ns["DEBUG"][0]
ni = tot // 4


def kind(k):
    return "A" if plans[k][5] < ns["BF"] else "B"


n = 0
for j in order:
    for i in order:
        if i == j or sbs[i] != sbs[j]:
            continue
        d = (plans[j][1] - plans[i][0]) % tot
        if 1 <= d < min(W, (plans[j][1] - plans[j][0]) % tot):
            n += 1
            print("turn %d: the %s gather with its LDSM in gap %d and its IADD3 in gap %d waits for the %s load of gap %d, "
                  "%d gaps before it (scoreboard %d)" % (plans[j][1] // ni % 4, kind(j), plans[j][0] % ni, plans[j][1] % ni,
                                                         kind(i), plans[i][0] % ni, d, sbs[j]))
print("%d waits a chunk for another gather's load, within %d gaps" % (n, W))
