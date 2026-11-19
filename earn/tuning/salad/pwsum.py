#!/usr/bin/env python3
"""pwsum.py (goal-rtx50c): one line summing a leg's fast power trace, and how its spikes line up with salt events.

  python3 pwsum.py TRACE EVENTS LIMIT_W

TRACE is nvidia-smi --query-gpu=timestamp,power.draw.instant,clocks.sm --format=csv,noheader,nounits -lms 100
(timestamps "YYYY/MM/DD HH:MM:SS.mmm"); EVENTS lines "<epoch seconds> <text>" (the core's stderr, timestamped as it
came), where "prep salt" and "switch salt" mark the side stream's next-salt work and a prepared salt taking over.
The first 20 s of the trace are left out (warm-up). Prints the samples' count, mean, SD, min, max, the share at or
above LIMIT - 10 W and above 590 W, the 98th percentile, and of the samples in the top 2%, how many lie within 300 ms
of a salt event, against the share of all samples that do (what chance alone would give).
"""
import datetime
import statistics
import sys


def epoch(ts):
    d = datetime.datetime.strptime(ts.strip(), "%Y/%m/%d %H:%M:%S.%f")
    return d.replace(tzinfo=datetime.timezone.utc).timestamp()


def main():
    trace, events, limit = sys.argv[1], sys.argv[2], float(sys.argv[3])
    rows = []
    for line in open(trace, encoding="utf-8", errors="replace"):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 3:
            continue
        try:
            rows.append((epoch(f[0]), float(f[1]), float(f[2])))
        except ValueError:
            continue
    if len(rows) < 50:
        print("pw too few samples: %d" % len(rows))
        return
    t0 = rows[0][0]
    rows = [r for r in rows if r[0] - t0 >= 20]
    ev = []
    for line in open(events, encoding="utf-8", errors="replace"):
        p = line.split(" ", 1)
        if len(p) == 2 and ("prep salt" in p[1] or "switch salt" in p[1]):
            try:
                ev.append(float(p[0]))
            except ValueError:
                pass
    w = [r[1] for r in rows]
    mhz = [r[2] for r in rows]
    srt = sorted(w)
    p98 = srt[int(0.98 * (len(srt) - 1))]

    def near(t):
        return any(abs(t - e) <= 0.3 for e in ev)

    top = [r for r in rows if r[1] >= p98]
    near_top = sum(1 for r in top if near(r[0]))
    near_all = sum(1 for r in rows if near(r[0]))
    span = rows[-1][0] - rows[0][0]
    # The second after each event against the rest: what the events cost in clock, and how much of the time they hold.
    after = [r for r in rows if any(0 <= r[0] - e <= 1.0 for e in ev)]
    rest = [r for r in rows if not any(0 <= r[0] - e <= 1.0 for e in ev)]
    # Which kind of event the top 2% sit next to: the next salt's tables queued, or a prepared salt taking over.
    kinds = {}
    for line in open(events, encoding="utf-8", errors="replace"):
        p = line.split(" ", 1)
        for kind in ("prep salt", "switch salt"):
            if len(p) == 2 and kind in p[1]:
                try:
                    kinds.setdefault(kind.split()[0], []).append(float(p[0]))
                except ValueError:
                    pass
    by = " ".join("%s=%d" % (k, sum(1 for r in top if any(abs(r[0] - e) <= 0.3 for e in v))) for k, v in sorted(kinds.items()))
    if after and rest:
        print("pwafter top2%%by %s share=%.1f%% w=%.1f/%.1f mhz=%.0f/%.0f max=%.0f/%.0f"
              % (by, 100.0 * len(after) / len(rows), statistics.mean(r[1] for r in after), statistics.mean(r[1] for r in rest),
                 statistics.mean(r[2] for r in after), statistics.mean(r[2] for r in rest),
                 max(r[1] for r in after), max(r[1] for r in rest)))
    print("pw n=%d mean=%.1f sd=%.1f min=%.0f max=%.0f p98=%.0f atcap=%.1f%% over590=%.1f%% mhz=%.0f mhzsd=%.0f "
          "events=%d (%.2f/s) top2%%near=%d/%d all_near=%.1f%%"
          % (len(w), statistics.mean(w), statistics.pstdev(w), srt[0], srt[-1], p98,
             100.0 * sum(1 for x in w if x >= limit - 10) / len(w), 100.0 * sum(1 for x in w if x > 590) / len(w),
             statistics.mean(mhz), statistics.pstdev(mhz), len(ev), len(ev) / span if span > 0 else 0,
             near_top, len(top), 100.0 * near_all / len(rows)))


if __name__ == "__main__":
    main()
