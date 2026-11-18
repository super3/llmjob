"""profsum.py NAME BASE: one line a fact from a profiler's report, for bench.sh's profiled legs and nv20box.sh's
prof stage (goal-rtx20b). A measuring tool: it reads what NVIDIA's profilers record about the kernels a miner
launches (launch shape, registers, shared memory, times, and with Nsight Compute the hardware counters). It
never reads a miner's code.

  BASE.nsys-rep  Nsight Systems: per kernel name and launch shape, launches, mean and total time, and the GPU's
                 busy share of the traced window; copies by kind.
  BASE.ncu-rep   Nsight Compute: per profiled kernel, the metrics named in KEEP, in lines of under 380 characters.
"""
import csv
import io
import os
import re
import sqlite3
import subprocess
import sys

NSYS = "/opt/prof/nsys/bin/nsys"
NCU = "/opt/prof/ncu/ncu"
KEEP = re.compile(
    r"^(launch__|gpu__time_duration\.sum|sm__cycles_elapsed\.avg\.per_second|sm__warps_active\.avg\.pct_of_peak"
    r"|sm__maximum_warps|smsp__issue_active\.avg\.pct|smsp__inst_executed\.sum$|sm__inst_executed_pipe_"
    r"|sm__pipe_.*(pct_of_peak_sustained_active|pct_of_peak_sustained_elapsed)$"
    r"|smsp__average_warps?_issue_stalled_.*_per_issue_active\.ratio$|smsp__average_warp_latency_per_inst_issued"
    r"|.*throughput\.avg\.pct_of_peak_sustained_elapsed$|dram__bytes_(read|write)\.sum$|lts__t_bytes\.sum$"
    r"|l1tex__t_bytes\.sum$|l1tex__data_bank_conflicts_pipe_lsu_mem_shared.*\.sum$|l1tex__data_pipe_lsu_wavefronts_mem_shared.*\.sum$"
    r"|smsp__inst_executed_op_.*\.sum$|smsp__sass_inst_executed_op_.*\.sum$|sm__sass_thread_inst_executed_op_.*sum$"
    r"|smsp__thread_inst_executed_per_inst_executed\.ratio$|sm__instruction_throughput|smsp__warps_eligible\.avg\.per_cycle_active"
    r"|smsp__warps_active\.avg\.per_cycle_active|smsp__issue_inst0|l1tex__t_sector_hit_rate|lts__t_sector_hit_rate)")


def emit(head, parts, width=370):
    line = head
    for p in parts:
        if len(line) + len(p) + 1 > width:
            print(line)
            line = head + "+"
        line += " " + p
    if line.strip("+") != head:
        print(line)


def nsys(base):
    rep = base + ".nsys-rep"
    db = base + ".sqlite"
    if not os.path.exists(rep):
        print("nsys no report")
        return
    r = subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true", "--output", db, rep],
                       capture_output=True, text=True)
    if not os.path.exists(db):
        print("nsys export failed: " + (r.stderr or r.stdout)[-300:].replace("\n", " "))
        return
    c = sqlite3.connect(db)
    tables = {t for (t,) in c.execute("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        print("nsys no kernels traced; tables: " + " ".join(sorted(tables))[:300])
        return
    t0, t1 = c.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    busy = c.execute("select sum(end - start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    print(f"nsys window {(t1 - t0) / 1e9:.2f}s kernels busy {100 * busy / max(1, t1 - t0):.2f}%")
    q = """select s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k.registersPerThread,
                  k.staticSharedMemory, k.dynamicSharedMemory, k.localMemoryPerThread, count(*),
                  avg(k.end - k.start), sum(k.end - k.start), min(k.end - k.start), max(k.end - k.start)
           from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName
           group by 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 order by 14 desc limit 12"""
    for row in c.execute(q):
        (name, gx, gy, gz, bx, by, bz, regs, ss, ds, lm, n, avg, tot, mn, mx) = row
        print(f"kern {name[:60]} grid={gx}x{gy}x{gz} block={bx}x{by}x{bz} regs={regs} smem={ss}+{ds} local={lm} "
              f"n={n} mean_ms={avg / 1e6:.3f} min_ms={mn / 1e6:.3f} max_ms={mx / 1e6:.3f} share={100 * tot / busy:.1f}%")
    # Gaps between back-to-back kernels on the GPU: idle time the launch pattern leaves.
    st = [r for r in c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    gaps = sorted(b[0] - a[1] for a, b in zip(st, st[1:]) if b[0] > a[1])
    if gaps:
        print(f"gaps n={len(gaps)} median_us={gaps[len(gaps) // 2] / 1e3:.1f} total_ms={sum(gaps) / 1e6:.1f}")
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for kind, n, b, d in c.execute("select copyKind, count(*), sum(bytes), sum(end - start) from "
                                       "CUPTI_ACTIVITY_KIND_MEMCPY group by 1"):
            print(f"memcpy kind={kind} n={n} bytes={b} ms={d / 1e6:.2f}")
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        q = ("select s.value, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId "
             "group by 1 order by 2 desc limit 14")
        emit("api", [f"{n.split('_v')[0]}={k}" for n, k in c.execute(q)])


def ncu(base):
    rep = base + ".ncu-rep"
    if not os.path.exists(rep):
        print("ncu no report")
        return
    r = subprocess.run([NCU, "--import", rep, "--page", "raw", "--csv"], capture_output=True, text=True)
    rows = list(csv.reader(io.StringIO(r.stdout)))
    if len(rows) < 3:
        print("ncu import: " + (r.stderr or r.stdout)[-300:].replace("\n", " "))
        return
    head, units = rows[0], rows[1]
    for row in rows[2:]:
        d = dict(zip(head, row))
        name = d.get("Kernel Name", "?")[:70]
        parts = []
        for h, u, v in zip(head, units, row):
            if not KEEP.match(h) or v in ("", "n/a"):
                continue
            if "stalled" in h:
                try:
                    if float(v.replace(",", "")) < 0.05:
                        continue
                except ValueError:
                    pass
                h = h.replace("smsp__average_warp_latency_issue_stalled_", "stall_") \
                     .replace("smsp__average_warps_issue_stalled_", "stall_").replace("_per_issue_active.ratio", "")
            parts.append(f"{h}={v}{u if u and u not in ('', 'n/a') and len(u) < 8 else ''}")
        emit(f"ncu {d.get('ID', '?')} {name}", parts)


if __name__ == "__main__":
    base = sys.argv[2]
    if os.path.exists(base + ".nsys-rep"):
        nsys(base)
    elif os.path.exists(base + ".ncu-rep"):
        ncu(base)
    else:
        print("no report at " + base + ".*: " + " ".join(sorted(os.listdir(os.path.dirname(base))))[:300])
