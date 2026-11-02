#!/usr/bin/env python3
"""Records the stratum lines a miner exchanges with the pool, and decodes the shares it sent. bench.sh runs it
when TAP=1: every miner then mines through the relay, so the run shows what each one puts in its shares.

  python3 sharetap.py tap PORT HOST POOL_PORT OUT.jsonl    relay 127.0.0.1:PORT to HOST:POOL_PORT, logging lines
  python3 sharetap.py decode OUT.jsonl [--worker W] [--since EPOCH] [--max 3]
                                                           three lines for each of the first shares, then a summary
                                                           per worker; --since keeps connections opened after EPOCH
  python3 sharetap.py jobs OUT.jsonl [--since EPOCH]       how often the pool sent a new job: the seconds between
                                                           one connection's mining.notify lines
  python3 sharetap.py shares OUT.jsonl [--since EPOCH]     every share as JSON lines for the reference verifier
                                                           (earn/native/probes/refverify.rs): the job's header and
                                                           target as the pool sent them, the proof unzipped

The relay changes nothing: bytes go through as they come, and each newline-ended line is also written to
OUT.jsonl with its connection and direction. The decoder reads a share's plain_proof, the public PlainProof
format (zk-pow/src/ffi/plain_proof.rs; ours is earn/src/shared/miner/plainProof.js): m, n, k, noise rank, and
for A and B^T the row indices (the share's tile) and the 1024-byte leaves (the operand bytes those rows hold).
It prints workers, never wallets, and keeps every line under 440 characters for Salad's log service.
"""
import asyncio, base64, collections, gzip, json, struct, sys, time


def tap(port, host, pool_port, path):
    out = open(path, "a", encoding="utf-8")
    seq = [0]

    def log(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    async def pipe(reader, writer, cid, d):
        buf = b""
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    log({"t": time.time(), "c": cid, "d": d, "line": line.decode("utf-8", "replace")})
        except Exception as e:
            log({"t": time.time(), "c": cid, "d": d, "err": repr(e)[:200]})
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def handle(cr, cw):
        seq[0] += 1
        cid = seq[0]
        log({"t": time.time(), "c": cid, "d": "open"})
        try:
            pr, pw = await asyncio.open_connection(host, pool_port)
        except Exception as e:
            log({"t": time.time(), "c": cid, "d": "up", "err": "connect: " + repr(e)[:200]})
            cw.close()
            return
        await asyncio.gather(pipe(cr, pw, cid, "up"), pipe(pr, cw, cid, "down"))
        log({"t": time.time(), "c": cid, "d": "close"})

    async def main():
        srv = await asyncio.start_server(handle, "127.0.0.1", port)
        print(f"tap listening on 127.0.0.1:{port} -> {host}:{pool_port}", flush=True)
        async with srv:
            await srv.serve_forever()

    asyncio.run(main())


class Reader:
    def __init__(self, b):
        self.b, self.o = b, 0

    def u64(self):
        v = struct.unpack_from("<Q", self.b, self.o)[0]
        self.o += 8
        return v

    def take(self, n):
        v = self.b[self.o:self.o + n]
        if len(v) != n:
            raise ValueError("proof ends early")
        self.o += n
        return v


def merkle(r):
    """One MatrixMerkleProof: leaves, leaf indices, total leaves, root, siblings, then the row indices."""
    leaves = [r.take(r.u64()) for _ in range(r.u64())]
    idx = [r.u64() for _ in range(r.u64())]
    total = r.u64()
    r.take(32)
    sib = r.u64()
    r.take(32 * sib)
    rows = [r.u64() for _ in range(r.u64())]
    return {"leaves": leaves, "idx": idx, "total": total, "sib": sib, "rows": rows}


def hist(leaves):
    """The leaves' bytes as signed values: how many distinct, the range, and the four commonest with their %."""
    c = collections.Counter()
    for leaf in leaves:
        c.update(leaf)
    n = sum(c.values()) or 1
    vals = sorted((v - 256 if v > 127 else v) for v in c)
    top = ",".join(f"{v - 256 if v > 127 else v}:{100 * k / n:.1f}%" for v, k in c.most_common(4))
    return f"distinct={len(c)} range={vals[0] if vals else '?'}..{vals[-1] if vals else '?'} top={top}"


def pattern(rows):
    """Row indices as base+offsets, sorted, so the tile's shape shows whatever the base."""
    s = sorted(rows)
    if not s:
        return "none"
    offs = ",".join(str(x - s[0]) for x in s)
    return f"{len(s)}@{s[0]}+[{offs[:150]}{'…' if len(offs) > 150 else ''}]"


def proof_field(params):
    """(param shape, the plain_proof string): by key if params is an object, else the longest string in the list."""
    if isinstance(params, dict):
        shape = "{" + ",".join(f"{k}:{len(v) if isinstance(v, str) else type(v).__name__}" for k, v in params.items()) + "}"
        return shape, params.get("plain_proof") or max((v for v in params.values() if isinstance(v, str)),
                                                        key=len, default="")
    if isinstance(params, list):
        shape = "[" + ",".join(str(len(v)) if isinstance(v, str) else type(v).__name__ for v in params) + "]"
        return shape, max((v for v in params if isinstance(v, str)), key=len, default="")
    return type(params).__name__, ""


def worker_of(params):
    """The worker name a login carries ("wallet.worker" or a worker field); never the wallet."""
    p = params
    if isinstance(p, dict):
        if p.get("worker"):
            return str(p["worker"])[:40]
        p = [p.get("login") or p.get("user") or p.get("wallet") or ""]
    if isinstance(p, list) and p:
        s = str(p[0])
        return s.rsplit(".", 1)[1][:40] if "." in s else "(none)"
    return "?"


def decode(path, want=None, most=3, since=0.0):
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    # Connections opened before `since` belong to an earlier miner; bench.sh passes each run's start time.
    opened = {e["c"]: e["t"] for e in recs if e.get("d") == "open"}
    recs = [e for e in recs if opened.get(e.get("c"), 0) >= since]
    worker, pending, replies = {}, {}, {}
    shares = []
    for e in recs:
        if "line" not in e:
            continue
        try:
            m = json.loads(e["line"])
        except ValueError:
            continue
        if e["d"] == "up" and m.get("method") in ("mining.authorize", "login", "mining.login"):
            worker[e["c"]] = worker_of(m.get("params"))
        elif e["d"] == "up" and m.get("method") == "mining.submit":
            shares.append((e, m))
        elif e["d"] == "down" and m.get("id") is not None and "method" not in m:
            replies[(e["c"], m["id"])] = "ok" if m.get("result") not in (None, False) and not m.get("error") \
                else "rej:" + json.dumps(m.get("error") or m.get("result"))[:80]
    out, summ = [], collections.defaultdict(lambda: collections.Counter())
    shown = collections.Counter()
    for e, m in shares:
        w = worker.get(e["c"], "?")
        if want and w != want:
            continue
        shape, field = proof_field(m.get("params"))
        verdict = replies.get((e["c"], m.get("id")), "noreply")
        try:
            raw = base64.b64decode(field)
            gz = raw[:2] == b"\x1f\x8b"
            if gz:
                raw = gzip.decompress(raw)
            r = Reader(raw)
            M, N, K, R = r.u64(), r.u64(), r.u64(), r.u64()
            a, bt = merkle(r), merkle(r)
            tail = len(raw) - r.o
        except Exception as ex:
            summ[w]["undecoded"] += 1
            if shown[w] < most:
                shown[w] += 1
                out.append(f"{w} {verdict} params={shape[:200]} undecoded: {repr(ex)[:120]}")
            continue
        key = f"m={M} n={N} k={K} r={R} tile={len(a['rows'])}x{len(bt['rows'])}"
        summ[w][key] += 1
        summ[w][verdict.split(":")[0]] += 1
        if shown[w] >= most:
            continue
        shown[w] += 1
        t = time.strftime("%H:%M:%S", time.gmtime(e["t"]))
        out.append(f"{w} {t} {verdict} {key} gz={int(gz)} wire={len(field)} plain={len(raw)} tail={tail} "
                   f"A:leaves={len(a['leaves'])}x{len(a['leaves'][0]) if a['leaves'] else 0} of {a['total']} sib={a['sib']} "
                   f"Bt:leaves={len(bt['leaves'])}x{len(bt['leaves'][0]) if bt['leaves'] else 0} of {bt['total']} "
                   f"sib={bt['sib']} params={shape[:120]}")
        out.append(f"{w} {t} A rows {pattern(a['rows'])} bytes {hist(a['leaves'])}"[:430])
        out.append(f"{w} {t} Bt rows {pattern(bt['rows'])} bytes {hist(bt['leaves'])}"[:430])
    for w, c in summ.items():
        out.append(f"{w} summary " + " ".join(f"{k}:{v}" for k, v in c.most_common())[:400])
    if not summ:
        out.append(f"{want or 'all'}: no shares")
    return out


def records(path, since=0.0):
    """The log's records with a parsed line, from connections opened at or after `since`, as (record, message)."""
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    opened = {e["c"]: e["t"] for e in recs if e.get("d") == "open"}
    out = []
    for e in recs:
        if "line" not in e or opened.get(e.get("c"), 0) < since:
            continue
        try:
            out.append((e, json.loads(e["line"])))
        except ValueError:
            continue
    return out


def jobs(path, since=0.0):
    """How often a new job came: mining.notify lines per connection and the gaps between them, in seconds. A job
    switch costs the miner a drain and a redraw, more at a larger k, so this is what a hashrate race has to match."""
    times = collections.defaultdict(list)
    for e, m in records(path, since):
        if e["d"] == "down" and m.get("method") == "mining.notify":
            times[e["c"]].append(e["t"])
    gaps = sorted(b - a for t in times.values() for a, b in zip(t, t[1:]))
    n = sum(len(t) for t in times.values())
    if not gaps:
        return f"jobs n={n} conns={len(times)}: too few for a gap"
    at = lambda q: gaps[min(len(gaps) - 1, int(q * len(gaps)))]
    return (f"jobs n={n} conns={len(times)} gaps={len(gaps)} gap_s median={at(0.5):.1f} p10={at(0.1):.1f} "
            f"min={gaps[0]:.1f} mean={sum(gaps) / len(gaps):.1f} max={gaps[-1]:.1f}")


def shares(path, since=0.0):
    """Every share as one JSON object, for the reference verifier: the header and target of the job it names, as
    the pool sent them on that connection, the proof as base64 bincode (unzipped), its k, the worker and the pool's
    verdict. A share whose job never came on its connection is left out and counted in the last line."""
    worker, notified, replies, subs = {}, {}, {}, []
    for e, m in records(path, since):
        p = m.get("params")
        if e["d"] == "up" and m.get("method") in ("mining.authorize", "login", "mining.login"):
            worker[e["c"]] = worker_of(p)
        elif e["d"] == "down" and m.get("method") == "mining.notify" and isinstance(p, dict):
            notified[(e["c"], str(p.get("job_id")))] = (p.get("header"), p.get("target"))
        elif e["d"] == "up" and m.get("method") == "mining.submit" and isinstance(p, dict):
            subs.append((e, m, notified.get((e["c"], str(p.get("job_id"))))))
        elif e["d"] == "down" and m.get("id") is not None and "method" not in m:
            replies[(e["c"], m["id"])] = "ok" if m.get("result") not in (None, False) and not m.get("error") else "rej"
    out, orphans = [], 0
    for e, m, job in subs:
        if not job:
            orphans += 1
            continue
        raw = base64.b64decode(m["params"].get("plain_proof") or "")
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        k = struct.unpack_from("<Q", raw, 16)[0] if len(raw) >= 24 else None
        out.append(json.dumps({"header": job[0], "target": job[1], "plain_proof": base64.b64encode(raw).decode(),
                               "k": k, "worker": worker.get(e["c"], "?"),
                               "verdict": replies.get((e["c"], m.get("id")), "noreply")}))
    out.append(json.dumps({"shares": len(out), "without_job": orphans}))
    return out


if __name__ == "__main__":
    opt = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    if len(sys.argv) >= 6 and sys.argv[1] == "tap":
        tap(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif len(sys.argv) >= 3 and sys.argv[1] == "decode":
        for line in decode(sys.argv[2], opt.get("--worker"), int(opt.get("--max", 3)),
                           float(opt.get("--since", 0))):
            print(line[:440], flush=True)
    elif len(sys.argv) >= 3 and sys.argv[1] == "jobs":
        print(jobs(sys.argv[2], float(opt.get("--since", 0))), flush=True)
    elif len(sys.argv) >= 3 and sys.argv[1] == "shares":
        for line in shares(sys.argv[2], float(opt.get("--since", 0))):
            print(line, flush=True)
    else:
        sys.exit(__doc__)
