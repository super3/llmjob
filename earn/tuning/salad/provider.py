#!/usr/bin/env python3
# One client for the two places a benchmark can run a container: Salad, or rigs in our own network (issue #318).
# BENCH_PROVIDER picks one: "salad" (the default, what bench.py and amdtest.py use today) or "network". Both have
# the same four calls, mirroring Salad's container-group calls: create, status, logs, delete.
#
#   p = provider.pick()
#   h = p.create({"name": "bench-40-rtx4090-1", "image": IMAGE, "env": {...}, "duration_sec": 3600,
#                 "replicas": 1, "target": {"gpu": "RTX 4090"}})          # or {"nodeId": ..., "card": 0}
#   p.status(h)   -> {"state": "pending"|"running"|"stopped"|"done", "instances": [{"id", "where", "state"}]}
#   p.logs(h, since) -> (lines, next_since), lines as [{"t": iso or ms, "where": ..., "text": ...}]
#   p.delete(h)
#
# Network: our server's /api/containers, with the admin token LLMJOB_CONTAINERS_TOKEN (the server's
# CONTAINERS_ADMIN_TOKEN) as the bearer, at LLMJOB_SERVER (production by default). The image must be ours on GHCR,
# pinned by digest, or the server refuses it. Salad: the watcher's client and SALAD_API_KEY, as bench.py has always
# used. A prototype: bench.py is not switched over to it yet.
import importlib.util, json, os, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.environ.get("LLMJOB_SERVER", "https://llmjob-production.up.railway.app")


def http(method, url, body=None, headers=None, timeout=60):
    """(status, json) for one call; status 0 when the call never got an answer. Never raises."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "llmjob-bench/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else None)
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(txt)
        except ValueError:
            return e.code, {"raw": txt[:500]}
    except Exception as e:   # a refused connection, a timeout, a bad URL
        return 0, {"error": str(e)[:300]}


class ProviderError(Exception):
    pass


def _ok(s, j, what):
    if s not in (200, 201):
        raise ProviderError(f"{what}: HTTP {s} {json.dumps(j)[:300]}")
    return j


class Network:
    """Rigs in our own network, through the server's admin calls."""
    name = "network"

    def __init__(self, server=None, token=None, call=http):
        self.server = (server or SERVER).rstrip("/")
        self.token = token if token is not None else os.environ.get("LLMJOB_CONTAINERS_TOKEN", "")
        self.call = call

    def _req(self, method, path, body=None):
        return self.call(method, self.server + "/api/containers" + path, body,
                         headers={"Authorization": "Bearer " + self.token}, timeout=60)

    def create(self, spec):
        body = {"name": spec["name"], "image": spec["image"], "env": spec.get("env") or {},
                "durationSec": int(spec.get("duration_sec") or 3600), "replicas": int(spec.get("replicas") or 1),
                "target": spec.get("target") or {}, "payRateUsdHour": spec.get("pay_rate_usd_hour") or 0}
        j = _ok(*self._req("POST", "", body), "create")
        return j["job"]["id"]

    def status(self, handle):
        job = _ok(*self._req("GET", "/" + handle), "status")["job"]
        state = {"pending": "pending", "running": "running", "done": "done"}.get(job["status"], "stopped")
        inst = [{"id": r["id"], "where": f"{r['nodeId']}/gpu{r['card']}", "state": r["status"],
                 "minutes": r.get("minutes", 0)} for r in job.get("runs") or []]
        return {"state": state, "instances": inst, "raw": job}

    def logs(self, handle, since=0):
        j = _ok(*self._req("GET", f"/{handle}/logs?since={int(since or 0)}"), "logs")
        lines = [{"t": x.get("ts"), "where": f"{x['nodeId']}/gpu{x['card']}", "text": x.get("line") or ""}
                 for x in j.get("lines") or []]
        return lines, j.get("next", since)

    def delete(self, handle):
        s, j = self._req("DELETE", "/" + handle)
        return s in (200, 404)


class Salad:
    """Salad container groups, through the watcher's client (its org, project and SALAD_API_KEY)."""
    name = "salad"

    def __init__(self, w=None):
        self._w = w

    @property
    def w(self):
        if self._w is None:
            spec = importlib.util.spec_from_file_location("watcher", os.path.join(HERE, "watcher.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._w = mod
        return self._w

    def create(self, spec):
        # Salad wants GPU class ids; the caller looks them up (bench.py does, from /gpu-classes).
        body = {"name": spec["name"], "replicas": int(spec.get("replicas") or 1), "restart_policy": "never",
                "autostart_policy": True,
                "container": {"image": spec["image"], "environment_variables": spec.get("env") or {},
                              "resources": {"cpu": int(spec.get("cpu") or 4), "memory": int(spec.get("memory_mb") or 8192),
                                            "gpu_classes": list(spec.get("gpu_classes") or [])}}}
        _ok(*self.w.http("POST", self.w.CB, body), "create")
        return spec["name"]

    def status(self, handle):
        g = _ok(*self.w.http("GET", f"{self.w.CB}/{handle}"), "status")
        st = ((g or {}).get("current_state") or {}).get("status") or ""
        state = {"running": "running", "deploying": "pending", "pending": "pending"}.get(st, "stopped")
        s, j = self.w.http("GET", f"{self.w.CB}/{handle}/instances")
        inst = [{"id": i.get("instance_id"), "where": i.get("machine_id"), "state": i.get("state")}
                for i in ((j or {}).get("instances") or [] if s == 200 else [])]
        return {"state": state, "instances": inst, "raw": g}

    def logs(self, handle, since=0):
        """Lines since `since` (epoch seconds). Salad's log API runs late and can drop lines."""
        t = time.time()
        body = {"start_time": self.w.iso(since or t - 3600), "end_time": self.w.iso(t), "page_size": 100,
                "sort_order": "asc",
                "query": f'resource.type = "container" AND resource.labels.container_group_name = "{handle}"'}
        j = _ok(*self.w.http("POST", f"{self.w.BASE}/log-entries", body, timeout=90), "logs")
        items = sorted((j or {}).get("items") or [], key=lambda it: it.get("time") or "")
        lines = [{"t": it.get("time"), "where": ((it.get("resource") or {}).get("labels") or {}).get("machine_id"),
                  "text": it.get("text_log") or ""} for it in items]
        nxt = self.w.parse_iso(items[-1]["time"]) if items else since
        return lines, nxt

    def delete(self, handle):
        s, _ = self.w.http("DELETE", f"{self.w.CB}/{handle}")
        return s in (200, 202, 204, 404)


def pick(name=None, **kw):
    """The provider BENCH_PROVIDER names: "salad" (default) or "network"."""
    name = (name or os.environ.get("BENCH_PROVIDER") or "salad").lower()
    if name == "network":
        return Network(**kw)
    if name == "salad":
        return Salad(**kw)
    raise ProviderError(f"BENCH_PROVIDER must be salad or network, not {name!r}")
