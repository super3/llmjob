# Small SaladCloud public-API helper for PRL mining. Settings come from config.json next to this file (copy
# config.example.json); the API key only from SALAD_API_KEY in the environment. Never prints the key.
import json, os, urllib.request, urllib.error
D = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(D, "config.json")))
BASE = f"https://api.salad.com/api/public/organizations/{CFG['org']}"
PROJ = f"{BASE}/projects/{CFG['project']}"
POOL = f"https://pearl.herominers.com/api/stats_address?address={CFG['wallet']}&longpoll=false"

def _key():
    k = os.environ.get("SALAD_API_KEY")
    if not k: raise RuntimeError("SALAD_API_KEY is not set")
    return k

def call(method, url, body=None, timeout=60, ctype="application/json"):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Salad-Api-Key": _key(), "Content-Type": ctype, "User-Agent": "llmjob-fleet/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(); return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:400]!r}")

def group(name): return call("GET", f"{PROJ}/containers/{name}")
def stop(name): return call("POST", f"{PROJ}/containers/{name}/stop")
def start(name): return call("POST", f"{PROJ}/containers/{name}/start")
def delete(name): return call("DELETE", f"{PROJ}/containers/{name}")
def instances(name): return call("GET", f"{PROJ}/containers/{name}/instances")
def set_priority(name, priority): return call("PATCH", f"{PROJ}/containers/{name}", {"container": {"priority": priority}}, ctype="application/merge-patch+json")
def reallocate(name, instance_id): return call("POST", f"{PROJ}/containers/{name}/instances/{instance_id}/reallocate")
