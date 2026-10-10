# The hourly Salad search: at 90 s past each hour, list the classes search.py finds (cheapest priority with a GPU free
# that clears 10%) and create a one-replica group for up to 3 classes not tried before, best profit first. A group that
# gets no PC moves up the priorities that still clear the margin (watch_all.py does that). Stays inside the replica
# quota. watch_all.py starts, watches, and stops/deletes them.  NOW=1: one round.
import glob, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import salad, search
D = os.path.dirname(os.path.abspath(__file__)); GD = os.path.join(D, "groups"); os.makedirs(GD, exist_ok=True)
CFG = salad.CFG; MAX_NEW = 3
REF = CFG["box_ref"]                  # the commit of tuning/box.sh each box runs: the same script as the Vast and Clore boxes
# The 12.2 base image: Salad left groups on the 12.8.1 image in its "preparing" step for 45+ minutes on Oct 9, while
# groups on this one started at once. The PCs run driver 610 and the miner carries its own CUDA runtime, so the
# image's CUDA version doesn't matter to it.
IMAGE = "nvidia/cuda:12.2.2-base-ubuntu22.04"

def log(msg): print(time.strftime("%H:%M:%S", time.gmtime()) + " " + msg, flush=True)

def command(worker):
    # box.sh carries placeholders for the worker, and its own wallet and miner release: set all three.
    # Salad refuses a command with "; curl" in it (HTTP 400, no detail), so curl follows "&&".
    sed = (f"sed -e 's/^WORKER=__WORKER__$/WORKER={worker}/' -e 's/^W=prl1.*$/W={CFG['wallet']}/'"
           f" -e 's/^VER=v[0-9.]*;/VER={CFG['miner_version']};/'")
    return ("apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq curl ca-certificates >/dev/null 2>&1 && "
            f"curl -fsSL https://raw.githubusercontent.com/super3/llmjob/{REF}/tuning/box.sh | {sed} > /root/box.sh; "
            "bash /root/box.sh 2>&1 | tee -a /root/box.log")

def create(o, name=None):
    slug = re.sub(r"[^a-z0-9]", "", o["card"].lower())
    name, worker = name or f"{CFG['prefix']}-{slug}-{o['priority']}", f"{CFG['prefix']}-{slug}"
    body = {"name": name, "display_name": name, "autostart_policy": True, "restart_policy": "always", "replicas": 1,
            "container": {"image": IMAGE, "command": ["/bin/sh", "-c", command(worker)],
                          "priority": o["priority"], "resources": dict(search.RES, gpu_classes=[o["id"]])}}
    salad.call("POST", salad.PROJ + "/containers", body)
    g = salad.group(name)
    st = dict(name=name, worker=worker, gpu=o["id"], gpu_name=o["name"], card=o["card"], priority=o["priority"], price_hr=o["cost_hr"],
              ladder=o["ladder"], prio_since=time.time(), est_th=o["th"], created=time.time(), spent=0.0, events=[], group_id=g.get("id"),
              image=IMAGE)
    json.dump(st, open(os.path.join(GD, name + ".json"), "w"), indent=1)
    return st

def round_once():
    try:
        price, U, cands = search.search()
        tried = {json.load(open(f))["gpu"] for f in glob.glob(os.path.join(GD, "*.json"))}
        new = [o for o in cands if o["id"] not in tried]
        q = salad.call("GET", salad.BASE + "/quotas")["container_groups_quotas"]
        # Our stopped groups don't count against Salad's quota (Salad stops them all when the credit runs out), but
        # they are classes that ran and get their slots back first when they restart, so leave room for them.
        held = sum(1 for f in glob.glob(os.path.join(GD, "*.json")) for g in [json.load(open(f))]
                   if not g.get("outcome") and g.get("status") == "stopped" and g.get("ever_ran"))
        room = q["container_replicas_quota"] - q["container_replicas_used"] - held
        made = []
        for o in new[:min(MAX_NEW, max(0, room))]:
            try: made.append(create(o))
            except Exception as ex: log(f"{o['name']}: create failed: {str(ex)[:150]}")
        log(f"search: PRL ${price:.3f}; {len(cands)} classes clear the margin, {len(new)} not tried, quota room {room}; created {len(made)}"
            + (": " + "; ".join(f"{s['gpu_name']} {s['priority']} ${s['price_hr']:.3f}/hr" for s in made) if made else ""))
    except Exception as ex:
        log(f"search failed: {str(ex)[:200]}")

# Only when run as a script, so create() can be imported for a one-off group without starting the loop
if __name__ == "__main__":
    if os.environ.get("NOW"): round_once(); sys.exit(0)
    while True:
        now = time.time(); time.sleep(3600 - now % 3600 + 90)
        round_once()
