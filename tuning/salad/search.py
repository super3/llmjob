# Salad GPU classes worth mining: NVIDIA RTX classes we have a rate for (laptops, AMD and GTX cards left out), at the
# cheapest priority that has a GPU free right now and still clears a 10% margin at the live PRL price. Salad's price
# per class and priority already includes CPU and RAM, and there is no fee on top. Every priority can be taken back by
# the PC's owner, so it gets the spot margin. Prints them best first and writes candidates.json. Reads only.
import json, os, re, sys, urllib.request
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import salad
# TH/s per card on release v0.5.13. Where we have a Salad measurement (Oct 9-10, the median of a group's readings),
# that is used: home PCs often cap the card's power, so most run under the benchmark, and the same class still varies
# about 15% from PC to PC. The 3080 Ti, 4070 Ti Super and one 3090 PC earned less than they cost at these rates.
# The rest are the benchmark.md figures, or our own Vast boxes for the 4090 and 5090.
RATE = {
    # measured on Salad
    "RTX 2060": 40, "RTX 2080": 72, "RTX 3060 Ti": 45, "RTX 3070": 59, "RTX 3080": 107, "RTX 3080 Ti": 70,
    "RTX 3090": 116, "RTX 3090 Ti": 152, "RTX 4070": 120, "RTX 4070 Ti": 152, "RTX 4070S Ti": 137, "RTX 4080": 199,
    "RTX 5070": 128, "RTX 5080": 223,
    # not measured on Salad
    "RTX 2060S": 44, "RTX 2070": 48, "RTX 2070S": 58, "RTX 2080 Ti": 79, "RTX 3060": 49, "RTX 3070 Ti": 86,
    "RTX 4060": 55, "RTX 4060 Ti": 89, "RTX 4070S": 142, "RTX 4080S": 197, "RTX 4090": 315, "RTX 4090D": 274,
    "RTX 5060": 76, "RTX 5060 Ti": 94, "RTX 5070 Ti": 174, "RTX 5090": 418,
}
NAME = {"RTX 4070 Ti Super": "RTX 4070S Ti", "RTX 4070 Super": "RTX 4070S", "RTX 4080 Super": "RTX 4080S", "RTX 2060 Super": "RTX 2060S",
        "RTX 2070 Super": "RTX 2070S"}
MARGIN = 0.10
RES = {"cpu": 2, "memory": 4096, "storage_amount": 10 * 2 ** 30}      # what each container asks for

def get(u): return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"Accept": "application/json"}), timeout=20))
def econ():
    # The PRL price, and what one TH/s earns in an hour at today's difficulty and block reward
    price = float(get("https://api.prlscan.com/v1/market/prl")["price_usd"])
    b = get("https://api.prlscan.com/v1/blocks?limit=1")["items"][0]
    return price, 1e12 * 3600 / (float(b["difficulty"]) * 2 ** 48) * float(b["reward_grains"]) / 1e8 * price

def card(name):
    if "Laptop" in name or not name.startswith("RTX "): return None
    c = re.sub(r"\s*\(\d+\s*GB\)$", "", name)
    c = NAME.get(c, c)
    return c if c in RATE else None

def search():
    price, U = econ()
    out = []
    for gc in salad.call("GET", salad.BASE + "/gpu-classes").get("items") or []:
        c = card(gc["name"])
        if not c: continue
        earn = RATE[c] * U
        prices = {p["priority"]: float(p["price"]) for p in gc.get("prices") or []}
        ok = [(p, prices[p]) for p in ("batch", "low", "medium", "high") if p in prices and (earn - prices[p]) / earn >= MARGIN]
        if not ok: continue
        av = salad.call("POST", salad.BASE + "/availability/sce-gpu-availability", dict(RES, gpu_classes=[gc["id"]]))
        free = [(p, cost) for p, cost in ok if av.get(f"available_gpu_{p}")]
        if not free: continue
        p, cost = free[0]                      # the cheapest priority with a GPU free
        # Salad's free counts for low and medium aren't reliable for busy cards (Oct 9: a 4090 sat 20+ minutes at
        # low and at medium with ~100 reported free, and got a PC at high in a minute), so the watcher climbs this
        # ladder when a priority gets no PC.
        ladder = [[q, c2] for q, c2 in ok if c2 >= cost]
        out.append(dict(id=gc["id"], name=gc["name"], card=c, priority=p, cost_hr=cost, ladder=ladder, th=RATE[c], earn_hr=round(earn, 4),
                        profit_hr=round(earn - cost, 4), margin=round((earn - cost) / earn * 100, 1), free=av.get(f"available_gpu_{p}")))
    out.sort(key=lambda o: -o["profit_hr"])
    json.dump(dict(prl=price, U=U, offers=out), open(os.path.join(D, "candidates.json"), "w"), indent=1)
    return price, U, out

if __name__ == "__main__":
    price, U, out = search()
    print(f"PRL ${price:.3f}, ${U:.6f}/TH-hr; {len(out)} Salad classes clear a {MARGIN:.0%} margin with a GPU free")
    for o in out:
        print(f"  {o['name']:<26} {o['priority']:<6} ${o['cost_hr']:.3f}/hr  ~{o['th']} TH/s earns ${o['earn_hr']:.3f} -> +${o['profit_hr']:.3f}/hr ({o['margin']:+.0f}%), {o['free']} free")
