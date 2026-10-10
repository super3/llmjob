# Watches every Salad test group in groups/*.json: reads each group and its instance, and the pool, once a minute. Keeps
# a box mining as long as it pays, and stops then deletes the group (freeing its place in the replica quota) only
# when it isn't doing its job:
#   an instance has been running 20 minutes and the pool has had no share from it for 10 (it never started mining,
#   or it stopped);
#   after 20 minutes on a PC, earning less than it costs on two 5-minute checks in a row, judged on the miner's own
#   TH/s from its log over the last 30 minutes on that PC (else its pool share count; one pool reading swings a lot).
#   But if its card class has earned more than it costs on some PC before (this group or an earlier one), the PC is
#   the problem, not the class: Salad PCs differ a lot in how hard their owners cap the card's power (Oct 9: a 3090
#   ran 121 TH/s at +17% on one PC and 92 TH/s, under cost, on another capped at 262 W). Then the instance moves to
#   another PC instead, up to 3 slow PCs in a row; the 4th ends the group.
#   a group that had a PC, lost it, and got no other in 20 minutes.
# Before stopping a group it logs the container's last output lines (Salad keeps them), so a failure has a reason.
# Waiting for a PC is not billed. But for busy cards Salad's low and medium priorities may never get one (Oct 9: a 4090
# at low and at medium, ~100 reported free, no PC in 20+ minutes; at high, a PC in a minute). So a group that has never
# had a PC moves up to the next priority in its ladder (only priorities that clear the margin) after 10 minutes, and
# is deleted when the ladder runs out. Salad can take a PC back at any priority; it then finds another, and that gap
# isn't billed and isn't held against the group. Salad's API has no spend figure: cost is the class price while an instance runs.
# Log lines: events and a summary every 15 minutes go to the user's monitor; detail lines start with two spaces.
import calendar, glob, json, os, re, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import salad
D = os.path.dirname(os.path.abspath(__file__)); GD = os.path.join(D, "groups"); os.makedirs(GD, exist_ok=True)
POOL = salad.POOL
UNIT = 419 / 93644    # TH/s per pool unit/s (Czechia 5090: 24-hour pool average vs its miner's 419 TH/s, Oct 9)

def get(u): return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"Accept": "application/json"}), timeout=20))
def log(msg, s=None):
    line = time.strftime("%H:%M:%S", time.gmtime()) + " " + msg
    print(line, flush=True)
    if s is not None: s["events"].append(line)
def path(s): return os.path.join(GD, s["name"] + ".json")
def save(s): json.dump(s, open(path(s) + ".tmp", "w"), indent=1); os.replace(path(s) + ".tmp", path(s))
def econ():
    price = float(get("https://api.prlscan.com/v1/market/prl")["price_usd"])
    b = get("https://api.prlscan.com/v1/blocks?limit=1")["items"][0]
    return price, 1e12 * 3600 / (float(b["difficulty"]) * 2 ** 48) * float(b["reward_grains"]) / 1e8 * price
def container_log(s, minutes=40):
    f = "%Y-%m-%dT%H:%M:%SZ"
    q = f'resource.type = "container" AND resource.labels.project_name = "{salad.CFG["project"]}" AND resource.labels.container_group_name = "{s["name"]}"'
    try:
        r = salad.call("POST", salad.BASE + "/log-entries", {"start_time": time.strftime(f, time.gmtime(time.time() - minutes * 60)),
                       "end_time": time.strftime(f, time.gmtime()), "query": q, "page_size": 12, "sort_order": "desc"})
        return [(e.get("text_log") or json.dumps(e.get("json_log")))[:200] for e in r.get("items") or []][::-1] or ["(no container output)"]
    except Exception as ex: return [f"log query failed: {str(ex)[:120]}"]
def miner_lines(minutes=3):
    # The miners' own rates from their log lines ("[mining g0] 08:46:32 223.2 TH/s · 8 accepted ..."), which Salad keeps.
    # Steadier than the pool's share count, which is far too noisy in a box's first minutes.
    # One query a round for every group: Salad's log API takes about 10 s a call and times out under load, so a query
    # per group made a round take 11 minutes (Oct 10). Returns {group: [(time, machine, gpu, TH/s)]}, newest first.
    f = "%Y-%m-%dT%H:%M:%SZ"
    r = salad.call("POST", salad.BASE + "/log-entries", {"start_time": time.strftime(f, time.gmtime(time.time() - minutes * 60)),
                   "end_time": time.strftime(f, time.gmtime()), "query": 'resource.type = "container" AND log contains "[mining g"',
                   "page_size": 100, "sort_order": "desc"}, timeout=30)
    out = {}
    for e in r.get("items") or []:
        m = re.search(r"\[mining (g\d+)\] \S+ ([0-9.]+) TH/s", e.get("text_log") or "")
        lab = (e.get("resource") or {}).get("labels") or {}
        if not m or not lab.get("container_group_name"): continue
        try: t = calendar.timegm(time.strptime(e["time"][:19], "%Y-%m-%dT%H:%M:%S"))
        except Exception: continue
        out.setdefault(lab["container_group_name"], []).append((t, lab.get("machine_id"), m.group(1), float(m.group(2))))
    return out

def miner_th(s, lines):
    # This group's rate now: the newest line per GPU, only from the current PC (after a move, the old PC's last
    # lines are still in the window for a few minutes).
    latest = {}
    for t, mach, g, th in lines.get(s["name"], []):
        if t < (s.get("run_since") or 0) - 5 or (s.get("machine") and mach and mach != s["machine"]): continue
        latest.setdefault(g, th)
    return sum(latest.values()) if latest else None

def end(s, why):
    log(f"{s['name']}: {why}; stopping and deleting the group", s)
    for l in container_log(s): log(f"why: {s['name']}: {l}", s)
    try: salad.stop(s["name"])
    except Exception as ex: log(f"{s['name']}: stop failed: {str(ex)[:120]}", s)
    time.sleep(5)
    try: salad.delete(s["name"])
    except Exception as ex: log(f"{s['name']}: delete failed: {str(ex)[:120]}", s)
    s["outcome"] = why; s["ended"] = time.time(); save(s)
def paid_before(s):
    # This card class has earned more than it costs on some PC, in this group or an earlier one
    if s.get("paid"): return True
    for f in glob.glob(os.path.join(GD, "*.json")):
        try: o = json.load(open(f))
        except Exception: continue
        if o.get("gpu") == s["gpu"] and o.get("paid"): return True
    return False

price, U = econ(); t_econ = time.time(); t_last = time.time(); tick = 0; bad = {}; hist = {}; mhist = {}
log(f"watching all Salad groups in {GD}")
while True:
    time.sleep(60); tick += 1; now = time.time()
    try:
        if now - t_econ > 600: price, U = econ(); t_econ = now
        live = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(GD, "*.json")))]
        live = [s for s in live if not s.get("outcome")]
        if not live: continue
        workers = get(POOL).get("workers") or []
        try: mlines = miner_lines()
        except Exception as ex: mlines = {}
        dt = now - t_last; t_last = now
        for s in live:
            for k, v in (("first_share", None), ("prl", 0.0), ("run_since", None), ("ever_ran", None), ("machine", None), ("spent", 0.0)): s.setdefault(k, v)
            g = salad.group(s["name"]); status = g.get("current_state", {}).get("status")
            if status == "stopped" and not s["ever_ran"] and not s.get("started"):
                salad.start(s["name"]); s["started"] = now; s["prio_since"] = now; log(f"{s['name']}: image ready; started", s)
            inst = salad.instances(s["name"]).get("instances") or []
            run = [i for i in inst if i.get("state") == "running"]
            states = ",".join(sorted(i.get("state", "?") for i in inst)) or "none"
            if run:
                m = run[0].get("machine_id")
                # A new PC starts a new rate window: time between PCs isn't billed and shouldn't dilute the rate
                if not s["run_since"]: s["run_since"] = now; hist[s["name"]] = []; log(f"{s['name']}: instance running on machine {m}", s)
                elif m != s["machine"]:
                    # Salad can move an instance with no gap in "running"; the new PC still gets its own window and 20 minutes
                    log(f"{s['name']}: moved to machine {m}", s); s["run_since"] = now; hist[s["name"]] = []; mhist[s["name"]] = []; bad[s["name"]] = 0
                s["machine"] = m; s["ever_ran"] = s["ever_ran"] or now
            elif s["run_since"]:
                log(f"{s['name']}: instance not running (states {states}, group {status}); Salad is finding another PC", s); s["run_since"] = None
            # A group that had a PC, lost it, and gets no other in 20 minutes is deleted, so its place in the quota goes to
            # another class (Oct 9: a 3090 waited 2 hours for a PC at batch, holding a slot).
            # A group Salad or the user stopped isn't looking for a PC, so this rule doesn't apply: leave it alone and say so
            # once (Oct 10 07:43: Salad stopped every group at once when the prepaid credit ran out). They don't restart
            # on their own: start them by hand after a top-up.
            if status == "stopped" and s["ever_ran"]:
                s.pop("lost_since", None); s.update(state=states, status=status, checked=now, th=0, earn_hr=0)
                if not s.get("stopped_seen"): s["stopped_seen"] = now; log(f"{s['name']}: group stopped (not by this watcher); leaving it alone", s)
                save(s); continue
            s.pop("stopped_seen", None)
            if run: s.pop("lost_since", None)
            elif s["ever_ran"]:
                if not s.get("lost_since"): s["lost_since"] = now
                if now - s["lost_since"] > 1200:
                    end(s, f"lost its PC and got no other in {(now - s['lost_since']) / 60:.0f} min"); continue
            # While Salad is still preparing the image ("pending") the group isn't looking for a PC yet, and a priority
            # change is refused ("pending_update_in_progress"); the 10 minutes start once it is.
            if status == "pending": s["prio_since"] = now
            if not s["ever_ran"] and not run and s.get("ladder") and now - s.get("prio_since", s["created"]) > 600:
                ps = [q for q, _ in s["ladder"]]
                nxt = ps.index(s["priority"]) + 1 if s["priority"] in ps else len(ps)
                if nxt >= len(ps):
                    end(s, f"no PC at {s['priority']} priority in 10 min, and no higher priority clears the margin"); continue
                q, c2 = s["ladder"][nxt]
                try:
                    salad.set_priority(s["name"], q)
                    log(f"{s['name']}: no PC at {s['priority']} priority in 10 min; moved to {q} (${c2:.3f}/hr)", s)
                    s.update(priority=q, price_hr=c2, prio_since=now)
                except Exception as ex: log(f"{s['name']}: priority change failed: {str(ex)[:150]}", s); s["prio_since"] = now
            # Some home PCs block outbound port 1200 (the pool's); box.sh then stops with "outbound port 1200 blocked"
            # and can never mine there (Oct 9: a 4080). Move the instance to another PC instead of paying for nothing,
            # up to 3 times a group.
            # Only until the pool has a share from this PC: first_share is the group's first ever, so it can't tell (every
            # group that had moved PCs kept querying the logs every other round, and Salad's log API answered 500s).
            seen = max([int(w.get("lastShare") or 0) for w in workers if (w.get("name") or "").startswith(s["worker"] + "-g")] or [0])
            if run and tick % 2 == 0 and now - s["run_since"] > 60 and seen < s["run_since"] - 60:
                f = "%Y-%m-%dT%H:%M:%SZ"
                q = f'resource.type = "container" AND resource.labels.container_group_name = "{s["name"]}" AND log contains "port 1200 blocked"'
                try:
                    r = salad.call("POST", salad.BASE + "/log-entries", {"start_time": time.strftime(f, time.gmtime(s["run_since"] - 120)),
                                   "end_time": time.strftime(f, time.gmtime()), "query": q, "page_size": 1})
                    if r.get("items"):
                        if s.get("moves", 0) >= 3: end(s, "three PCs in a row blocked the pool port"); continue
                        salad.reallocate(s["name"], run[0]["id"]); s["moves"] = s.get("moves", 0) + 1
                        log(f"{s['name']}: PC {s['machine']} blocks the pool port (1200); moved it to another PC ({s['moves']} of 3)", s)
                        s["run_since"] = None; save(s); continue
                except Exception as ex: log(f"{s['name']}: port check failed: {str(ex)[:120]}", s)
            ws = [w for w in workers if (w.get("name") or "").startswith(s["worker"] + "-g")]
            H = sum(float(w.get("hashes") or 0) for w in ws)
            last = max([int(w.get("lastShare") or 0) for w in ws] or [0])
            good = sum(int(w.get("shares_good") or 0) for w in ws)
            # A recreated group mines under the same worker name as the one it replaced, so the pool can still show
            # that group's last share: only a share after this group was created counts (Oct 9, the 3090).
            if last and not s["first_share"] and last >= s["created"]: s["first_share"] = last; log(f"{s['name']}: first share at the pool; mining", s)
            h = hist.setdefault(s["name"], []); h.append((now, H)); h[:] = [x for x in h if now - x[0] <= 1800]
            if len(h) >= 2 and h[-1][0] - h[0][0] >= 300: th = (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0]) * UNIT
            elif s["first_share"]: th = sum(float(w.get("hashrate") or 0) for w in ws) * UNIT
            else: th = 0.0
            if run: s["spent"] += s["price_hr"] / 3600 * dt
            # Prefer the miner's own rate, averaged over the last 30 minutes on this PC
            if run and s["first_share"]:
                try:
                    mt = miner_th(s, mlines)
                    if mt is not None:
                        mh = mhist.setdefault(s["name"], []); mh.append((now, mt)); mh[:] = [x for x in mh if now - x[0] <= 1800 and x[0] >= s["run_since"]]
                except Exception: pass
            mh = [x for x in mhist.get(s["name"], []) if run and x[0] >= (s["run_since"] or now)]
            if mh: th = sum(x[1] for x in mh) / len(mh)
            if run and th: s["prl"] += th * U / price / 3600 * dt
            cost = s["price_hr"]; earn = th * U
            s.update(th=round(th), earn_hr=round(earn, 4), state=states, status=status, shares=good, mined_usd=round(s["prl"] * price, 4), checked=now)
            # A share is 2^21 pool units, so a slow card finds one only every few minutes (a 2060 at 47 TH/s: ~3.3 min on
            # average; Oct 9 one went 11 min without one while mining steadily, and was deleted by mistake). Wait 8 times
            # the card's expected gap, at least 10 minutes: a working miner goes that long without a share about 1 time in 3000.
            quiet = max(600, 8 * 2 ** 21 * UNIT / th) if th > 0 else 600
            if run and now - s["run_since"] > 1200 and now - last > quiet:
                why = (f"instance running {(now - s['run_since']) / 60:.0f} min, " +
                       ("no share since it started" if last < s["run_since"] else f"no share for {(now - last) / 60:.0f} min"))
                # A class that has paid before: the PC is the problem (Oct 9: a 4070 Super PC that mined at +44% earlier
                # later couldn't reach github.com to fetch the miner). Same moves as a slow PC, sharing its 3.
                if paid_before(s) and s.get("slow_moves", 0) < 3:
                    try:
                        salad.reallocate(s["name"], run[0]["id"])
                        s["slow_moves"] = s.get("slow_moves", 0) + 1
                        log(f"{s['name']}: {why} on PC {s['machine']}; the class has paid before, so moved it to another PC ({s['slow_moves']} of 3)", s)
                        for l in container_log(s, 25): log(f"why: {s['name']}: {l}", s)
                        s["run_since"] = None
                    except Exception as ex: log(f"{s['name']}: move to another PC failed, trying again at the next check: {str(ex)[:120]}", s)
                    save(s); continue
                end(s, why); continue
            if tick % 5 == 0:
                margin = (earn - cost) / earn * 100 if earn else float("-inf")
                log(f"  {s['name']}: group {status}, instance {states}; ~{th:.0f} TH/s over {(now - h[0][0]) / 60:.0f} min, {good} shares; "
                    f"earns ${earn:.3f}/hr vs ${cost:.3f}/hr ({margin:+.0f}%); spent ~${s['spent']:.3f}, mined ~${s['prl'] * price:.3f}", s)
                # A test group (s["test"]) is there for its measurements, not its mining margin: the user approved its cost
                # Enough readings to judge: 10+ minutes of the miner's own rate (4+ readings), else 15+ minutes of pool
                # readings. A span, not a count: with 10 groups a round takes ~2.5 minutes, so 30 minutes holds only ~12
                # rounds, and some miner-log queries time out (Oct 10: a 4070 Ti sat at -6% for 40 minutes unjudged).
                mh_, h_ = mhist.get(s["name"], []), hist.get(s["name"], [])
                ready = (len(mh_) >= 4 and mh_[-1][0] - mh_[0][0] >= 600) or (len(h_) >= 2 and h_[-1][0] - h_[0][0] >= 900)
                if run and not s.get("test") and s["first_share"] and now - s["run_since"] > 1200 and ready:
                    if earn >= cost: s["paid"] = True; s["slow_moves"] = 0
                    bad[s["name"]] = bad.get(s["name"], 0) + 1 if earn < cost else 0
                    if bad[s["name"]] >= 2:
                        if paid_before(s) and s.get("slow_moves", 0) < 3:
                            try:
                                salad.reallocate(s["name"], run[0]["id"])
                                s["slow_moves"] = s.get("slow_moves", 0) + 1; bad[s["name"]] = 0
                                log(f"{s['name']}: earning ${earn:.3f}/hr ({th:.0f} TH/s), less than its ${cost:.3f}/hr cost, on PC {s['machine']}; "
                                    f"the class has paid before, so moved it to another PC ({s['slow_moves']} of 3)", s)
                                s["run_since"] = None
                            except Exception as ex: log(f"{s['name']}: move to another PC failed, trying again at the next check: {str(ex)[:120]}", s)
                            save(s); continue
                        end(s, f"earning ${earn:.3f}/hr, less than its ${cost:.3f}/hr cost, on two checks in a row" +
                               (" on its 4th slow PC in a row" if s.get("slow_moves") else "")); continue
            save(s)
        if tick % 15 == 0:
            run = [json.load(open(path(s))) for s in live]; run = [s for s in run if not s.get("outcome")]
            n_run = sum(1 for s in run if "running" in (s.get("state") or ""))
            rate = sum(s['price_hr'] for s in run if 'running' in (s.get('state') or ''))
            log(f"summary: {len(run)} Salad groups, {n_run} on a PC, {sum(s.get('th', 0) for s in run):.0f} TH/s, earning "
                f"${sum(s.get('earn_hr', 0) for s in run):.2f}/hr vs ${rate:.2f}/hr; "
                f"spent ~${sum(s['spent'] for s in run):.2f}, mined ~${sum(s.get('mined_usd', 0) for s in run):.2f} at PRL ${price:.3f}")
            # Salad's API has no balance, so count down from the starting credit by what every group has cost
            if salad.CFG.get("credit"):
                every = [json.load(open(f)) for f in glob.glob(os.path.join(GD, "*.json"))]
                spent = sum(s.get("spent", 0) for s in every); prl = sum(s.get("prl", 0) for s in every); left = salad.CFG["credit"] - spent
                log(f"credit: spent ~${spent:.2f} of ${salad.CFG['credit']:.2f} and mined ~{prl:.3f} PRL in all; ~${left:.2f} left"
                    + (f", about {left / rate:.0f} hours at ${rate:.2f}/hr" if rate else ""))
    except Exception as ex:
        log(f"check failed: {str(ex)[:150]}")
