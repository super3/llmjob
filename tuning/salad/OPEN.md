# Open problems

Every known problem with the Salad fleet, the watcher, its report, the trader and the benchmark tools. An item
leaves this file only with a fix commit or the founder's decision, and keeps its number for good. Costs are per day
at the fleet of 22 October (about 7 boxes, PRL $2.18, the fleet making about +$1.7/day), from the live ledger and
events. They overlap, so they don't add up exactly.

Built on 22 October from a sweep of the whole session's chat, the code and two days of live data, each item
checked against the code as it stood then. Until then these were scattered over "waiting on you" lines in report
notes, and many had dropped off them with no fix or decision.

## Fixed

| # | Problem | Cost while open | Fixed in |
|---|---|---|---|
| 1 | A group on any PC it had ever left was skipped by every rule. 2070-3 billed at 0 TH/s for 4.5 h. | $1.44/day while it lasted, about 80% of fleet profit | `9df635e` |
| 2 | The report summed only page 1 of Kryptex's payouts: margins read about 25 points low for about a day. | Misled every decision read off the report | `b50df9c` |
| 3 | A box paused for a prompt showed a margin like −3180% and turned the All row negative. | Display | `8e39885` |
| 4 | The keep, leave, climb and search rules valued every box at HeroMiners' credit ratio (0.93) though every box mines on Kryptex (0.97). 63 of 125 margin moves and deletions in the 48 h before would not have happened. A review before it went live added: the pool a box's PC really mines on, Kryptex's ratio held through API outages, and a second margin check in the search. Live since 8:15 pm EDT on 22 Oct. | $0.2–0.5/day | `310c1f9`, `3874caf` |
| 8 | `max_active` was held at 7–8 for the AMD benchmark from 4:24 pm on 22 Oct. | about $0.5/day while it ran | back to 10 at 8:25 pm when the run ended |
| 7 | The search sized its room by our own group count, so with Salad counting a phantom replica every search sent creates it refused, one per class. It now uses Salad's own count of replicas in use and stops at the first refusal. The phantom replica itself is Salad's and still costs a slot: worth asking Salad support. | $0.3–0.5/day while the phantom lasts | `89bf78c` (code part) |
| 15 | Moves off port-blocked PCs counted toward a group's 3, so 4060-54 was deleted on its fourth PC for a PC's fault. Now they don't, up to 6 a group. | about $0.08/day | `89bf78c` |
| 9 | The bad-PC list kept no rate, so a PC left at high was refused at medium where it cleared, and entries made at the wrong ratio (#4) barred good PCs for 24 h. A PC left for its margin now keeps the TH/s it was left at (read from the reason for older entries), and a group that rate clears 10% for, at its own price and pool, keeps the PC. In the event log, 277 of 681 moves off PCs left for their margin were off PCs that clear for the group moved at the price and ratio of 23 October. | $0.2–0.3/day | `26bb0ee` |
| 13 | Bad-list entries expire after 24 h, so known-slow PCs came back and cost a counted move: 85 of 452 margin moves in 14 days were off PCs that had failed for their class 1–7 days before. A new PC whose rate measured in the last 7 days fails 10% at the group's price and pool is now left at once, uncounted. | $0.1–0.2/day | `9c89f14` |
| 5 | The rules counted Pearl only, not each box's Monero (about $0.002/hr a box), so boxes that cleared 10% with it were left: 56 of 151 margin moves and deletions in the 48 h before. They now add the median Monero box's rate. Offered at 8:03 pm with a 9 pm default; live since 9:00 pm EDT on 22 Oct. | $0.3–0.7/day | `0453786` |
| 6 | A class whose last 3 groups got no PC only went last in the search and still took any slot nothing else wanted: on 22 Oct the account was full for 573 min, in 567 of them with such a group waiting, and a class that gets PCs waited for one 84 times. Such a class is now left out for 3 h. Offered at 8:03 pm with a 9 pm default; live since 9:00 pm EDT on 22 Oct. | about $0.15–0.3/day | `45d9ca4` |
| 12 | A box Salad restarted on the same PC went quiet while it started again, and the 6-minute rule gave the PC up. A restart on a PC the box mines on in this stint now starts the reading and share clocks again, once a PC. Live since 9:00 pm EDT on 22 Oct. The first figure given, "34 of 58 such moves in 14 days", was wrong (it counted first starts): the review on 23 Oct found 4 of 39 in 12 days (3090ti-19, 5060-46, 3060-54, and 2070-12, which couldn't mine), so about 3 good PCs kept and 6 later bad-list moves avoided. A box's first start on a PC it had mined on hours before was also taken as a restart; fixed in `7c550e6`. | about $0.01–0.02/day (it was given as $0.1–0.3) | `36ce1f2` |
| 35 | Leave messages rounded the margin, so a box was dropped as "+10%, under 10%". A margin that would round to the bar now shows one decimal, rounded down. Live since 9:01 pm EDT on 22 Oct. | a wrong number in events | `7f1064e` |
| 39 | The Monero market read had no sanity check or back-off. At least seven HeroMiners reads on 22 Oct gave a hash 10 to 2,300 times too little; each was kept 5 minutes or more, valued the chart buckets first seen then at almost nothing (19 buckets, 0.000016 XMR short in the XMR column) and, from 9 pm, took the rules' Monero to nearly 0: at 9:20 pm 4060-56 lost a PC at +10.5% with Monero (+8% without), a counted move. A read 3 times off is now set aside until reads agree on it for 20 minutes (taking it on the second read let a bad answer through at 02:01Z on 23 Oct, for 7 minutes; fixed in `f66ac93`), off buckets are valued again at the median, and a failed read waits 5 minutes. Live since 9:25 pm EDT on 22 Oct. | wrong XMR figures; one good PC and a counted move | `5651027` |
| 48 | A PC left by rule 5 at its first judgment never had its rate recorded (pc_rates), so the 7-day check from #13 missed about half the PCs it was for. Found by the review on 23 Oct; the rate it was left at is recorded now. | part of #13's $0.1–0.2/day | `7c550e6` |
| 49 | One warm-up reading right after a Gemma pause could stand as a PC's rate (0.1 TH/s on 3060-70 at 00:55Z, 23 Oct), and the 7-day check would then move good groups off it. The 2 minutes after a pause are left out now, as rule 5 does. | about one bad rate in 10 days, 1–3 good PCs refused each | `7c550e6` |
| 50 | A box's first start on a PC it had mined on hours before was taken as a restart (#12), moving its clocks and using up its once-a-PC allowance. Only readings from this stint count now. | small | `7c550e6` |
| 51 | When Kryptex's Monero balance or payouts call failed, the boxes' Kryptex rates weren't read either; after 15 minutes the rules' Monero went to 0 and boxes kept only by their Monero were left. The rates are read first now. | a few counted moves per outage | `7c550e6` |
| 52 | The rules' Monero median took the upper middle of an even count, and Kryptex Monero was valued at HeroMiners' 99.1% instead of Kryptex's 94%, in the rules and the report's Profit/hr (about 5% high on the Monero part). | about $0.0001/hr a box in Profit/hr | `7c550e6` |
| 53 | A box long on its PC was credited the median box's Monero, not its own, so one whose Monero doesn't reach the pool (#40) kept credit it doesn't earn. After 30 minutes of the pools' averages on this PC its own figure counts. | up to $0.002/hr on such a box | `7c550e6` |
| 54 | The no-PC hold (#6) also held a class whose no-PC tries were all cheaper than the priority it places at (4060-32 waited 64 minutes on 19 Oct). Such a class is tried again at that priority or above. | under $0.01/day | `7c550e6` |

## Open: costs money

| # | Problem | Cost a day | Fix |
|---|---|---|---|
| 10 | One 10% bar for creating, keeping and climbing, with no buffer, so PCs near the line flip in and out as the price moves. | $0.05–0.1 alone; about $0.5 with #4, #5 | A lower keep bar (or two failed checks) and a higher create bar. Decision. |
| 11 | Gemma's prompts pause mining on billed PCs, sometimes 4–11 min while the model loads; Gemma's earnings aren't counted anywhere. | about $0.13 billed with no Pearl | CLI: don't pause until the model answers (release). Watcher: show pause minutes and Gemma's earnings. |
| 14 | Release v0.6.7's Linux cores need glibc 2.38; the boxes' image has 2.35, so no box gets its 4.6–6.6% faster RTX 20 mining. | about $0.10 a mining RTX 20 box | Build the Linux cores against an older glibc (release fix in `earn/`). |
| 16 | PCs that keep dropping or restarting our boxes are never held back. | $0.05–0.12 | Bad-list a PC that drops a box 3 times in 30 min. |
| 55 | Of the 39 "no mining reading for 6 min" moves in 12 days, #12 catches 4. Four more came after Salad restarted the box on the same PC with no release line before the rule fired (3060-4, 2080ti-28, 5060-4, 4070ti-18), and 31 had no restart: the miner just went quiet. Found by the review on 23 Oct. | unknown share wrong; each a counted move and a PC | Read the box's log lines before each of the 31; count readings across a same-PC LOST/NEW (#17). |
| 17 | One flaky Salad instance read is taken as real: a false LOST PC, a "new PC" on the same machine. | $0.05–0.15 | Need two reads that agree. |
| 18 | The CLI never resumes mining when a prompt pauses it before its first hashrate. | $0.05–0.1 | CLI fix (release). |
| 19 | Rule 5 averages the last 3 readings, so one dip or warm-up reading moves a good box. | about $0.05 | Use the median, or more readings. |
| 20 | A no-reading leave bars PCs that were fine (slow image start, a one-off fault, our own failed release). | $0.03–0.05 | Don't bad-list a PC for one no-reading leave. |
| 21 | A brief PC counts as "had a PC", so the next group of the class starts at batch and climbs. | about $0.04 | Count a PC only after it mined. |
| 22 | New groups come up stopped and a failed start is retried every 5 min without the reason. | about $0.03 | Start at once, log the reason. |
| 23 | Salad can run two copies of one group; the watcher sees one. | $0.02–0.05 | Read every instance; delete the extra. |
| 24 | Monero boxes ask Salad for 8 vCPUs and 8 GB, but the search checks free GPUs at 2 and 4. | unknown | Check at the size the box asks for (1 line), then compare time to a PC. |
| 25 | The search values a class at a rate its PCs don't make at the priority Salad gives them. | up to $0.01/hr a box | Rates by priority. |
| 26 | Shares stop reaching the pool while the miner still reports hashing. | $0.01–0.02 | Leave on no share for 20 min whatever the miner says (check the existing no-share rule covers Kryptex). |
| 27 | The watcher runs inside this session's container. It and the keeper were down 167 min on 22 Oct (09:38–12:25Z) and 101 min on 23 Oct (02:36–04:17Z, when the session's worker restarted), with no alert. | small on average; a risk | Run it somewhere that outlives the session, with an alert when it stops. |
| 28 | The trader and SafeTrade can't be read from here, so unsold PRL or failed sales go unnoticed (it couldn't sell from 18 Oct 17:51Z until #269). | risk on about $9.4/day of PRL | A Railway token, then the trader's last sale and unsold PRL in the report. |
| 29 | The RX 9070 XT pays on Salad (92.8 TH/s, about $0.02/hr at low) but the fleet can't rent AMD. | missed gain, about $0.5 a box | Release the AMD core; add a 9070 XT class. Decision. |
| 30 | Smaller, none seen in the last 2 days: a failed bad-list move isn't retried; the swap rule can swap out a box still warming up; a class rate can come from one capped PC; the card line isn't used to reject a wrong card; our own groups can outbid each other; a failed miner download costs a counted move; a 5090 would run the 27B model, which stops mining for each prompt; a release trial runs on a box that mines well; a restart before a save can lose a counted move; the swap rule prices a replacement before the box it replaces frees a Kryptex slot (errs on the safe side). | under $0.01 each | Each when it next shows. |

## Open: numbers that are wrong or missing

| # | Problem | Fix |
|---|---|---|
| 31 | The hourly drift check covers HeroMiners only, so it has said nothing since every box moved to Kryptex; it would have caught #2. | Check Kryptex's hashrate and credit too. |
| 32 | No view of what we should have earned against what the pools credited, over time. | A daily line in the report or README. |
| 33 | "Salad credit left" is our own count and drifts (the 167-min outage left about $0.8 uncounted). | Check against the portal; note the gap. |
| 34 | During a Salad log outage the report shows mining boxes at 0 TH/s with no stale mark. | Mark rows whose last reading is old. |
| 36 | Other report figures: the Kryptex pool row, the All row's "+100%", credit hours, stale rows. | Each checked and fixed. |
| 37 | A FIRST SHARE event can report a share from the PC the box was just moved off. | Match the share to the current PC. |
| 38 | `events.log` lines carry no date. | Add the date. |
| 40 | Monero on some boxes doesn't reach the pool (they show 0.0 kH/s while Pearl mines). | Find out why; $0.09/day a box. |

## Open: other

| # | Problem | Fix |
|---|---|---|
| 41 | `bench.py` counts minutes Salad refused a start for quota toward its 30-minute no-PC limit; lane a nearly skipped the 9070 XT for it. | Restart the clock on a quota refusal. |
| 42 | `bench.sh` finds a PC that can't reach the pool only after its hit check (23 min lost on 22 Oct). | Check the pool port first, as box.sh does. |
| 43 | `trader/README.md` says Railway reads `railway.toml`; it doesn't. | Fix the README. |
| 44 | Coins sit at the pools below their payout thresholds (about 0.6 PRL, 0.0026 XMR). | Lower Kryptex's Monero threshold to 0.01 on its site. |
| 45 | Every box installs build tools it never uses. | Drop them from box.sh's setup. |
| 46 | Review fixes 1, 3, 4, 8, 9 and 11 from before 13 October were never decided, and two benchmark fixes from 18 October have unknown status. | Look each up and decide. |
| 47 | The CI core-picking script's two edge cases (a stacked PR's base with only pull_request runs; a run deleted by hand mid-build). | Leave unless they happen. |
