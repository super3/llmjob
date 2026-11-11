"""mkrecord.py : write record-tables.md from the runs and logs in this folder (uses mdtable.py)."""
import io
import os
import re
import sys
import textwrap
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = ["mdtable.py", "none"]          # mdtable runs its main on import; give it nothing to do
import mdtable  # noqa: E402


def cap(fn, paths):
    paths = [os.path.join(HERE, p) for p in paths if os.path.exists(os.path.join(HERE, p))]
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(paths)
    return buf.getvalue()


def blocks(text):
    """Split mdtable output into (comment, table) pairs."""
    out, cur = [], None
    for line in text.split("\n"):
        m = re.match(r"<!-- (.*) -->", line)
        if m:
            cur = [m.group(1), []]
            out.append(cur)
        elif cur is not None:
            cur[1].append(line)
    return [(c, "\n".join(t).strip("\n")) for c, t in out]


parts = []
rig = blocks(cap(mdtable.rig, ["runs/p0_450.jsonl", "runs/s1_450.jsonl", "runs/s2_450.jsonl", "runs/s1_300.jsonl",
                               "runs/s2_300.jsonl"]))
for c, t in rig:
    pl = c.split(" W")[0]
    m = re.match(r"\d+ W: control (\d+) runs, ([\d.]+) to ([\d.]+), range ([\d.]+)%", c)
    extra = (" The three competitors read 338.6 to 341.9 TH/s on this card at 450 W and 2625 MHz (item 16)."
             if pl == "450" else " No competitor was run at 300 W.")
    head = (f"**RTX 4090, rig01, {pl} W limit, `hashrate.js`.** The control ran {m.group(1)} times, {m.group(2)} to "
            f"{m.group(3)} TH/s, a range of {m.group(4)}%. Each arm's change is against the controls on either side "
            f"of it.{extra}")
    parts.append(textwrap.fill(head, 120) + f"\n\n{t}")

SALAD = [
    ("RTX 5090, Salad `08f8e389`, 575 W limit, race mode", ["logs-5090-1.txt", "logs-5090-2.txt"],
     ["The competitors, then the first race (5 s turns).",
      "Our CLI and a second race on the same PC (10 s turns). Its spin arms are the first version of the spin "
      "kernel, whose waiting blocks slept 250 ns at a time."]),
    ("RTX 3060, Salad `16548f76`, 170 W, 66 °C, race mode", ["logs-3060-1.txt"],
     ["The competitors, then the race. The second race never printed on this PC."]),
    ("RTX 3090, Salad `71128c8f`, 390 W limit, 88 °C, race mode", ["logs-3090-r1.txt", "logs-3090-r2.txt"],
     ["The competitors and our CLI, then the first race. The driver named heat as what held the clock in 352 of "
      "359 samples over both races (`clocks_event_reasons` 0x20).",
      "The second race: the sham's parts and the spin kernel (its waiting blocks spin on a barrier test)."]),
    ("RTX 5070 Ti, Salad `d0a033a5`, 300 W limit, race mode", ["logs-5070ti-r1.txt", "logs-5070ti-r2.txt"],
     ["The competitors and our CLI, then the first race. The card ran 2805 MHz in nearly every arm, at 165 to 220 W "
      "of its 300 W limit and 58 to 61 °C, so nothing but its top clock held it.",
      "The second race: the sham's parts and the spin kernel (its waiting blocks wait on a barrier, which "
      "suspends them on sm_120)."]),
]
for title, logs, notes in SALAD:
    text = cap(mdtable.salad, logs)
    body = []
    for n, (c, t) in enumerate(blocks(text)):
        note = notes[n] if n < len(notes) else ""
        if note:
            body.append(textwrap.fill(note, 120))
        body.append(t)
    parts.append(f"**{title}.**\n\n" + "\n\n".join(b for b in body if b))

open(os.path.join(HERE, "record-tables.md"), "w", encoding="utf-8", newline="\n").write("\n\n".join(parts) + "\n")
print("wrote", sum(p.count("\n") for p in parts), "lines")
