"""splitlog.py <log> <out1> <out2> : cut a box's log where its second race starts ("card before race 2")."""
import sys

src, o1, o2 = sys.argv[1:4]
lines = open(src, encoding="utf-8", errors="replace").read().split("\n")
cut = next((i for i, l in enumerate(lines) if "card before race 2" in l), None)
if cut is None:
    open(o1, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    print("no second race; wrote", len(lines), "lines to", o1)
else:
    open(o1, "w", encoding="utf-8", newline="\n").write("\n".join(lines[:cut]))
    open(o2, "w", encoding="utf-8", newline="\n").write("\n".join(lines[cut:]))
    print("first", cut, "lines; second", len(lines) - cut)
