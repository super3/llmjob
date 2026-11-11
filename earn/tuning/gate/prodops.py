import collections, re, sys
for path in sys.argv[1:]:
    on = False; seen = False; c = collections.Counter(); n = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("//---") and ".text." in line:
            on = ".text.pearl_tile_fold_tall " in line; seen = False; continue
        if not on: continue
        if "pearl_gate_producer" in line and line.rstrip().endswith(":"):
            seen = True; continue
        if not seen: continue
        m = re.match(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?);", line)
        if m:
            t = m.group(2).strip()
            if t.startswith("@"): t = t.split(None, 1)[1]
            c[t.split()[0].split(".")[0]] += 1; n += 1
    print(path.split("/")[-2], n, dict(c.most_common()))
