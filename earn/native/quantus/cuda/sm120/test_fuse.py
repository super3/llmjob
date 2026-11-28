"""test_fuse.py: fuse.py's pattern matching and encoding on small hand-written listings. No toolkit or card needed.

  python test_fuse.py        (test/quantusRed4.test.js runs it when a Python is there)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuse  # noqa: E402


def listing(texts, hi=None):
    return [{"addr": 16 * k, "text": t, "lo": 0, "hi": (hi or {}).get(k, 0), "labels": []}
            for k, t in enumerate(texts)]


# find() only trusts a carry whose predicate is written again later in the block (so it knows every reader):
# four adds that do that, as the kernel's next sum would.
END = ["IADD3 R30, P0, PT, R31, R32, RZ", "IADD3 R33, P1, PT, R31, R32, RZ", "IADD3 R34, P2, PT, R31, R32, RZ",
       "IADD3 R35, P3, PT, R31, R32, RZ"]


def test_find_pairs():
    # a 64-bit sum of three values, as ptxas writes it: two adds a word, the carries ending in one add of both
    ins = listing(["IADD3 R4, P0, PT, R1, R2, RZ",
                   "IADD3 R4, P1, PT, R4, R3, RZ",
                   "IADD3.X R8, P2, PT, R9, R10, RZ, P0, !PT",
                   "IADD3.X R8, P3, PT, R8, R11, RZ, P1, !PT",
                   "IADD3.X R12, PT, PT, RZ, RZ, R12, P2, P3"] + END)
    pairs = fuse.find(ins, 0, 8)
    assert [(c["i"], c["j"]) for c in pairs] == [(0, 1), (2, 3)], pairs
    assert fuse.text_of(pairs[0]) == "IADD3.X R4, P0, P1, R1, R2, R3, !PT, !PT"
    assert fuse.text_of(pairs[1]) == "IADD3.X R8, P2, P3, R9, R10, R11, P0, P1"


def test_find_refuses():
    base = ["IADD3 R4, P0, PT, R1, R2, RZ",
            "IADD3 R4, P1, PT, R4, R3, RZ",
            "IADD3.X R8, P2, PT, R9, R10, RZ, P0, !PT",
            "IADD3.X R8, P3, PT, R8, R11, RZ, P1, !PT",
            "IADD3.X R12, PT, PT, RZ, RZ, R12, P2, P3"]
    # the first add waits on a scoreboard: its pair is left alone, the other still fuses
    pairs = fuse.find(listing(base + END, {0: 1 << 52}), 0, 8)
    assert [(c["i"], c["j"]) for c in pairs] == [(2, 3)], pairs
    # the two carries are not read as the two carry-ins of one add: nothing fuses
    t = list(base)
    t[4] = "IADD3.X R12, PT, PT, RZ, RZ, R12, P2, !PT"
    assert fuse.find(listing(t + END), 0, 8) == []
    # the first add's result is read twice
    t = list(base)
    t[4] = "IADD3.X R12, PT, PT, R4, RZ, R12, P2, P3"
    t.insert(1, "IADD3 R20, PT, PT, R4, R4, RZ")
    assert all(c["i"] != 0 for c in fuse.find(listing(t + END), 0, 9))
    # a subtraction's "-R" needs a carry-in slot for its +1: one is free here, so it fuses as ~R and a true carry
    pairs = fuse.find(listing(["IADD3 R4, PT, PT, R1, -R2, RZ", "IADD3 R4, PT, PT, R4, R3, RZ"]), 0, 1)
    assert len(pairs) == 1 and fuse.text_of(pairs[0]) == "IADD3.X R4, PT, PT, R1, ~R2, R3, PT, !PT", pairs


def test_encode():
    # ptxas's own "IADD3.X R50, PT, PT, R51, R44, R47, P3, P6" as the template; the expected words are what
    # nvdisasm read back as the texts below from a fused sm_120 cubin (28 Nov).
    tmpl = {"lo": 0x0000002c33327210, "hi": 0x000fe40001fec42f}
    r = lambda n, neg="": ("r", n, neg)
    c = {"d": 30, "pouts": (4, 6), "srcs": [r(31), r(25), r(31)], "cins": [(4, False), (6, False)]}
    lo, hi = fuse.encode_add(c, tmpl)
    assert fuse.text_of(c) == "IADD3.X R30, P4, P6, R31, R25, R31, P4, P6"
    assert (lo, hi & 0xFFFFFFFF) == (0x000000191f1e7210, 0x0268c41f), (hex(lo), hex(hi))
    c = {"d": 6, "pouts": (4, 6), "srcs": [r(30), r(3), r(30)], "cins": []}
    lo, hi = fuse.encode_add(c, tmpl)
    assert fuse.text_of(c) == "IADD3.X R6, P4, P6, R30, R3, R30, !PT, !PT"
    assert (lo, hi & 0xFFFFFFFF) == (0x000000031e067210, 0x07e9e41e), (hex(lo), hex(hi))
    # the control field (stall and the rest, bits 41 up of the high word) is the template's: apply() sets it
    assert hi >> 41 == tmpl["hi"] >> 41


def test_red4():
    tail = ["IADD3 R9, P2, PT, R2, -R3, RZ",
            "IADD3.X R13, P2, PT, R13, R3, RZ, P2, !PT",
            "IADD3.X R11, P4, PT, ~R6, R9, RZ, P2, !PT",
            "IADD3.X R21, PT, PT, RZ, -0x2, R13, P4, P2"]
    assert fuse.find_red4(listing(tail)) == ([2], 1)
    # CUDA 13's two-input forms, and the carry renamed between the first two adds
    tail13 = ["IADD R9, P3, R2, -R3", "IADD.X R13, P2, R13, R3, P3", "IADD.X R11, P4, ~R6, R9, P2",
              "IADD3.X R21, PT, PT, RZ, -0x2, R13, P4, P2"]
    assert fuse.find_red4(listing(tail13)) == ([2], 1)
    # r3 zero: -0x1 in ~r3's place
    t = list(tail)
    t[2] = "IADD3.X R11, P4, PT, R9, -0x1, RZ, P2, !PT"
    assert fuse.find_red4(listing(t)) == ([2], 1)
    # a tail whose first add is missing, or whose second adds another register, is seen but not matched
    assert fuse.find_red4(listing(tail[1:])) == ([], 1)
    t = list(tail)
    t[1] = "IADD3.X R13, P2, PT, R13, R7, RZ, P2, !PT"
    assert fuse.find_red4(listing(t)) == ([], 1)
    # two tails interleaved
    two = [tail[0], tail13[0].replace("R9", "R39").replace("P3", "P5").replace("R2,", "R32,").replace("-R3", "-R33"),
           tail[1], "IADD.X R43, P5, R43, R33, P5", tail[2], "IADD.X R41, P6, ~R36, R39, P5", tail[3],
           "IADD3.X R51, PT, PT, RZ, -0x2, R43, P6, P5"]
    assert fuse.find_red4(listing(two)) == ([4, 5], 2)


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
    print("fuse.py tests passed")
