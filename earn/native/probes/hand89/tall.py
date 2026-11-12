"""tall.py: TIMING ONLY. Prices a warp tile of 192 rows by 32 columns (twelve m16 tiles by four n8 tiles, the same
48 IMMA and 192 sums a k-step) on rig01's RTX 4090, against the rotation fold's 96 by 64. Every result is wrong
on purpose; the stream is for the clock and the pipe only.

  python tall.py <base.cubin> <base.sass> <nvdisasm> <out.cubin> [key=value ...]

Why (goal-nvk11, S/item-reports/goal-nvk11.md). At 450 W the board's power holds the clock, and what costs power
is instructions: a B gather (one LDSM and four ALU steps) is worth about 20 MHz a warp a k-step, a 16-byte shared
load or store a few. With 32 columns a warp, B is four gathers a k-step, not eight. A' for the block's 192 rows is
then shared by all eight warps: it is made once (24 gathers a k-step for the block, 3 a warp) into a stage a
k-step ahead, and every warp loads its twelve A fragments from the stage, one LDS.128 every four IMMA. So a warp
a k-step: 7 gathers, 12 LDS.128 and 2 STS.128 (the rotation fold: 11 gathers, 4.75 LDS.128, 1.75 STS.128), and a
barrier of all eight warps at every k-step's end, since every warp reads what every warp made.

What the stream does, a warp a k-step (48 gaps, gap g behind IMMA g; IMMA i is A tile i // 4, n8 tile i % 4):
  - A tile m + 1's fragment, LDS.128 from the stage buffer of this k-step (the next k-step's for tile 0), in
    gap 4 m + 1; tile m + 1's IMMA wait for it;
  - B's four gathers for the next k-step (pairs 0, 1, two halves) from gap `bl0`, `bls` apart, their PRMT where
    the pair's last reader allows (gaps 46 + 2 p, + 1);
  - three gathers of A' (stand-ins in B's tables of pairs 2 and 3, which this tile does not use) into a quad of
    four registers and two STS.128 of it into the other stage buffer;
  - the offsets record (8 bytes, parted in two ALU steps), the readout (96 LOP3 a chunk), as rfold.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
from asm89 import Ins  # noqa: E402
import fold  # noqa: E402
from fold import addi, andi, muladd, shr  # noqa: E402
import gfold  # noqa: E402
from gfold import ACC, AF, SB_OFF, SB_RD, gather_ins, ldg  # noqa: E402
import rfold  # noqa: E402

CFG = dict(rfold.CFG)
CFG.update(dict(bl0=28, bls=3, mk0=3, mks=6, nbar=0))
BT = 12                     # B's eight registers (four n8 tiles, two words each)
MQ = 20                     # the quad A' gathers are made into, then stored
SB_ST = 3


def chunk(cfg, R):
    ni, tot, nt = 48, 192, 12
    G = [[] for _ in range(tot)]
    iwait = {}
    lead = cfg["lead"]

    def put(g, ins, wrapped=None):
        G[g % tot].append((ins, g >= tot if wrapped is None else wrapped))

    plans = []          # (LDSM gap, IADD3 gap, PRMT gap, address register, offset, d1, d2, rsb, PRMT wait)
    for s in range(4):
        b = ni * s
        cur, nxt = R["oq"][s & 1], R["oq"][(s + 1) & 1]
        put(b + 2, ldg(nxt, R["poff"], 0x100 * (s + 1), size=64, wsb=SB_OFF, rsb=SB_RD if s == 3 else None,
                       wait=(SB_RD,)))
        first = b + cfg["bl0"] - 4
        put(first, Ins("LOP3.LUT R%d, R%d, 0xffff, RZ, 0xc0, !PT" % (nxt + 2, nxt + 1)), False)
        put(first, Ins(shr(nxt + 3, nxt + 1, 16)), False)
        for g in (first - 2, first - 1, first):
            iwait.setdefault(g, []).append(SB_OFF)
        # A tile m + 1 from the stage: buffer s & 1 (tile 0 of the next k-step: the other buffer).
        for m in range(nt):
            # Stand-in places: the warp's own six tiles of rfold's stage (R0), which its stores below refresh
            # every turn and a half, so the operands keep changing as real ones would.
            off = 512 * ((m + 1) % 6)
            put(b + 4 * m, Ins("LDS.128 R%d, [R0%s]" % (AF[(m + 1) & 1], "+0x%x" % off if off else ""),
                                   wsb=SB_ST, stall=3), False)
            iwait.setdefault((b + 4 * m + 4) % tot, []).append(SB_ST)
        # B's four gathers for the next k-step.
        for n in range(4):
            p, h = divmod(n, 2)
            g_ld = b + cfg["bl0"] + cfg["bls"] * n
            g_prm = b + 45 + 2 * p + h
            plans.append((g_ld, g_prm - 1, g_prm, nxt + 2 + h, 2048 * p, BT + 4 * p + h, BT + 4 * p + 2 + h, None, ()))
        # A' made: three gathers into MQ, two stores into the other buffer (stand-in places, inside the stage).
        for n in range(3):
            g_ld = b + cfg["mk0"] + cfg["mks"] * n
            g_prm = g_ld + lead + 1
            d = (MQ + (n & 1), MQ + 2 + (n & 1))
            plans.append((g_ld, g_prm - 1, g_prm, cur + 2 + (n & 1), 2048 * (2 + (n & 1)), d[0], d[1],
                          SB_RD if n == 2 else None, (SB_RD,) if n == 2 else ()))
            if n in (1, 2):
                put(g_prm + 1, Ins("STS.128 [R0+0x%x], R%d" % (512 * ((2 * s + n) % 6), MQ) if (2 * s + n) % 6 else
                                   "STS.128 [R0], R%d" % MQ,
                                   rsb=SB_RD), False)
    # Quads: the lowest free one; scoreboards 0-2 in issue order.
    order = sorted(range(len(plans)), key=lambda k: plans[k][0])
    assert len({plans[k][0] for k in order}) == len(order), "two LDSM in one gap"
    quads = R["quads"]
    free_at = [-tot] * len(quads)
    for walk in range(2):
        took = []
        for k in order:
            g_ld, g_prm = plans[k][0] + walk * tot, plans[k][2] + walk * tot
            qi = next((i for i in range(len(quads)) if free_at[i] <= g_ld), None)
            assert qi is not None, "more than 5 gathers under way in gap %d" % plans[k][0]
            free_at[qi] = g_prm + 1
            took.append(qi)
    for n, k in enumerate(order):
        g_ld, g_add, g_prm, areg, aoff, d1, d2, rsb, pw = plans[k]
        ld, adds, prm = gather_ins(quads[took[n]], areg, aoff, d1, d2, n % 3, cfg, rsb=rsb)
        put(g_ld, ld, wrapped=False)
        for x in adds:
            put(g_add, x, wrapped=False)
        for x in prm:
            x.wait = tuple(pw)
            put(g_prm, x, wrapped=False)
    # The readout: two chains of XOR3 (even and odd tiles), as rfold's.
    b3 = 3 * ni
    end = b3
    for ch in range(2):
        g = b3 + cfg["roff"]
        firstx = True
        for t in range(ch, nt, 2):
            for j in range(8):
                made, again = b3 + 4 * t + j // 2, tot + 4 * t + j // 2
                g = max(g, made + cfg["rmar"])
                assert g < again, "tile %d's sum is read in gap %d, behind its next IMMA" % (t, g)
                r = ACC + 4 * (4 * t + j // 2) + 2 * (j & 1)
                if firstx:
                    ins = Ins("LOP3.LUT R%d, R%d, R%d, RZ, 0x3c, !PT" % (R["xr"][ch], r, r + 1))
                    firstx = False
                else:
                    ins = Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (R["xr"][ch], R["xr"][ch], r, r + 1))
                put(g, ins)
                g += 1
        end = max(end, g)
    trw, w, xr = R["trw"], R["w"], R["xr"]
    put(end - 4, Ins("LDS R%d, [R%d]" % (w, trw), wsb=SB_RD), wrapped=True)
    put(end + 0, Ins("SHF.L.W.U32.HI R%d, R%d, 0xd, R%d" % (w, w, w), wait=(SB_RD,)), wrapped=True)
    put(end + 1, Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (w, w, xr[0], xr[1])), wrapped=True)
    put(end + 2, Ins("STS [R%d], R%d" % (trw, w), rsb=SB_RD), wrapped=True)
    put(end + 4, Ins(addi(w, trw, 4), wait=(SB_RD,)), wrapped=True)
    put(end + 5, Ins(andi(w, w, 0x3c)), wrapped=True)
    put(end + 6, Ins("LOP3.LUT R%d, R%d, 0x3f, R%d, 0xba, !PT" % (trw, trw, w), wait=(SB_RD,)), wrapped=True)
    put(b3 + 6, Ins(addi(R["cnt"], R["cnt"], -1)), wrapped=False)
    put(b3 + 7, Ins("ISETP.NE.AND P1, PT, R%d, 0x0, PT" % R["cnt"]), wrapped=False)
    put(b3 + 14, Ins(addi(R["poff"], R["poff"], 0x400), wait=(SB_RD,)), wrapped=False)
    return G, iwait


def emit(G, iwait, cfg):
    tot = 192
    out, most = [], 0
    for g in range(tot):
        mb, nb = divmod(g % 48, 4)
        items = [ins for ins, _w in G[g]]
        most = max(most, len(items))
        out.append(Ins("IMMA.16832.S8.S8 R%d, R%d.ROW, R%d.COL, R%d" % (ACC + 4 * (g % 48), AF[mb & 1], BT + 2 * nb,
                                                                       ACC + 4 * (g % 48)),
                       stall=cfg["pad"] if not items else cfg["ist"], yld=bool(cfg["iy"]), wait=iwait.get(g, ()),
                       label=".L0" if g == 0 else None))
        for k, x in enumerate(items):
            x = Ins(x.text, x.stall, bool(cfg["py"]), x.wsb, x.rsb, x.wait, raw=x.raw)
            if k == len(items) - 1:
                x.yld = bool(cfg["ly"])
            out.append(x)
    out.append(Ins("@P1 BRA .L0", stall=5))
    ep = []
    for g in range(tot):
        for x, wrapped in G[g]:
            if wrapped:
                ep.append(Ins(x.text, cfg["pst"], x.yld, x.wsb, x.rsb, x.wait, raw=x.raw))
    return out, ep, most


def build(cfg):
    gfold.UR = cfg["ur"]
    R = rfold.layout(cfg["mt"])
    q = R["quads"]
    pro = rfold.prologue(dict(cfg, band=0), R, [(q[2], 0, True), (q[3], 1, True), (q[4], 2, True)])
    bad = gfold.check(pro, "prologue", fix=True)
    G, iwait = chunk(cfg, R)
    loop, ep, most = emit(G, iwait, cfg)
    out, seen = [], 0
    for x in loop:
        if x.text.startswith("IMMA"):
            if seen and seen % 48 == 0 and not cfg["nbar"]:
                out.append(Ins("BAR.SYNC.DEFER_BLOCKING 0x0", stall=1))
            seen += 1
        if "BRA" in x.text and not cfg["nbar"]:
            out.append(Ins("BAR.SYNC.DEFER_BLOCKING 0x0", stall=1))
        out.append(x)
    body = [x for x in out if "BRA" not in x.text and "BAR" not in x.text]
    bad += gfold.check(pro + body + body + body, "loop")
    bad += gfold.check(body + ep, "epilogue", fix=True)
    ep.append(Ins(addi(R["tmp"], R["tmp"], 0), wait=(0, 1, 2, 3, SB_OFF, SB_RD)))
    ep.append(Ins("NOP", stall=8))
    return pro + out + ep, bad, most, G


def main():
    cubin, sass, nvd, out = sys.argv[1:5]
    cfg = dict(CFG)
    for kv in sys.argv[5:]:
        k, v = kv.split("=")
        assert k in cfg, k
        cfg[k] = int(v, 0)
    text = asm89.read_text(sass)
    code, bad, most, G = build(cfg)
    if bad:
        print("%d latency problems: not written" % bad)
        sys.exit(1)
    n, room = asm89.patch(cubin, text, out, code, nvd)
    kinds = {}
    for g in G:
        for ins, _w in g:
            op = ins.text.split(" ")[0].split(".")[0]
            kinds[op] = kinds.get(op, 0) + 1
    print("TIMING ONLY %s: %d instructions of %d, %d at most in a gap; a chunk: %s" % (
        os.path.basename(out), n, room, most, ", ".join("%s %d" % kv for kv in sorted(kinds.items(), key=lambda kv: -kv[1]))))


if __name__ == "__main__":
    main()
