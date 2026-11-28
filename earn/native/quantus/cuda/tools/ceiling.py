"""ceiling.py: the most nonces a second a kernel can do on a card, from its SASS counts and the card's issue rates.

  python ceiling.py --sms 128 --mhz 2610 --issue 4 --alu 2 --fma 2 --wide 1 --hi 1 \\
      --counts "alu=11643 fma=4087 fma.wide=7367 fma.hi=398 other=345"

Per SM and cycle a card issues at most --issue warp-instructions (4 schedulers), --alu of them on the ALU pipe
(IADD3, LOP3, SHF, ...), and --fma IMAD-class ones on the FMA pipe, where an IMAD.WIDE costs 1/--wide of a slot
and an IMAD.HI 1/--hi (ubench.cu measures all of these; 0.5 means it takes two). Counts are SASS instructions a
nonce (sasscount.py). A warp does 32 nonces at once, so an SM needs max(total/issue, alu/ALU, fma-slots/FMA)
cycles for 32 nonces. Prints the bound, which limit sets it, and the rate at the clock given.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sms", type=int, required=True)
    ap.add_argument("--mhz", type=float, required=True)
    ap.add_argument("--issue", type=float, default=4.0)
    ap.add_argument("--alu", type=float, required=True)
    ap.add_argument("--fma", type=float, required=True)
    ap.add_argument("--wide", type=float, default=1.0, help="IMAD.WIDE rate relative to IMAD")
    ap.add_argument("--hi", type=float, default=1.0, help="IMAD.HI rate relative to IMAD")
    ap.add_argument("--counts", required=True)
    ap.add_argument("--unified", action="store_true", help="ALU and FMA share one pool of --alu slots")
    a = ap.parse_args()
    c = {}
    for kv in a.counts.split():
        k, v = kv.split("=")
        c[k] = float(v)
    total = sum(c.values())
    fma_slots = c.get("fma", 0) + c.get("fma.wide", 0) / a.wide + c.get("fma.hi", 0) / a.hi
    alu = c.get("alu", 0)
    limits = {"issue": total / a.issue}
    if a.unified:
        limits["int"] = (alu + fma_slots) / a.alu
    else:
        limits["alu"] = alu / a.alu
        limits["fma"] = fma_slots / a.fma
    which = max(limits, key=limits.get)
    cyc = limits[which]  # SM cycles for 32 nonces
    rate = a.sms * 32 / cyc * a.mhz * 1e6
    print("total %.0f, alu %.0f, fma slots %.0f; SM cycles / 32 nonces: %s; bound by %s; %.1f MH/s at %.0f MHz"
          % (total, alu, fma_slots, ", ".join("%s %.0f" % kv for kv in limits.items()), which, rate / 1e6, a.mhz))


if __name__ == "__main__":
    main()
