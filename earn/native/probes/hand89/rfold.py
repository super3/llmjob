"""rfold.py: the rotation fold's chunk loop written by hand for sm_89. It replaces the loop function ptxas compiled
from gfold_h.cuh (built with -DGP_STAGE=24576) and computes the same transcripts from gprobe's `rot` offsets:
gprobe's CPU check judges it.

  python rfold.py <base.cubin> <base.sass> <nvdisasm> <out.cubin> [key=value ...]

Why this form. Measured on rig01's RTX 4090 (probes/README.md, "The price of a step, and the rotation fold"):
under the card's cap the clock falls with the count of shared-memory steps a k-step (LDSM, LDS, STS: about one
unit each, 16 MHz) and hardly with ALU steps (a tenth of a unit). The gather fold has 20 LDSM and 80 ALU steps a
warp a k-step because every warp makes all of its fragments itself. But the four warps of a row band fold the
same rows: an A fragment (an m16 tile's four words of a lane) is the same in all four.

What the loop does. Every warp takes the four k-steps of a chunk in an order of its own (any order inside a chunk
gives the same sums), so that at the chunk's first turn the four warps of a band fold four different k-steps.
At that turn a warp makes its k-step's six A fragments with gathers, as the gather fold does, uses them, and
stores each with one 16-byte store. At its other three turns it folds k-steps another warp of its band made at
the first turn, and an A fragment is one 16-byte load from the stage. B's fragments are gathered by every warp
at every turn, as in the gather fold. A warp a k-step: 11 LDSM, 4.5 LDS, 1.5 STS and 44 ALU steps for the draw
(the gather fold: 20 LDSM and 80 ALU steps).

What keeps a load behind its store (sync=4, band=1, the defaults). The four warps of a row band meet at a
barrier of their own at every k-step's end (bar.sync 1, 128 for band 0 and 2, 128 for band 1; the bands do not
wait for each other). Tiles 1 to 5 are stored in the first turn and loaded in turns 1 to 3. Tile 0, which a
turn loads in the turn before, is stored in the chunk's last turn (the first chunk's in the prologue, ahead of
a barrier). So every load has a barrier between it and the store that wrote its fragment, and another between
it and the store that overwrites it. Without barriers the loop is faster and wrong: nothing then orders a load
after its store (12 to 54 of 64 tiles failed the check).

Two things the barriers made necessary, each measured. One barrier for all eight warps costs 3 points of pipe
more than one a band. And warps a barrier has put in step all gather in the same gaps: with bank collisions
that is more than shared memory serves, so the first five of B's eight LDSM go out early and two gaps apart
(bsp=1: 96.0% of the pipe against 93.8%).

Where the loop's time went, and two things that got some back (goal-nvk10, probes/README.md, "Where the
rotation fold's time goes"). With every gather's LDSM turned into a move (nold, timing only) the pipe holds 99.1%
against 97.5%: the loop loses most of its pipe waiting for gathers. A gather's IADD3 waits for its LDSM, and
when the load is slow (bank collisions, eight warps at shared memory at once) the warp stands still although
nothing needs the fragment for several gaps. So the IADD3 and PRMT now go as late as their readers allow (al,
bl, bp: 98.1% of the pipe). And an offsets record is 8 bytes, not 16 (co): the record comes from the L2 every
k-step, and that trip costs clock under the card's cap. Two offsets fit a word; the loop parts them with two
ALU steps a turn. Run gprobe with rot 2 for this form (rot 1 writes the 16-byte records, for co=0). Last, A's
first-turn gathers go out early, every gather sets a scoreboard on which nobody waits for another gather's
load, and the sixth B gather of a turn has a quad of its own (ae, sbo, cq: 99.0%; each alone gains nothing).

Registers: as gfold.py's, but R0 is the lane's place in the warp's own part of the stage and the transcript
word is worked in the scratch register. P2 is the warp's row band.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm89  # noqa: E402
from asm89 import Ins  # noqa: E402
import fold  # noqa: E402
from fold import addi, andi, muladd, shr  # noqa: E402
import gfold  # noqa: E402
from gfold import ACC, AF, BF, SB_OFF, SB_RD, XR, gather_ins, ldg  # noqa: E402

CFG = dict(mt=6, pad=8, ist=5, py=0, iy=1, ly=1, ru=0, lead=3, sf=0, hold=1, pst=2, rmar=5, roff=11, pol=2, ur=4,
           # sync=4: a barrier at every k-step's end, and tile 0's store in the turn before its first use: every
           # load of a stored fragment then has a barrier between it and the store before and the store after.
           # sync=0: only a barrier every barn chunks (a power of two; 0 for none), tile 0's store in gap st0:
           # correct only while the warps of a band stay within 39 IMMA of each other (they do not).
           sync=4, barn=16, st0=1,
           # band=1: the k-step barriers are one for each row band (128 threads each): only the four warps
           # that share fragments wait for each other.
           band=1, bsp=1,
           # lead: the gaps between a gather's LDSM and its IADD3; plead: between its IADD3 and its PRMT. Three and
           # one hold 0.5 points more of the pipe than two and two; py=0 (the plain steps' yield flag off) 0.2.
           plead=1,
           # Timing only (wrong results on purpose): nost, no stores; nold, the gathers' LDSM are moves (1 all
           # of them, 2 A's only, 3 B's only).
           nost=0, nold=0,
           # bsh=1, TIMING ONLY (wrong results on purpose): B shared by the two warps of a column slot. The
           # warps of row band 0 gather B every turn, as now, and store each pair of fragments (4 STS.128 a
           # turn). The warps of band 1 gather no B: 4 LDS.128 a turn. Band 1 also loads an offsets record
           # only once a chunk (its first turn's, for A's gathers), and every warp keeps the stage address of
           # the turn in a register it moves on each turn (two ALU steps), where the record had it. One
           # barrier for all eight warps at every k-step's end. The addresses are stand-ins inside the
           # stage: this prices the steps and the barrier, it does not compute the fold.
           bsh=0,
           # bfa=1 (with bsh, TIMING ONLY): the warp that loads B keeps two sets of B's sixteen registers and
           # reads them turn about, so the next turn's four loads go out early in the turn (gaps 20 to 32)
           # and nothing waits for them. It has the registers: it gathers no B, so four of its five quads
           # are free, and it loads one offsets record a chunk, so one of the two record quads is.
           bfa=0,
           # Where a gather's IADD3 and PRMT go. The IADD3 waits for its LDSM: when the load is slow (bank
           # collisions, eight warps at the shared memory at once) the warp stands still until it lands, though
           # nothing may need the fragment for several gaps yet. With the gathers' LDSM as moves (nold, timing
           # only) the pipe holds 99.09% against 97.48%: most of what the loop loses is these waits. These keys
           # move the waits later and change no result.
           #   al: A's gathers of the first turn do their IADD3 and PRMT a gap later (0 or 1: the fragment
           #       is first read 8 gaps behind its LDSM, and a PRMT has to be two gaps ahead of its reader).
           #       Not the last tile's: its quads are wanted by B's early gathers.
           #   bl: the B gathers whose LDSM go out early (bsp) do their IADD3 a gap ahead of their PRMT,
           #       not `lead` gaps behind the LDSM.
           #   bp: the last three B gathers of a turn do their PRMT as late as the fragments' first readers
           #       allow (two gaps ahead of them: gaps 50 to 52 of 48), and their IADD3 a gap ahead of it.
           #   cq: a sixth quad for the gathers. The turn's own offsets record is done with after its last
           #       stage load (gap 33; in the first turn, after A's last gather), and its four registers are
           #       not loaded again until gap 2 of the next turn. The sixth B gather of a turn (the first of
           #       the three that had to wait for a quad) takes them: its LDSM goes out in gap `cq` (35 or
           #       36), nine gaps or more ahead of its IADD3, and its PRMT is in gap 48 (the next turn's 0).
           #   ae: A's gathers of the first turn go out early. With al, bl and bp what the loop still waits
           #       for is these (their LDSM as moves, timing only: 99.12% of the pipe against 98.11%): all four
           #       warps of a band gather A in the same gaps of the same turn. With ae=1 the two LDSM of
           #       tiles 3 to 5 go out six gaps earlier, during the tile before last, and tile 2's three gaps
           #       earlier (the quads of the turn before are busy until then); their IADD3 and PRMT stay where
           #       they are. Tile 1's cannot move: the turn before has no quad for them.
           #   sbo: which scoreboard a gather's LDSM sets. A scoreboard counts the loads that set it and are
           #       still out, and an IADD3 that waits on it waits for all of them: for its own gather's load,
           #       and for any later gather's that was given the same scoreboard and went out since. There
           #       are three scoreboards for up to six gathers under way, handed out in turn (sbo=0). With
           #       sbo=W a gather takes the scoreboard on which the fewest IADD3 would wait for a load of
           #       another gather that went out less than W gaps before them (a search; W 5).
           #   bt: which turns of a chunk end at a barrier (sync=4), a bit a turn; 15 is all four. The
           #       barrier after the second turn (bit 1) orders nothing: no fragment is stored in the second
           #       or the third turn. 13 leaves it out and every load still has a barrier between it and the
           #       stores before and after. 9 and 8 leave out more and then rely on the warps of a band
           #       staying within some 43 gaps of each other for a few turns: not proven, so not a default.
           al=1, bl=1, bp=1, cq=35, ae=1, sbo=7, bt=15,
           # co=1: offsets records of 8 bytes, not 16 (gprobe's rot 2; in the miner, rot_aux_offs). A record
           # comes from the L2, 512 bytes a warp a k-step, and that trip is 2.9 units of the clock's cap. Every
           # offset is under 65536, so two fit a word: a first turn's record is {A's two, B's two}, a later
           # turn's {the stage address, B's two}. One LDG.E.64 a turn, and two ALU steps to part B's two
           # (four in the chunk's last turn, for the next chunk's A).
           co=1,
           # nlg=1, TIMING ONLY: three of a chunk's four offsets loads read the same record (the chunk's
           # second turn's), so two of the four find their bytes in the SM's cache and not in the L2. What
           # the offsets' trip from the L2 costs. (With co=0 it was three of four: all four read the next
           # chunk's first record; out3 to out10 of the tenth round are that form.)
           nlg=0,
           # u8=1: the IMMA read their operands as unsigned bytes (U8.U8). The operands are 0 to 126, so the sums are
           # the same; what it may change is the tensor datapath's power, which holds the clock at 450 W.
           u8=0,
           # fa, fb (goal-nvk17): the byte A's and B's gathers add (0 keeps 0x3f, the fill). ty: the IMMA's operand
           # types, 1 S8.S8, 2 S8.U8, 3 U8.S8, 4 U8.U8 (0 follows u8). A fill of 0xc0 read as S8 is operands of
           # -64 plus the noise, -127 to -1 (correct with the matrix filled with -64); 0x40 is a fill of 64. A fill
           # under 0x3f borrows across bytes: TIMING ONLY (0x100, a fill of 0, gives operands near the noise itself, signed).
           fa=0, fb=0, ty=0,
           # xa, xb (goal-nvk17): XOR every operand byte of A's / B's gathers with this byte after the subtract (two
           # LOP3 a gather, a gap behind it). xa=0x80 with fa=0x80 and ty=2 is A as the plain signed noise (a fill of
           # 0), CORRECT with A filled with 0; 0x100 is an XOR of 0 (the same steps, operands unchanged).
           xa=0, xb=0,
           # nord=1, TIMING ONLY: no readout XORs (the transcripts are then wrong). What the 96 LOP3 a chunk cost.
           nord=0,
           # bsym=1, TIMING ONLY: B shared by the two warps of a column slot, both making and both loading in turn
           # (the eighth round's proposal, never priced). Band 1 already takes the k-steps of a chunk in the order
           # (j + (i ^ 1)) & 3, so each k-step's B is wanted first by one partner and a turn later by the other:
           # at the ends of turns 3 and 1 a warp gathers the next turn's B and stores it (4 STS.128), at the
           # ends of turns 0 and 2 it loads its partner's (4 LDS.128). Only the band barriers: nothing orders a
           # load after its partner's store, and the addresses are stand-ins in the warp's own stage.
           # bsym=2: the same, but a load turn's four loads go out early into the gather quads (free then: no B
           # gathers at the end of a load turn) and four moves a pair put them into B's registers where the
           # gathers' PRMT would be; and a make turn's store goes a gap behind the pair's last PRMT.
           # bld: the gap of a turn where the first of those loads goes out (the others follow, one a gap).
           # bsym=3: as 1 (loads straight into B's registers, a gap behind the pair's last reader), but on
           # scoreboard 5, not 3: on 3 the next turn's IMMA that wait for a pair also wait for the stage load
           # of A's tile 1, which goes out in the turn's gap 1. Stores as 2.
           # bnst / bnld (with bsym, TIMING ONLY): leave out B's stores / B's loads, to see which costs the pipe.
           # bsym=4: as 3, but pair p's load sets scoreboard (0, 1, 2, 5)[p] (no gather is under way then), so
           # an IMMA waits only for its own pair; and bsg >= 0 puts pair p's store in gap bsg + 3 p of the next
           # turn (the registers hold the pair until that turn's end), out of the busy last gaps of a turn.
           bsym=0, bld=34, bnst=0, bnld=0, bsg=-1,
           # bord=1 (with bsym=4 and bsg >= 0; goal-nvk13): what orders a partner's load after the store, and the
           # next store after that load. The two warps of a column slot (bands 0 and 1, the same scheduler) share
           # barrier 4 + slot (64 threads), and BOTH sync on it: behind a make turn's stores (gap bsg + 10 of the
           # next turn, ahead of the loads at its end), and in the middle of every make turn (gap bmid: behind
           # the loads at the end of the turn before, ahead of the next stores). The chunk count's register holds
           # 16 x the chunks left + the number. (An arrive by one warp and a sync by the other, taken in turn on
           # one barrier, deadlocks on the card: a warp's own arrive and its own later sync make up the 64.)
           # bord=2: the same orders with no warp waiting for its partner where nothing needs it. Four barriers of
           # 256 threads, each arrived on by one row band and synced on by the other: 4 + band "this band's pairs
           # are stored", 6 + band "the other band has loaded this band's pairs". A make turn: sync 6 + band,
           # stores, arrive 4 + band; the loads at the end of the next turn: sync 5 - band, loads, arrive 7 - band.
           # The register holds 16 x the chunks left + the number in use (moved by -2, ^1, +2, ^1). The prologue
           # arrives once on 7 - band; the epilogue (whose wrapped stores arrive on 4 + band) syncs once on
           # 5 - band. (Checked on the card with nv-study/k13/pp2.cu: 0 stale reads in 12,000 rounds a block.)
           # barv=1 (with bord=2): the arrives take their number as an immediate under the row band's predicate
           # (two instructions, one issued idle), the syncs from the register; the register then moves by XOR 3.
           # In case bar.arrive reads more of the register than bar.sync's low four bits.
           barv=0,
           # bst: the stall of the bord=2 barrier instructions (ptxas gives every BAR 6).
           bst=1,
           bord=0, bmid=24,
           # bx=1 (with bsym=4, bsg, bord; goal-nvk13): B drawn once, CORRECT. Needs gprobe/pearl_rot built with
           # -DGP_BX=1: the thread's transcript is then in the kernel's frame (local memory at R1 + 0, word w at
           # R1 + 4 w), and the 16 KB where the transcripts were hold the exchange, [slot][band][pair][lane] of
           # 16 bytes. A warp's own area and its partner's differ only in bit 11, so R3 (the readout's second
           # chain, free from the transcript's tail to the next readout) holds the own area's address, reloaded
           # each chunk from R1 + 0x40 where the prologue left it, and an XOR of 0x800 turns it into the
           # partner's around the loads.
           bx=0,
           # bxt=1 (with bx and bord=2): the exchange address lives in the transcript pointer's register (R251) for
           # the whole loop, so the first turn's stores need not wait for the readout's second chain to end. The
           # transcript's word is found from the chunk count instead: in the tail (which runs in the next chunk's
           # first turn, after the count went down) -(cnt >> 2) & 0x3c is 4 x (chunk & 15), because the count holds
           # 16 x the chunks left + a barrier number of 4 to 7 and the chunks are a multiple of 16.
           bxt=0,
           # bxs: the scoreboard the last exchange store and load release once they have read the address register,
           # and that the XORs moving it wait on. 4 (the records' load) made the turn-end XOR wait for a record
           # from the L2; 3 (the stage loads) is idle there.
           bxs=3,
           # bnsy=1, TIMING ONLY: the bord=2 syncs left out (the arrives stay): what the waits cost.
           bnsy=0,
           # bsg3, bss (with bord): the stores of the chunk's last make turn (they go out in the next chunk's first
           # turn) start at gap bsg3 when it is 0 or more (bx needs about 32: the transcript's tail holds R3 until
           # gap 31), and a make turn's four stores are bss gaps apart.
           bsg3=-1, bss=3,
           # brg=1: one barrier instruction, not the predicated pair (@!P2 bar 1, @P2 bar 2): the barrier's number
           # comes from the chunk count's register, which then holds 16 x the chunks left + 1 + the row band. The
           # card reads a barrier number from a register's low four bits, so the count steps by 16 and the loop
           # runs while it is over 15. One instruction fewer a barrier, and the same barriers.
           brg=0,
           # noprm=1, TIMING ONLY (wrong results on purpose): the gathers' two PRMT left out; the two IADD3 write
           # the fragment words where the PRMT would have. What the 22 PRMT a warp a k-step cost.
           noprm=0,
           # snk=1, TIMING ONLY: B read in reverse in every other tile (gfold.emit), so that across a tile's end
           # only A changes; the sums are not moved with it. What a snake order would save in operand toggling.
           snk=0,
           # ldk, TIMING ONLY (goal-nvk14): gathers whose LDSM.T becomes an LDS.128 at the same addresses (1 B's, 2 A's,
           # 3 all). The same bytes from the same banks and the same instruction count; the values stay spread as
           # operands are. What LDSM's transpose costs per byte against a plain 16-byte load, under the real load.
           ldk=0,
           # rsp (goal-nvk14): the readout's 96 XOR3 spread over rsp gaps of the window (a sum is read between its
           # last IMMA of the chunk and its first of the next), not two a gap from roff. 0 keeps two a gap.
           rsp=0,
           # rdx=1, TIMING ONLY (goal-nvk14): the readout's steps read no sums. Is its pipe cost the reads of the
           # accumulators (register banks) or the steps themselves?
           rdx=0,
           # rpr (goal-nvk14, correct): which two sums a readout step reads; see the readout.
           rpr=0,
           # prr=1 (goal-nvk15): a gather's first PRMT marks both its sources for the reuse latch, so the pair's
           # second PRMT (same sources) reads neither from the register file (gfold.emit). Register-file energy.
           prr=0,
           # rgf=1 (goal-nvk16, correct): the readout's steps go first in their gap, right behind the IMMA's stall, not
           # last, right ahead of the next IMMA. 99.39% of the pipe against 99.18%.
           rgf=0,
           # rb0, rb1 (goal-nvk16, correct): the readout keeps out of the chunk's last rb0 gaps and the next chunk's first
           # rb1. 9 and 3 were chosen when the steps went last in a gap. With rgf=1, rb0=3 rb1=0 holds 99.54% of the
           # pipe, the level of the loop with no readout at all (nord, 99.55%); rb0=0 loses (99.24%).
           rb0=9, rb1=3,
           # rst (goal-nvk16): the stall of each readout step (0 keeps the default, 1). rst=2 gains nothing.
           rst=0,
           # lgg (goal-nv50c): the gap of a turn where the next turn's offsets record is loaded (an L2 trip whose
           # latency in cycles grows with the clock; nlg=1, timing only, is +1.1 points of pipe on an RTX 5090).
           # Its readers are 22 gaps on at 2; 0 or 1 sends it out earlier, still behind the last readers of the
           # set it overwrites (gaps 44-46 of the turn before).
           lgg=2,
           # b8=1 (goal-nv50d, sm_120 only, CORRECT with tables in plain row order): every gather is one
           # ldmatrix.m16n16.x2.trans.b8 (LDSM.8.MT1616.2), whose byte transpose hands each lane rows g and g + 8 in
           # registers of their own, so the two subtracts write the fragment words and the two PRMT go (noprm's
           # +1.2 points of pipe, now with right sums). The k each lane's four bytes hold change, the same way for A
           # and B, so the sums do not. The tables' entry p holds row c of the group at rank index p in byte c
           # (pearl_rot.cu ROT_TAB8, gprobe rot 3), not rows e and 8 + e in bytes 2e and 2e + 1.
           b8=0,
           # acz, bcz (goal-nv50e, TIMING ONLY, with b8): A's / B's gathers load as before but their fragment words
           # come out as the fill, so that side's operands never change (gfold.gather_ins). Against nold=2 / nold=3
           # it parts a side's draw at the power cap into the loads' cost and the tensor datapath's toggling.
           acz=0, bcz=0)
SB_ST = 3                   # the stage loads' scoreboard; the gathers' LDSM go round 0-2
# bsp: the gaps of a turn's eight B gathers' LDSM, in the turns 0 to 2 and in the chunk's last turn (where
# the next chunk's first tile is gathered in gaps 37 and 38). Their PRMT stay in gaps 43 to 50.
BLD = ([28, 30, 34, 37, 39, 44, 45, 46], [28, 30, 34, 42, 43, 44, 45, 46])
SW = 0                      # R0: where the lane's fragments of the warp's first turn go
SBO = [0, 0]                # sbo: the waits for another gather's load, counted before the search and after


def layout(mt):
    R = gfold.layout(mt, 0)
    R["w"] = R["tmp"]
    R["stg"] = R["tr"] + 256 * 64 + 256            # GP_STG
    return R


def prologue(cfg, R, tail):
    """As gfold's, with the stage address. `tail`: the quad and scoreboard the loop gave the chunk's last three
    gathers, whose ends lie in the next chunk's first gaps."""
    mt = cfg["mt"]
    q = R["quads"]
    S = q[0]
    c = [Ins("S2R R0, SR_TID.X", wsb=SB_RD),
         Ins("ULDC.64 UR%d, c[0x0][0x118]" % gfold.UR, raw=(0x0000460000007ab9 | (gfold.UR << 16), 0xa00))]
    for r in list(range(AF[0], ACC + 32 * mt, 2)) + [XR[0]]:
        c.append(Ins("CS2R R%d, SRZ" % r))
    c.append(Ins(andi(S, 0, 0x1f), wait=(SB_RD,)))                            # lane
    c.append(Ins(shr(S + 1, 0, 5), wait=(SB_RD,)))                            # warp
    c.append(Ins(muladd(R["trw"], 0, 0x40), wait=(SB_RD,)))                   # the thread's transcript ...
    c.append(Ins("IMAD.MOV.U32 R%d, RZ, RZ, c[0x0][0x178]" % R["cnt"]))       # chunks
    rec = 0x100 if cfg["co"] else 0x200                                       # a k-step's record, 32 lanes
    c.append(Ins(muladd(S + 2, R["cnt"], 4 * rec)))                           # a warp's offsets: 4 chunks + 1
    c.append(Ins(addi(S + 2, S + 2, rec)))                                    # k-steps
    c.append(Ins("IMAD.WIDE.U32 R%d, R%d, R%d, c[0x0][0x170]" % (R["poff"], S + 1, S + 2)))
    c.append(Ins(shr(S + 3, S + 1, 2)))                                       # row band
    if cfg["band"] or cfg["bsh"]:
        c.append(Ins("ISETP.NE.AND P2, PT, R%d, 0x0, PT" % (S + 3)))
    if cfg["brg"]:
        c.append(Ins(muladd(R["cnt"], R["cnt"], 0x10, S + 3)))               # 16 chunks + band ...
        c.append(Ins(addi(R["cnt"], R["cnt"], 1)))                            # ... + 1: the barrier's number
    elif cfg["bord"]:
        assert cfg["bsym"] == 4 and cfg["bsg"] >= 0 and cfg["band"] and cfg["sync"] == 4
        c.append(Ins(andi(q[3], S + 1, 3)))                                   # column slot j
        if cfg["bord"] == 2:
            c.append(Ins(muladd(R["cnt"], R["cnt"], 0x10, S + 3)))            # 16 chunks + band ...
            c.append(Ins(addi(R["cnt"], R["cnt"], 6)))                        # ... + 6: 6 + band, then XOR 1:
            c.append(Ins("LOP3.LUT R%d, R%d, 0x1, RZ, 0x3c, !PT" % (R["cnt"], R["cnt"])))
            c.append(Ins("NOP", stall=6))
            if cfg["barv"]:
                c.append(Ins("@!P2 BAR.ARV 0x7, 0x100", stall=cfg["bst"]))
                c.append(Ins("@P2 BAR.ARV 0x6, 0x100", stall=cfg["bst"]))
            else:
                c.append(Ins("BAR.ARV R%d, 0x100" % R["cnt"], stall=cfg["bst"]))       # 6 + (1 - band): "loaded"
            c.append(Ins("LOP3.LUT R%d, R%d, 0x1, RZ, 0x3c, !PT" % (R["cnt"], R["cnt"])))   # 6 + band
        else:
            c.append(Ins(muladd(R["cnt"], R["cnt"], 0x10, q[3])))             # 16 chunks + j ...
            c.append(Ins(addi(R["cnt"], R["cnt"], 4)))                        # ... + 4: the slot's barrier
        if cfg["bx"]:
            c.append(Ins(muladd(q[3], q[3], 2, S + 3)))                       # 2 j + band
            c.append(Ins(muladd(q[3], q[3], 0x800)))
            c.append(Ins(muladd(q[3], S, 0x10, q[3])))                        # + 16 lane
            c.append(Ins(addi(q[3], q[3], R["tr"])))                          # the warp's own exchange area
            c.append(Ins("NOP", stall=6))
            if not cfg["bxt"]:
                c.append(Ins("STL [R1+0x40], R%d" % q[3]))
    c.append(Ins("IADD3 R%d, R%d, 0x0, R%d" % (q[1], S + 1, S + 3)))
    c.append(Ins(andi(q[1], q[1], 3)))                                        # the k-step of the first turn
    c.append(Ins(muladd(q[1], q[1], 512 * mt)))
    c.append(Ins(muladd(S + 3, S + 3, 2048 * mt, q[1])))
    if cfg["co"]:
        c.append(Ins(muladd(q[2], S, 0x8)))                                   # 8 lane: its place in a record
    c.append(Ins(muladd(S, S, 0x10)))                                         # 16 lane
    if cfg["bx"] and cfg["bxt"]:
        c.append(Ins(fold.mov(R["trw"], q[3])))                               # the own exchange area, for good
    elif cfg["bx"]:
        c.append(Ins(addi(R["trw"], 1, 60)))                                  # the frame's transcript, word 15
    else:
        c.append(Ins(addi(R["trw"], R["trw"], R["tr"] + 60)))                 # ... at its word 15
    c.append(Ins("NOP", stall=4))
    c.append(Ins("IADD3 R%d, R%d, 0x0, R%d" % (R["poff"], R["poff"], q[2] if cfg["co"] else S)))  # no carry: the host's care
    c.append(Ins("IADD3 R%d, R%d, 0x%x, R%d" % (SW, S + 3, R["stg"], S)))
    cur = R["oq"][0]
    c.append(ldg(cur, R["poff"], 0, size=64 if cfg["co"] else 128, wsb=SB_OFF))
    c.append(Ins("NOP", stall=8))
    wl = (SB_OFF,)
    if cfg["co"]:
        c.append(Ins("LOP3.LUT R%d, R%d, 0xffff, RZ, 0xc0, !PT" % (cur + 2, cur + 1), wait=wl))
        c.append(Ins(shr(cur + 3, cur + 1, 16), wait=wl))
        c.append(Ins(shr(cur + 1, cur, 16), wait=wl))
        c.append(Ins("LOP3.LUT R%d, R%d, 0xffff, RZ, 0xc0, !PT" % (cur, cur), wait=wl))
        c.append(Ins("NOP", stall=8))
        wl = ()
    jobs = [(cur + h, 0, AF[0] + 2 * h, AF[0] + 2 * h + 1) for h in range(2)]
    jobs += [(cur + 2 + h, 2048 * p, BF + 4 * p + h, BF + 4 * p + 2 + h) for p in range(4) for h in range(2)]
    tq = [t[0] for t in tail]
    free = [x for x in q if x not in tq]
    for part, quads in ((jobs[0:5], q), (jobs[5:7], free)):
        part = [gather_ins(quads[n], a, o, d1, d2, n % 4, cfg, wait_ld=wl) for n, (a, o, d1, d2) in enumerate(part)]
        c += [g[0] for g in part]
        for g in part:
            c += g[1]
        for g in part:
            c += g[2]
    if cfg["sync"] == 4 and not cfg["nost"]:
        # The first chunk's tile 0 goes to the stage here, and the warps meet before any of them loads.
        c.append(Ins("STS.128 [R%d], R%d" % (SW, AF[0])))
        c.append(Ins("NOP", stall=8))
        c += barrier(cfg)
    last = [gather_ins(tail[n][0], a, o, d1, d2, tail[n][1], cfg, wait_ld=wl) for n, (a, o, d1, d2) in enumerate(jobs[7:])]
    c += [g[0] for g in last]
    for n, g in enumerate(last):
        if tail[n][2]:                  # its IADD3 lie in the chunk's last gaps: the loop does not redo them
            c += g[1]
    return c


def chunk(cfg, R, role=None):
    """Gaps 0 .. 32 mt - 1 of a chunk, each a list of (Ins, wrapped), the IMMA that wait, and the quad and
    scoreboard of the chunk's last three gathers. `role` (bsh): "make" for a warp that gathers B and stores
    it, "load" for one that loads its partner's."""
    mt = cfg["mt"]
    assert mt % 2 == 0
    ni, tot = 8 * mt, 32 * mt
    G = [[] for _ in range(tot)]
    iwait = {}

    def put(g, ins, wrapped=None, front=False):
        item = (ins, g >= tot if wrapped is None else wrapped)
        if front:
            G[g % tot].insert(0, item)
        else:
            G[g % tot].append(item)

    plans = []          # (LDSM gap, IADD3 gap, PRMT gap, address register, offset, d1, d2, rsb)
    quads, bfalt = R["quads"], None
    last3 = {}          # n (5, 6, 7) -> the plan of the last turn's B gather n
    ldsplans = set()    # bsym=2: plans that are a 16-byte load of the partner's pair and four moves
    if role == "load" and cfg["bfa"]:
        quads, bfalt = [R["quads"][4], R["oq"][1]], R["quads"][0]
    lead, plead = cfg["lead"], cfg["plead"]
    xr_ = R["trw"] if cfg["bxt"] else XR[1]          # bx: the exchange address register
    hold = SB_RD if cfg["hold"] else None
    hw = (SB_RD,) if cfg["hold"] else ()
    for s in range(4):
        b = ni * s
        cur, nxt = R["oq"][s & 1], R["oq"][(s + 1) & 1]
        # The next turn's record. The load overwrites the set the turn before last read: it waits for that
        # turn's last readers (hold). In the chunk's last turn it is the last user of the pointer.
        if role != "load" or s == 3:
            rec = 0x100 if cfg["co"] else 0x200
            # nlg (timing only): the chunk's second-turn record three times, then the next chunk's first: a
            # later turn's record has another form than a first turn's (co), so it has to be one of its kind.
            put(b + cfg["lgg"], ldg(nxt, R["poff"], rec * (1 if s < 3 else 4) if cfg["nlg"] else rec * (s + 1),
                           size=64 if cfg["co"] else 128,
                           wsb=SB_OFF, rsb=hold if s == 3 else None, wait=hw))
            # The record's first readers issue 8 cycles or more behind three IMMA that wait for the load.
            first = b + (min(BLD[0]) if cfg["bsp"] else ni - 11) - 2
            if cfg["co"]:
                # The record's two words are parted two gaps earlier, behind the same three IMMA: B's two
                # offsets into the quad's upper pair, then (for a first turn's record) A's two in place.
                assert cfg["bsp"] and role is None
                first -= 2
                put(first, Ins("LOP3.LUT R%d, R%d, 0xffff, RZ, 0xc0, !PT" % (nxt + 2, nxt + 1)), False)
                put(first, Ins(shr(nxt + 3, nxt + 1, 16)), False)
                if s == 3:
                    put(first + 1, Ins(shr(nxt + 1, nxt, 16)), False)
                    put(first + 2, Ins("LOP3.LUT R%d, R%d, 0xffff, RZ, 0xc0, !PT" % (nxt, nxt)), False)
            for g in (first - 2, first - 1, first):
                iwait.setdefault(g, []).append(SB_OFF)
        # bsh: the stage address of a turn is in R0, moved on a turn at each turn's end (behind the turn's
        # last use of it, ahead of the load of the next turn's first tile).
        stage_cur = SW if role else cur
        stage_nxt = SW if role else nxt
        if role:
            put(b + ni - 9, Ins(addi(R["w"], SW, 0)), False)
            put(b + ni - 8, Ins("LOP3.LUT R%d, R%d, 0x3fff, R%d, 0xe2, !PT" % (SW, R["w"], SW)), False)
        for m in range(1, mt):                         # tile m's fragment, during tile m - 1
            a = AF[m & 1]
            if s == 0:                                 # the first turn: made here
                for h in range(2):
                    g0 = b + 8 * (m - 1) + h
                    late = cfg["al"] if m < mt - 1 else 0
                    assert 0 <= late and 1 + lead + plead + late <= 6, "al: the fragment is read 8 gaps on"
                    early = (0, 0, 3, 6, 6, 6)[m] if cfg["ae"] else 0
                    plans.append((g0 - early, g0 + lead + late, g0 + lead + plead + late, cur + h, 2048 * m,
                                  a + 2 * h, a + 2 * h + 1, hold if m == mt - 1 else None))
            else:                                      # a later turn: another warp's, from the stage
                put(b + 8 * (m - 1) + 1, Ins("LDS.128 R%d, [R%d+0x%x]" % (a, stage_cur, 512 * m), wsb=SB_ST,
                                             rsb=hold if m == mt - 1 else None), False)
                for g in (b + 8 * m, b + 8 * m + 1):
                    iwait.setdefault(g % tot, []).append(SB_ST)
        a = AF[0]                                      # the next turn's first tile
        if s == 3:                                     # the next chunk's first turn: made here
            for h in range(2):
                g0 = b + ni - 11 + h
                plans.append((g0, g0 + lead, g0 + lead + plead, nxt + h, 0, a + 2 * h, a + 2 * h + 1, None))
        else:
            put(b + ni - 7, Ins("LDS.128 R%d, [R%d]" % (a, stage_nxt), wsb=SB_ST), False)
            for g in (b + ni, b + ni + 1):
                iwait.setdefault(g % tot, []).append(SB_ST)
        brole = ("load" if s % 2 == 0 else "make") if cfg["bsym"] else role
        if cfg["bord"] and brole == "make":
            # the stores go out in gaps bsg + 3 p of the next turn; the loads at the end of that turn
            g0 = b + ni + (cfg["bsg3"] if s == 3 and cfg["bsg3"] >= 0 else cfg["bsg"])
            sp = cfg["bss"]
            assert g0 + 3 * sp + 2 < b + 2 * ni - 6 and cfg["bmid"] < ni
            cn = R["cnt"]
            if cfg["bord"] == 2:
                L = b + 2 * ni - 6                       # the first load, at the end of the next turn
                lw = False if s == 3 else None           # the loads belong to the chunk they are in
                Lg = (lambda g: g % tot) if s == 3 else (lambda g: g)
                if not cfg["bnsy"]:
                    put(g0 - 1, Ins("BAR.SYNC.DEFER_BLOCKING R%d, 0x100" % cn, stall=cfg["bst"]))
                if cfg["barv"]:
                    put(g0 + 3 * sp + 1, Ins("@!P2 BAR.ARV 0x4, 0x100", stall=cfg["bst"]))
                    put(g0 + 3 * sp + 1, Ins("@P2 BAR.ARV 0x5, 0x100", stall=cfg["bst"]))
                    put(g0 + 3 * sp + 2, Ins("LOP3.LUT R%d, R%d, 0x3, RZ, 0x3c, !PT" % (cn, cn)))
                    if not cfg["bnsy"]:
                        put(Lg(L - 1), Ins("BAR.SYNC.DEFER_BLOCKING R%d, 0x100" % cn, stall=cfg["bst"]), lw)
                    put(Lg(L + 7), Ins("@!P2 BAR.ARV 0x7, 0x100", stall=cfg["bst"]), lw)
                    put(Lg(L + 7), Ins("@P2 BAR.ARV 0x6, 0x100", stall=cfg["bst"]), lw)
                    put(Lg(L + 8), Ins("LOP3.LUT R%d, R%d, 0x3, RZ, 0x3c, !PT" % (cn, cn)), lw)
                else:
                    put(g0, Ins(addi(cn, cn, -2)))
                    put(g0 + 3 * sp + 1, Ins("BAR.ARV R%d, 0x100" % cn, stall=cfg["bst"]))
                    put(g0 + 3 * sp + 2, Ins("LOP3.LUT R%d, R%d, 0x1, RZ, 0x3c, !PT" % (cn, cn)))
                    if not cfg["bnsy"]:
                        put(Lg(L - 1), Ins("BAR.SYNC.DEFER_BLOCKING R%d, 0x100" % cn, stall=cfg["bst"]), lw)
                    put(Lg(L), Ins(addi(cn, cn, 2)), lw)
                    put(Lg(L + 7), Ins("BAR.ARV R%d, 0x100" % cn, stall=cfg["bst"]), lw)
                    put(Lg(L + 8), Ins("LOP3.LUT R%d, R%d, 0x1, RZ, 0x3c, !PT" % (cn, cn)), lw)
            else:
                put(g0 + 3 * sp + 1, Ins("BAR.SYNC.DEFER_BLOCKING R%d, 0x40" % cn, stall=1))
                put(b + cfg["bmid"], Ins("BAR.SYNC.DEFER_BLOCKING R%d, 0x40" % cn, stall=1), False)
            if cfg["bx"]:              # to the partner's area for the loads, back after them
                L = b + 2 * ni - 6
                put(g0 + 3 * sp + 2, Ins("LOP3.LUT R%d, R%d, 0x800, RZ, 0x3c, !PT" % (xr_, xr_), wait=(cfg["bxs"],)))
                put(L + 7 if s != 3 else (L + 7) % tot, Ins("LOP3.LUT R%d, R%d, 0x800, RZ, 0x3c, !PT" % (xr_, xr_),
                                                             wait=(cfg["bxs"],)),
                    False if s == 3 else None)
        xa = xr_ if cfg["bx"] else SW
        for p in range(4):                             # the next turn's B: pair p's PRMT at r 8mt-5+2p (+1),
            at = "+0x%x" % (512 * p) if p else ""
            if brole == "load" and cfg["bsym"] == 2:
                ldsplans.add(len(plans))
                plans.append((b + cfg["bld"] + p, b + cfg["bld"] + p + lead, b + ni - 2 + 2 * p, SW, 512 * p,
                              BF + 4 * p, None, None))
                continue
            if brole == "load":
                # The partner's pair, one 16-byte load a gap behind the last IMMA of this turn that reads
                # the pair; the next turn's first readers of it, six gaps on, wait for the load. With a
                # second set of registers (bfa) the load goes out early.
                tgt = (bfalt if bfalt is not None and (s + 1) & 1 else BF) + 4 * p
                lsb = SB_RD if cfg["bsym"] == 3 else (0, 1, 2, SB_RD)[p] if cfg["bsym"] == 4 else SB_ST
                if cfg["bnld"]:
                    continue
                put(b + 20 + 4 * p if bfalt is not None else b + ni - 6 + 2 * p,
                    Ins("LDS.128 R%d, [R%d%s]" % (tgt, xa, at), wsb=lsb, rsb=cfg["bxs"] if cfg["bx"] and p == 3 else None))
                for g in (b + ni + 2 * p, b + ni + 2 * p + 1):
                    iwait.setdefault(g % tot, []).append(lsb)
                continue
            if brole == "make" and not cfg["bnst"]:     # the pair to the stage, a gap behind its last PRMT
                put(b + ni + (cfg["bsg3"] if s == 3 and cfg["bsg3"] >= 0 else cfg["bsg"]) + cfg["bss"] * p if cfg["bsg"] >= 0 else b + ni - (1 if cfg["bsym"] >= 2 else 3) + 2 * p, Ins("STS.128 [R%d%s], R%d" % (xa, at, BF + 4 * p),
                    wait=(SB_ST,) if cfg["bx"] and not cfg["bxt"] and s == 3 else (), rsb=cfg["bxs"] if cfg["bx"] and p == 3 else None))
            for h in range(2):                         # two IMMA behind the last IMMA that reads the pair
                n = 2 * p + h
                g0 = b + ni - 9 + n
                if cfg["bsp"]:
                    # bsp: the LDSM of the first B gathers go out early and apart (the quads allow five
                    # gathers under way), so that the warps a barrier has put in step do not all gather in
                    # the same eight gaps: with bank collisions that is more than shared memory serves.
                    g_ld = b + BLD[s == 3][n]
                    g_prm = g0 + lead + plead
                    if cfg["cq"] and n > 5:
                        g_ld = b + BLD[s == 3][n - 1]      # the sixth gather's LDSM gap is free: a gap earlier
                    if cfg["cq"] and n == 5:
                        # In the record quad (`cur`), which the load in gap 2 of the next turn writes again.
                        assert role is None and cfg["cq"] in (35, 36) and g_prm <= b + ni + 1
                        if s == 3:
                            last3[n] = len(plans)
                        plans.append((b + cfg["cq"], g_prm - plead, g_prm, nxt + 2 + h, 2048 * p, BF + 4 * p + h,
                                      BF + 4 * p + 2 + h, None, cur))
                        continue
                    if cfg["bp"] and n >= 5:
                        g_prm = b + ni - 3 + 2 * p + h     # pair p is first read in gap ni + 2 p of the next turn
                    g_add = g_ld + lead
                    if (cfg["bl"] and n < 5) or (cfg["bp"] and n >= 5):
                        g_add = max(g_add, g_prm - plead)
                    if s == 3 and n >= 5:
                        last3[n] = len(plans)
                    plans.append((g_ld, g_add, g_prm, nxt + 2 + h, 2048 * p, BF + 4 * p + h,
                                  BF + 4 * p + 2 + h, None))
                else:
                    if s == 3 and n >= 5:
                        last3[n] = len(plans)
                    plans.append((g0, g0 + lead, g0 + lead + plead, nxt + 2 + h, 2048 * p, BF + 4 * p + h,
                                  BF + 4 * p + 2 + h, None))
    # The first turn's stores: tile m's four words, a gap behind its last PRMT (tile 0's, made at the end of
    # the chunk before, in gap st0). The set is written again 9 gaps later or more.
    for m in range(mt) if not cfg["nost"] else ():
        g = 8 * (m - 1) + lead + plead + 2 + (cfg["al"] if m < mt - 1 else 0) if m else (
            tot - 9 + lead + plead if cfg["sync"] == 4 else cfg["st0"])
        put(g, Ins("STS.128 [R%d%s], R%d" % (SW, "+0x%x" % (512 * m) if m else "", AF[m & 1])), False)
    # A gather takes the lowest quad that is free (a quad is free from the gap behind its last PRMT); the
    # chunk before left its last gathers under way, so the chunk is walked twice and the second walk kept.
    # The scoreboards go round 0-2 in the order the LDSM are issued.
    order = sorted(range(len(plans)), key=lambda k: plans[k][0])
    assert len({plans[k][0] for k in order}) == len(order), "two LDSM in one gap"
    nq = len(quads)
    free_at = [-tot] * nq
    for walk in range(2):
        took = []
        for k in order:
            if len(plans[k]) > 8:
                took.append(None)
                continue
            g_ld, g_prm = plans[k][0] + walk * tot, plans[k][2] + walk * tot
            qi = next((i for i in range(nq) if free_at[i] <= g_ld), None)
            assert qi is not None, "more than %d gathers under way in gap %d" % (nq, plans[k][0])
            free_at[qi] = g_prm + 1
            took.append(qi)
    sbs = {k: n % 3 for n, k in enumerate(order)}
    SBO[:] = [0, 0]
    if cfg["sbo"]:
        W = cfg["sbo"]

        def held(i, j):
            # What gather j's IADD3 waits for gather i's load: i went out after j's own LDSM, d gaps (1 to
            # W - 1) ahead of the IADD3. A load in the IADD3's own gap goes out behind it.
            d = (plans[j][1] - plans[i][0]) % tot
            return W - d if 1 <= d < min(W, (plans[j][1] - plans[j][0]) % tot) else 0

        def cost(k, x):
            return sum(held(i, k) + held(k, i) for i in order if i != k and sbs[i] == x)

        SBO[0] = sum(cost(k, sbs[k]) for k in order) // 2
        for _sweep in range(16):
            moved = 0
            for k in order:
                best = min(range(3), key=lambda x: (cost(k, x), x != sbs[k]))
                if cost(k, best) < cost(k, sbs[k]):
                    sbs[k], moved = best, moved + 1
            if not moved:
                break
        SBO[1] = sum(cost(k, sbs[k]) for k in order) // 2
    use, assign = {}, {}
    for n, k in enumerate(order):
        g_ld, g_add, g_prm, areg, aoff, d1, d2, rsb = plans[k][:8]
        assert g_ld + lead <= g_add < g_prm
        q, sb = plans[k][8] if len(plans[k]) > 8 else quads[took[n]], sbs[k]
        for (a0, a1) in use.get(q, []):
            assert not any(a0 <= g_prm + w and g_ld + w <= a1 for w in (-tot, 0, tot)), (q, g_ld)
        use.setdefault(q, []).append((g_ld, g_prm))
        if k in ldsplans:
            put(g_ld, Ins("LDS.128 R%d, [R%d+0x%x]" % (q, areg, aoff) if aoff else "LDS.128 R%d, [R%d]" % (q, areg),
                          wsb=sb), wrapped=False)
            for i in range(4):
                # regs 0, 1 a gap before regs 2, 3 (tile 2p reads the first two, tile 2p + 1 the others)
                put(g_prm - 1 + (i >= 2), Ins(fold.mov(d1 + i, q + i), wait=(sb,)), wrapped=False)
            assign[k] = (q, sb, True)
            continue
        moves = cfg["nold"] == 1 or (cfg["nold"] == 2 and d1 < BF) or (cfg["nold"] == 3 and d1 >= BF)
        ldk = cfg["ldk"] == 3 or (cfg["ldk"] == 2 and d1 < BF) or (cfg["ldk"] == 1 and d1 >= BF)
        ld, adds, prm = gather_ins(q, areg, aoff, d1, d2, sb, dict(cfg, nold=int(moves), ldk=int(ldk)), rsb=rsb)
        if not prm:
            g_add = g_prm               # b8: the subtracts write the fragment words, where the PRMT did
        if cfg["noprm"]:
            # TIMING ONLY: no regroup; the subtracts write the fragment words, in the PRMT's gap.
            adds = [Ins("IADD3 R%d, -R%d, 0x%x, R%d" % (d, q + 2 + h, gfold.FILL4, q + h), wait=(sb,))
                    for h, d in enumerate((d1, d2))]
            prm, g_add = [], g_prm
        put(g_ld, ld, wrapped=False)
        if len(adds) == 4:
            # xa/xb: the XOR steps a gap behind the subtracts and ahead of the PRMT, both in the same chunk.
            g_add = min(g_add, g_prm - 2)
            if g_add == tot - 1:
                g_add -= 1
            assert g_add > g_ld, (g_ld, g_add, g_prm)
            for x in adds[2:]:
                put(g_add + 1, x, wrapped=False)
            adds = adds[:2]
        for x in adds:
            put(g_add, x, wrapped=False)
        for x in prm:
            put(g_prm, x, wrapped=False)
        assign[k] = (q, sb, g_add < tot)
    # What the prologue has to leave under way: the last turn's B gathers 5, 6 and 7, whose PRMT (and, for
    # some, IADD3) lie in the next chunk's first gaps.
    tail = [assign[last3[n]] for n in (5, 6, 7)] if len(last3) == 3 else []
    assert role == "load" or cfg["cq"] or [plans[k][0] for k in order[-3:]] == [tot - 4, tot - 3, tot - 2]
    # The readout, as gfold's: two chains of XOR3, one step of each a gap, from r roff of the last turn.
    b3 = 3 * ni
    busy = set(range(b3 + ni - cfg["rb0"], tot + cfg["rb1"]))
    end = b3
    for ch in range(2):
        g = b3 + cfg["roff"]
        first = True
        for t in (range(mt) if cfg["rpr"] == 2 else range(ch, mt, 2)):
            for j in range(16):
                if cfg["rpr"] == 2 and (j & 1) == ch:
                    continue                                              # rpr=2: chain ch takes the other parity
                nb = j // 2
                made, again = b3 + 8 * t + nb, tot + 8 * t + nb          # the sum's last IMMA, and its next
                g = max(g, made + cfg["rmar"])
                if cfg["rsp"]:
                    # rsp: the chain's 48 steps spread evenly over rsp gaps from roff, not one a gap.
                    g = max(g, b3 + cfg["roff"] + (8 * (t // 2) * 2 + j) * cfg["rsp"] // 48)
                while g in busy:
                    g += 1
                assert g < again, "tile %d's sum %d is read in gap %d, behind its next IMMA" % (t, nb, g)
                r = ACC + 4 * (8 * t + nb) + 2 * (j & 1)
                r2 = r + 1
                if cfg["rpr"]:
                    # rpr: an IMMA's four sums read as (0, 2) and (1, 3), not (0, 1) and (2, 3): a step's two sums
                    # have the same register parity. rpr=2 also gives the even pairs to the odd chain (R3) and the
                    # odd pairs to the even one (R2). Every sum is still read once: correct.
                    r = ACC + 4 * (8 * t + nb) + (j & 1)
                    r2 = r + 2
                if cfg["rdx"] == 2:
                    # rdx=2, TIMING ONLY: the chain and one sum (two reads), not the chain and two.
                    ins = Ins("LOP3.LUT R%d, R%d, R%d, RZ, 0x3c, !PT" % (R["xr"][ch], R["xr"][ch], r))
                    first = False
                elif cfg["rdx"]:
                    # rdx, TIMING ONLY: the same XOR3 steps and chains, but reading no sums (RZ in their place).
                    ins = Ins("LOP3.LUT R%d, R%d, RZ, RZ, 0x3c, !PT" % (R["xr"][ch], R["xr"][ch]))
                    first = False
                elif first:
                    ins = Ins("LOP3.LUT R%d, R%d, R%d, RZ, 0x3c, !PT" % (R["xr"][ch], r, r2))
                    first = False
                else:
                    ins = Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (R["xr"][ch], R["xr"][ch], r, r2))
                if cfg["rst"]:
                    ins.stall = cfg["rst"]
                if not cfg["nord"]:
                    put(g, ins, front=bool(cfg["rgf"]))
                g += 1
        end = max(end, g)
    assert end > tot, "the readout's tail is taken to lie in the next chunk"
    trw, w, xr = R["trw"], R["w"], R["xr"]
    lds, sts = ("LDL", "STL") if cfg["bx"] else ("LDS", "STS")
    cn = R["cnt"]
    if cfg["bxt"]:
        # The word's address from the count, into the scratch register, then the word over it; after the chains'
        # last read the address again into R3 for the store.
        assert cfg["bx"] and cfg["bord"] == 2
        for i, t in enumerate((shr(w, cn, 2), "IADD3 R%d, -R%d, 0x0, RZ" % (w, w), andi(w, w, 0x3c),
                               "IADD3 R%d, R%d, 0x0, R1" % (w, w))):
            put(end - 8 + i, Ins(t), wrapped=True)
        put(end - 4, Ins("LDL R%d, [R%d]" % (w, w), wsb=SB_RD), wrapped=True)
    else:
        put(end - 4, Ins("%s R%d, [R%d]" % (lds, w, trw), wsb=SB_RD), wrapped=True)
    put(end + 0, Ins("SHF.L.W.U32.HI R%d, R%d, 0xd, R%d" % (w, w, w), wait=(SB_RD,)), wrapped=True)
    put(end + 1, Ins("LOP3.LUT R%d, R%d, R%d, R%d, 0x96, !PT" % (w, w, xr[0], xr[1])), wrapped=True)
    if cfg["bxt"]:
        x3 = xr[1]
        for i, t in enumerate((shr(x3, cn, 2), "IADD3 R%d, -R%d, 0x0, RZ" % (x3, x3), andi(x3, x3, 0x3c),
                               "IADD3 R%d, R%d, 0x0, R1" % (x3, x3))):
            put(end + 2 + i, Ins(t), wrapped=True)
        put(end + 6, Ins("STL [R%d], R%d" % (x3, w)), wrapped=True)
    else:
        put(end + 2, Ins("%s [R%d], R%d" % (sts, trw, w), rsb=hold), wrapped=True)
    # The next word of the sixteen: the pointer is 64-aligned, so its low six bits are the word. The scratch
    # register is the word the store is still reading: its next writer waits for the store.
    if cfg["bxt"]:
        pass
    elif cfg["bx"]:
        # The own exchange area's address back into R3 (the second chain is done with), ahead of the first store
        # (which waits for it); the frame's word pointer moves on without assuming the frame is 64-aligned.
        assert end + 6 <= tot + (cfg["bsg3"] if cfg["bsg3"] >= 0 else cfg["bsg"]), "bx: the tail (to gap %d) runs into the first store; raise bsg" % (end + 6 - tot)
        put(end + 3, Ins("LDL R%d, [R1+0x40]" % XR[1], wsb=SB_ST), wrapped=True)
        put(end + 4, Ins("IADD3 R%d, -R1, 0x4, R%d" % (w, trw), wait=hw), wrapped=True)
        put(end + 5, Ins(andi(w, w, 0x3c)), wrapped=True)
        put(end + 6, Ins("IADD3 R%d, R%d, 0x0, R1" % (trw, w)), wrapped=True)
    else:
        put(end + 4, Ins(addi(w, trw, 4), wait=hw), wrapped=True)
        put(end + 5, Ins(andi(w, w, 0x3c)), wrapped=True)
        put(end + 6, Ins("LOP3.LUT R%d, R%d, 0x3f, R%d, 0xba, !PT" % (trw, trw, w), wait=hw), wrapped=True)
    assert end + 6 - tot < (ni - 4 if cfg["rsp"] else ni - 16)
    # The chunk's count and the offsets pointer, in quiet gaps of the last turn; and the barrier's test.
    put(b3 + 6, Ins(addi(R["cnt"], R["cnt"], -16 if cfg["brg"] or cfg["bord"] else -1)), wrapped=False)
    put(b3 + 7, Ins(("ISETP.GT.U32.AND P1, PT, R%d, 0xf, PT" if cfg["brg"] or cfg["bord"] else "ISETP.NE.AND P1, PT, R%d, 0x0, PT")
                    % R["cnt"]), wrapped=False)
    put(b3 + 14, Ins(addi(R["poff"], R["poff"], 0x400 if cfg["co"] else 0x800), wait=hw), wrapped=False)
    if cfg["barn"] and cfg["sync"] != 4:
        assert cfg["barn"] & (cfg["barn"] - 1) == 0
        put(b3 + 9, Ins(andi(w, R["cnt"], cfg["barn"] - 1)), wrapped=False)
        put(b3 + 10, Ins("ISETP.NE.AND P2, PT, R%d, 0x0, PT" % w), wrapped=False)
    return G, iwait, tail, bfalt


def barrier(cfg):
    """The k-step barrier: all eight warps, or (band) the four of the warp's row band; P2 is set for band 1."""
    if not cfg["band"]:
        return [Ins("BAR.SYNC.DEFER_BLOCKING 0x0", stall=1)]
    if cfg["brg"]:
        return [Ins("BAR.SYNC.DEFER_BLOCKING R%d, 0x80" % layout(cfg["mt"])["cnt"], stall=1)]
    return [Ins("@!P2 BAR.SYNC.DEFER_BLOCKING 0x1, 0x80", stall=1), Ins("@P2 BAR.SYNC.DEFER_BLOCKING 0x2, 0x80", stall=1)]


def with_barriers(cfg, loop):
    """The loop with a barrier at every k-step's end (sync=4)."""
    ni, out, seen = 8 * cfg["mt"], [], 0
    for x in loop:
        if x.text.startswith("IMMA"):
            if seen and seen % ni == 0 and cfg["bt"] >> (seen // ni - 1) & 1:
                out += barrier(cfg)
            seen += 1
        if "BRA" in x.text and cfg["bt"] & 8:
            out += barrier(cfg)
        out.append(x)
    return out


def build_bsh(cfg):
    """bsh (TIMING ONLY): two loops in the function, band 0's ("make") and band 1's ("load"). The prologue is
    band 0's for both; band 1 branches to its loop behind it."""
    assert cfg["sync"] == 4
    gfold.UR = cfg["ur"]
    R = layout(cfg["mt"])
    # bsh=1: band 0 makes and band 1 loads. bsh=2: the other way round (each scheduler serves its band 1 warp
    # first, so this gives the warp with more to do the first turn at the pipe). With band=1 the barriers are
    # one a row band, as in the shipped loop: no fold could be correct that way without a deeper stage, it
    # prices B shared apart from what one barrier for all eight warps costs.
    roles = ("make", "load") if cfg["bsh"] == 1 else ("load", "make")
    G0, iw0, tail0, alt0 = chunk(cfg, R, roles[0])
    G1, iw1, tail1, alt1 = chunk(cfg, R, roles[1])
    tail = tail0 if roles[0] == "make" else tail1
    pro = prologue(cfg, R, tail)
    bad = gfold.check(pro, "prologue", fix=True)
    code, most = list(pro) + [Ins("@P2 BRA .L1", stall=5)], 0
    for n, (G, iw, alt) in enumerate(((G0, iw0, alt0), (G1, iw1, alt1))):
        loop, ep, m = gfold.emit(G, iw, cfg, alt)
        most = max(most, m)
        loop = with_barriers(cfg, loop)
        body = [x for x in loop if "BRA" not in x.text and "BAR" not in x.text]
        bad += gfold.check(pro + body + body + body, "loop %d" % n)
        bad += gfold.check(body + ep, "epilogue %d" % n, fix=True)
        if n:
            assert loop[0].label == ".L0" and loop[-1].text == "@P1 BRA .L0"
            loop[0].label, loop[-1].text = ".L1", "@P1 BRA .L1"
        code += loop + ep
        if not n:
            code.append(Ins("BRA .END", stall=5))
    # Every load has landed before the function returns.
    code.append(Ins(addi(R["tmp"], R["tmp"], 0), wait=(0, 1, 2, 3, SB_OFF, SB_RD), label=".END"))
    code.append(Ins("NOP", stall=8))
    return code, bad, most, G0 + G1


def build(cfg):
    if cfg["bsh"]:
        return build_bsh(cfg)
    gfold.UR = cfg["ur"]
    R = layout(cfg["mt"])
    G, iwait, tail, _alt = chunk(cfg, R)
    pro = prologue(cfg, R, tail)
    bad = gfold.check(pro, "prologue", fix=True)
    loop, ep, most = gfold.emit(G, iwait, cfg)
    if cfg["sync"] == 4:
        loop = with_barriers(cfg, loop)
    elif cfg["barn"]:
        loop.insert(len(loop) - 1, Ins("@!P2 BAR.SYNC.DEFER_BLOCKING 0x0", stall=1))
    body = [x for x in loop if "BRA" not in x.text and "BAR" not in x.text]
    bad += gfold.check(pro + body + body + body, "loop")
    bad += gfold.check(body + ep, "epilogue", fix=True)
    # Every load has landed before the function returns.
    if cfg["bord"] == 2:
        # The epilogue's wrapped stores arrived on 4 + band; the other band closes that here.
        ep.append(Ins("NOP", stall=6))
        if not cfg["bnsy"]:
            ep.append(Ins("BAR.SYNC.DEFER_BLOCKING R%d, 0x100" % R["cnt"], stall=cfg["bst"]))
    ep.append(Ins(addi(R["tmp"], R["tmp"], 0), wait=(0, 1, 2, 3, SB_OFF, SB_RD)))
    ep.append(Ins("NOP", stall=8))
    return pro + loop + ep, bad, most, G


def expect(text, cfg, code):
    """What the hand loop takes for granted about the compiled kernel, checked against its listing. Returns the
    list of things that differ (empty when the loop fits). A toolkit whose ptxas compiles src/pearl_rot.cu
    another way shows up here, and the cubin is then not written."""
    out = []
    regs, preds = asm89.clobbers(text)
    used, consts = set(), set()
    for ins in code:
        consts.update(int(x, 16) for x in re.findall(r"c\[0x0\]\[0x([0-9a-f]+)\]", ins.text))
        if "BRA" in ins.text or ins.text.startswith("ULDC"):
            continue
        _r, w, _k = fold.rw(ins.text)
        used |= {x for x in w if not isinstance(x, str)}
    if not used <= regs:
        out.append("the loop writes registers the compiled function does not, so the kernel may keep values in "
                   "them: R%s" % ", R".join(str(x) for x in sorted(used - regs)))
    if len(preds) < 2:
        out.append("the compiled function writes %d predicates and the loop needs two" % len(preds))
    facts = asm89.loop_facts(text)
    if facts["urs"] != {cfg["ur"]}:
        out.append("the loop loads the address base into UR%d and the compiled function into %s" % (
            cfg["ur"], ", ".join("UR%d" % u for u in sorted(facts["urs"])) or "no uniform register"))
    if not consts <= facts["consts"]:
        out.append("the loop reads kernel parameters the compiled function does not: %s" % ", ".join(
            "c[0x0][0x%x]" % c for c in sorted(consts - facts["consts"])))
    # The transcripts' place in shared memory: the kernel zeroes them at this address before the call.
    tr = layout(cfg["mt"])["tr"]
    if cfg["bx"]:
        if not re.search(r"STL\.128 \[R1\], RZ", text) or not re.search(r"STL\.128 \[R1\+0x30\], RZ", text):
            out.append("the kernel does not zero its transcript at R1 + 0 to 0x3f of its frame (bx)")
        if "[R1+0x40]" in text or "[R1+0x44]" in text:
            out.append("the kernel uses R1 + 0x40 of its frame, where the loop keeps its exchange address (bx)")
    elif not re.search(r"STS\.128 \[R\d+\+0x%x\], RZ" % tr, text):
        out.append("the kernel does not zero its transcripts at shared address 0x%x, where the loop keeps them" % tr)
    if len(code) + 3 > facts["room"]:
        out.append("the loop is %d instructions and the compiled function has room for %d" % (
            len(code) + 3, facts["room"]))
    return out


def main():
    cubin, sass, nvd, out = sys.argv[1:5]
    cfg = dict(CFG)
    given = set()
    for kv in sys.argv[5:]:
        k, v = kv.split("=")
        assert k in cfg, k
        cfg[k] = int(v, 0)
        given.add(k)
    text = asm89.read_text(sass)
    # The address base goes into the uniform register pair the compiled function uses (UR4 under ptxas 12.6),
    # unless a key names one.
    urs = asm89.loop_facts(text)["urs"]
    if "ur" not in given and len(urs) == 1 and urs != {cfg["ur"]}:
        cfg["ur"] = next(iter(urs))
        print("note: the compiled function keeps the address base in UR%d; the loop uses it too" % cfg["ur"])
    code, bad, most, G = build(cfg)
    plain = sum(len(x) for x in G)
    if bad:
        print("%d latency problems: not written" % bad)
        sys.exit(1)
    print("latency check: every register is read late enough behind its writer")
    regs, preds = asm89.clobbers(text)
    wrong = expect(text, cfg, code)
    if wrong:
        print("the compiled kernel is not what the hand loop expects: not written")
        for w in wrong:
            print("  - " + w)
        sys.exit(1)
    # The loop's own predicate is P1 and the row band's P2 where the compiled function writes them; a build whose
    # function writes others gets those, since the kernel may keep the rest (with the hash in the probe's kernel
    # the function writes P0 and P2, in the miner's kernel, src/pearl_rot.cu, P0 and P1).
    bp = 2 if 2 in preds else min(preds - {1})
    lp = 1 if 1 in preds else min(preds - {bp})
    assert bp != lp
    for ins in code:
        t = ins.text
        if lp != 1:
            if t.startswith("ISETP.NE.AND P1, "):
                t = "ISETP.NE.AND P%d, " % lp + t[len("ISETP.NE.AND P1, "):]
            elif t.startswith("ISETP.GT.U32.AND P1, "):
                t = "ISETP.GT.U32.AND P%d, " % lp + t[len("ISETP.GT.U32.AND P1, "):]
            elif t.startswith("@P1 BRA "):
                t = "@P%d BRA " % lp + t[len("@P1 BRA "):]
        if bp != 2:
            if t.startswith("ISETP.NE.AND P2, "):
                t = "ISETP.NE.AND P%d, " % bp + t[len("ISETP.NE.AND P2, "):]
            elif t.startswith("@!P2 BAR") or t.startswith("@P2 BAR"):
                t = t.replace("P2 BAR", "P%d BAR" % bp, 1)
            elif t.startswith("@P2 BRA "):
                t = "@P%d BRA " % bp + t[len("@P2 BRA "):]
        ins.text = t
    n, room = asm89.patch(cubin, text, out, code, nvd)
    if cfg["band"] and cfg["bord"]:
        # bord: barriers 4 to 7 (bord=2) or 4 + slot (bord=1) besides the bands' 1 and 2.
        asm89.set_barriers(out, 12)
        if asm89.barriers(asm89.disassemble(out, nvd)) != 12:
            os.remove(out)
            print("the cubin does not say 12 barriers after the patch: not written")
            sys.exit(1)
    elif cfg["band"]:
        asm89.set_barriers(out, 3)
        have = asm89.barriers(asm89.disassemble(out, nvd))
        if have != 3:
            os.remove(out)
            print("the cubin says %d barriers after the patch, not 3: not written" % have)
            sys.exit(1)
    print("registers: the loop writes only what the compiled function writes; predicates P%d (loop) and P%d (row "
          "band); address base in UR%d; read back through nvdisasm: %d instructions as asked" % (lp, bp, cfg["ur"], n))
    if cfg["bsh"]:
        print("TIMING ONLY (bsh): this cubin computes wrong results on purpose")
    if cfg["sbo"]:
        print("scoreboards: waits for another gather's load, weighed, %d handed out in turn, %d as chosen" % tuple(SBO))
    tot = 32 * cfg["mt"] * (2 if cfg["bsh"] else 1)
    kinds = {}
    for g in G:
        for ins, _w in g:
            op = ins.text.split(" ")[0].split(".")[0]
            kinds[op] = kinds.get(op, 0) + 1
    print("%s: %d instructions of %d; a chunk is %d IMMA and %d others (%.2f an IMMA), %d at most in a gap; %s" % (
        os.path.basename(out), n, room, tot, plain, plain / float(tot), most,
        ", ".join("%s %d" % kv for kv in sorted(kinds.items(), key=lambda kv: -kv[1]))))


if __name__ == "__main__":
    main()
