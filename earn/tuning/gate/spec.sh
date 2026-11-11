#!/bin/bash
# spec.sh <3060|3090|5090|5070ti> : the VARIANTS_SPEC for that card's race (name|flags;...), one line.
# Arms (see pearl_config.h, PEARL_GATE_*): dram = DRAM ablation (copies kept, source in L2); sk3/sk1/sk2 = copies
# gated off (both, A', B'); sk3p = sk3 with phantom operands (no materialise on a new salt); ncp = copies compiled
# out, phantom; ncpr = ncp with the two stages as a loop (the shape the interleaved shams build without spill in);
# sh1 = sham 1 instruction an IMMA in runs at the k-step ends; si05r/si1r/si2r = sham between the mma at 0.5, 1, 2;
# sa1r = ALU only between the mma at 1; *k = the same at long k.
P="-DPEARL_ABLATE_DRAM -DPEARL_GATE_PHANTOM=1"
NC="-DPEARL_GATE_NOCOPY=1 $P"
R="-DPEARL_GATE_S_ROLL=1"
I="-DPEARL_GATE_SHAM_AT=2"
case "$1" in
  3060|3090)
    K="-DPEARL_FOLD_K=16384 -DPEARL_AMPERE_COL_BATCH=64"
    s="base|;dram|-DPEARL_ABLATE_DRAM;sk3|-DPEARL_TALL_AB_SKIP=3"
    [ "$1" = 3090 ] && s="$s;sk1|-DPEARL_TALL_AB_SKIP=1;sk2|-DPEARL_TALL_AB_SKIP=2;br|$R"
    s="$s;sk3p|-DPEARL_TALL_AB_SKIP=3 $P;ncp|$NC;ncpr|$NC $R;sh1|$NC -DPEARL_GATE_SHAM=2"
    s="$s;si05r|$NC $R -DPEARL_GATE_SHAM=1 $I;si1r|$NC $R -DPEARL_GATE_SHAM=2 $I;si2r|$NC $R -DPEARL_GATE_SHAM=4 $I"
    s="$s;sa1r|$NC $R -DPEARL_GATE_SHAM=2 -DPEARL_GATE_SHAM_AT=3"
    s="$s;dramk|$P $K;nck|$NC $K;ncprk|$NC $R $K;si1rk|$NC $R -DPEARL_GATE_SHAM=2 $I $K"
    [ "$1" = 3090 ] && s="$s;si05rk|$NC $R -DPEARL_GATE_SHAM=1 $I $K;si2rk|$NC $R -DPEARL_GATE_SHAM=4 $I $K"
    ;;
  5090|5070ti)
    # t*: the TMA ring (the release's). c*: the cp.async ring on sm_120 (-DPEARL_TALL_TMA=0), which the sham needs.
    if [ "$1" = 5090 ]; then K=65536; CB=2048; KB=64; else K=16384; CB=1024; KB=128; fi
    C="-DPEARL_TALL_TMA=0 -DPEARL_GATE_COL_BATCH=$CB"
    TK="-DPEARL_FOLD_K=$K -DPEARL_TMA_COL_BATCH=$KB"
    CK="-DPEARL_TALL_TMA=0 -DPEARL_FOLD_K=$K -DPEARL_GATE_COL_BATCH=$KB"
    s="base|;tdram|-DPEARL_ABLATE_DRAM;tsk3|-DPEARL_TALL_AB_SKIP=3"
    [ "$1" = 5090 ] && s="$s;tsk1|-DPEARL_TALL_AB_SKIP=1;tsk2|-DPEARL_TALL_AB_SKIP=2"
    s="$s;tsk3p|-DPEARL_TALL_AB_SKIP=3 $P;cbase|$C;cncp|$NC $C;cncpr|$NC $R $C;csh1|$NC -DPEARL_GATE_SHAM=2 $C"
    s="$s;csi05r|$NC $R -DPEARL_GATE_SHAM=1 $I $C;csi1r|$NC $R -DPEARL_GATE_SHAM=2 $I $C"
    s="$s;csi2r|$NC $R -DPEARL_GATE_SHAM=4 $I $C;csi1|$NC -DPEARL_GATE_SHAM=2 $I $C"
    s="$s;csa1r|$NC $R -DPEARL_GATE_SHAM=2 -DPEARL_GATE_SHAM_AT=3 $C"
    s="$s;tdramk|$P $TK;tsk3k|-DPEARL_TALL_AB_SKIP=3 $P $TK;cnck|$NC $CK;cncprk|$NC $R $CK"
    s="$s;csi1rk|$NC $R -DPEARL_GATE_SHAM=2 $I $CK"
    [ "$1" = 5090 ] && s="$s;csi05rk|$NC $R -DPEARL_GATE_SHAM=1 $I $CK;csi2rk|$NC $R -DPEARL_GATE_SHAM=4 $I $CK"
    ;;
  *) echo "spec.sh: 3060, 3090, 5090 or 5070ti" >&2; exit 2 ;;
esac
# The spin arms (a draw's work in a warp of its own, as a second kernel beside the no-copy fold): idle blocks
# (the control), then 0.5, 1 and 2 instructions an IMMA's worth. They race apart, after the main race.
case "$1" in
  3060|3090) SP="$NC -DPEARL_GATE_SPIN=1" ;;
  *) SP="$NC $C -DPEARL_GATE_SPIN=1 -DPEARL_GATE_SPIN_THREADS=96" ;;
esac
spin="spn0|$SP -DPEARL_GATE_SPIN_UNITS=0;spn05|$SP -DPEARL_GATE_SPIN_ALU=4;spn1|$SP -DPEARL_GATE_SPIN_ALU=11;spn2|$SP -DPEARL_GATE_SPIN_ALU=24"
B="-DPEARL_GATE_SHAM_AT=4"
Z="-DPEARL_GATE_SHAM_AT=3"
# The arms that split the sham's cost (see PEARL_GATE_SHAM_BAR, _NOP): d* names, on the rolled no-copy fold.
#   sa0r   ALU-only sham with no ALU steps: its stage-end release alone
#   sa1rb  ALU-only sham, 1 an IMMA, with the no-copy fold's arrival: the ALU steps alone
#   si0r   unit sham with no ALU steps: loads, stores and the release
#   si1rb  unit sham, 1 an IMMA, with the no-copy fold's arrival: loads, stores and ALU, no release
case "$1" in
  3060|3090) X="" ;;
  *) X=" $C" ;;
esac
split="sa0r|$NC $R -DPEARL_GATE_SHAM=2 $Z -DPEARL_GATE_SHAM_NOP=1$X;sa1rb|$NC $R -DPEARL_GATE_SHAM=2 $Z -DPEARL_GATE_SHAM_BAR=2$X;si0r|$NC $R -DPEARL_GATE_SHAM=2 $I -DPEARL_GATE_SHAM_NOP=1$X;si1rb|$NC $R -DPEARL_GATE_SHAM=2 $I -DPEARL_GATE_SHAM_BAR=2$X;si05rb|$NC $R -DPEARL_GATE_SHAM=1 $I -DPEARL_GATE_SHAM_BAR=2$X"
case "$1" in
  3060|3090) p2="base|;ncp|$NC;ncpr|$NC $R;si05r|$NC $R -DPEARL_GATE_SHAM=1 $I;si1r|$NC $R -DPEARL_GATE_SHAM=2 $I;sb1r|$NC $R -DPEARL_GATE_SHAM=2 $B"; x="sb1r|$NC $R -DPEARL_GATE_SHAM=2 $B"; r2="base ncp ncpr si1r sb1r sa0r sa1rb si0r si1rb si05rb spn0 spn05 spn1 spn2" ;;
  *) p2="base|;cncp|$NC $C;cncpr|$NC $R $C;csi05r|$NC $R -DPEARL_GATE_SHAM=1 $I $C;csi1r|$NC $R -DPEARL_GATE_SHAM=2 $I $C;csb1r|$NC $R -DPEARL_GATE_SHAM=2 $B $C;csi05rk|$NC $R -DPEARL_GATE_SHAM=1 $I $CK"; x="csb1r|$NC $R -DPEARL_GATE_SHAM=2 $B $C"; r2="base cncp cncpr csi1r csb1r sa0r sa1rb si0r si1rb si05rb spn0 spn05 spn1 spn2" ;;
esac
case "${2:-spec}" in
  pass2) echo "$p2;$split;$spin" ;;      # a second look at a PC already raced
  spec) echo "$s" ;;
  spec+spin) echo "$s;$x;$split;$spin" ;;
  race1) echo "$s" | tr ';' '
' | cut -d'|' -f1 | paste -sd' ' - ;;
  race2) echo "$r2" ;;
esac
