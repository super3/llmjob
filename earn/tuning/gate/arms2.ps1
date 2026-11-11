# arms2.ps1 <mode> <arm> [<arm> ...] : build the follow-up's arms (all with no names) with build.ps1.
# The loop gap (measurement 1) and the producer warp (measurement 2), 11 November. rig01, sm_89.
param([string]$mode, [Parameter(ValueFromRemainingArguments = $true)][string[]]$only)
$S = $env:GATE_S
$P = '-DPEARL_ABLATE_DRAM -DPEARL_GATE_PHANTOM=1'
$NC = "-DPEARL_GATE_NOCOPY=1 $P"
$K = '-DPEARL_FOLD_K=65536 -DPEARL_ADA_COL_BATCH=64'
$B = "$NC $K"                                    # the base of both measurements: no copies, k 65536 (nck)
$M4 = "$B -DPEARL_GATE_MMA_MASK=0x0F"            # warps 0 to 3 fold, one a scheduler
$M7 = "$B -DPEARL_GATE_MMA_MASK=0x7F"
$M6 = "$B -DPEARL_GATE_MMA_MASK=0x3F"
$N = '-DPEARL_GATE_PROD=1 -DPEARL_GATE_PROD_NAP=100'   # producers that test their barrier every 100 ns
$W = "$N -DPEARL_GATE_PROD_RNAP=50"
$T4 = "$B -DPEARL_TALL_ROW_TILES=4 -DPEARL_TALL_HASH_PAIRS=0 -DPEARL_TALL_BPT=6"   # 64x64 warps, a 128x256 tile
$arms = [ordered]@{
  # measurement 1: one thing removed or moved at a time
  kro    = "$B -DPEARL_ABLATE_READOUT"
  krg    = "$B -DPEARL_ABLATE_RING -DPEARL_ABLATE_ARRIVE"
  krw    = "$B -DPEARL_ABLATE_RING"
  khs    = "$B -DPEARL_ABLATE_TRANSCRIPT_HASH"
  kall   = "$B -DPEARL_ABLATE_READOUT -DPEARL_ABLATE_RING -DPEARL_ABLATE_ARRIVE -DPEARL_ABLATE_TRANSCRIPT_HASH"
  kbs    = "$B -DPEARL_GATE_BSTREAM=1"
  kz1    = "$B -DPEARL_GATE_RO_LAZY=1"
  kz2    = "$B -DPEARL_GATE_RO_LAZY=2"
  kbz1   = "$B -DPEARL_GATE_BSTREAM=1 -DPEARL_GATE_RO_LAZY=1"
  rz1    = '-DPEARL_GATE_RO_LAZY=1'
  rz2    = '-DPEARL_GATE_RO_LAZY=2'
  # measurement 2, the eight-warp block with warps given up. Warps 0 to 3 fold (one a scheduler) and 4 to 7
  # produce: the tile's used part is 96x256, 44 units of 512 bytes a stage, 11 a producer lane. A unit is
  # 3 + ALU instructions, and a lane's 11 stand against the 96 IMMA a folding lane makes: ALU 6, 14 and 29
  # are 1.03, 1.95 and 3.67 instructions an IMMA (9, 17 and 32 a unit).
  m4k    = $M4
  m4p0k  = "$M4 -DPEARL_GATE_PROD=1"
  m4p2k  = "$M4 -DPEARL_GATE_PROD=1 -DPEARL_GATE_PROD_UNITS=11 -DPEARL_GATE_PROD_ALU=14"
  m4n0k  = "$M4 $N"
  m4n1k  = "$M4 $N -DPEARL_GATE_PROD_UNITS=11 -DPEARL_GATE_PROD_ALU=6"
  m4n2k  = "$M4 $N -DPEARL_GATE_PROD_UNITS=11 -DPEARL_GATE_PROD_ALU=14"
  m4n3k  = "$M4 $N -DPEARL_GATE_PROD_UNITS=11 -DPEARL_GATE_PROD_ALU=29"
  # Seven fold and warp 7 produces all 56 units of the stage: ALU 9 is 1.00 an IMMA.
  m7k    = $M7
  m7n0k  = "$M7 $N"
  m7n1k  = "$M7 $N -DPEARL_GATE_PROD_UNITS=56 -DPEARL_GATE_PROD_ALU=9"
  # measurement 2, a smaller fold with producer warps beside its eight: 64x64 warps, a 128x256 tile, 48 units
  # a stage against 64 IMMA a folding lane. Four producers (12 a lane): ALU 8, 18 and 29 are 1.03, 1.97 and
  # 3.00 an IMMA (11, 21 and 32 a unit). One producer (48 a lane): the same numbers, on one scheduler.
  t4k    = $T4
  t12n0k = "$T4 $N -DPEARL_GATE_PROD_WARPS=4"
  t12n1k = "$T4 $N -DPEARL_GATE_PROD_WARPS=4 -DPEARL_GATE_PROD_UNITS=12 -DPEARL_GATE_PROD_ALU=8"
  t12n2k = "$T4 $N -DPEARL_GATE_PROD_WARPS=4 -DPEARL_GATE_PROD_UNITS=12 -DPEARL_GATE_PROD_ALU=18"
  t12n3k = "$T4 $N -DPEARL_GATE_PROD_WARPS=4 -DPEARL_GATE_PROD_UNITS=12 -DPEARL_GATE_PROD_ALU=29"
  t12p2k = "$T4 -DPEARL_GATE_PROD=1 -DPEARL_GATE_PROD_WARPS=4 -DPEARL_GATE_PROD_UNITS=12 -DPEARL_GATE_PROD_ALU=18"
  # The same with folding warps that nap between tests of a stage that is not full yet (W): spinning, they
  # starve a producer that cannot stay ahead.
  t12w0k = "$T4 $W -DPEARL_GATE_PROD_WARPS=4"
  t12w1k = "$T4 $W -DPEARL_GATE_PROD_WARPS=4 -DPEARL_GATE_PROD_UNITS=12 -DPEARL_GATE_PROD_ALU=8"
  t12w2k = "$T4 $W -DPEARL_GATE_PROD_WARPS=4 -DPEARL_GATE_PROD_UNITS=12 -DPEARL_GATE_PROD_ALU=18"
  t12w3k = "$T4 $W -DPEARL_GATE_PROD_WARPS=4 -DPEARL_GATE_PROD_UNITS=12 -DPEARL_GATE_PROD_ALU=29"
  m7w1k  = "$M7 $W -DPEARL_GATE_PROD_UNITS=56 -DPEARL_GATE_PROD_ALU=9"
  m4w2k  = "$M4 $W -DPEARL_GATE_PROD_UNITS=11 -DPEARL_GATE_PROD_ALU=14"
  # The same with producers that spin on their barrier (no nanosleep anywhere in the block).
  t12p0k = "$T4 -DPEARL_GATE_PROD=1 -DPEARL_GATE_PROD_WARPS=4"
  t12p1k = "$T4 -DPEARL_GATE_PROD=1 -DPEARL_GATE_PROD_WARPS=4 -DPEARL_GATE_PROD_UNITS=12 -DPEARL_GATE_PROD_ALU=8"
  t12p3k = "$T4 -DPEARL_GATE_PROD=1 -DPEARL_GATE_PROD_WARPS=4 -DPEARL_GATE_PROD_UNITS=12 -DPEARL_GATE_PROD_ALU=29"
  m7p0k  = "$M7 -DPEARL_GATE_PROD=1"
  m7p1k  = "$M7 -DPEARL_GATE_PROD=1 -DPEARL_GATE_PROD_UNITS=56 -DPEARL_GATE_PROD_ALU=9"
  t9n0k  = "$T4 $N -DPEARL_GATE_PROD_WARPS=1"
  t9n2k  = "$T4 $N -DPEARL_GATE_PROD_WARPS=1 -DPEARL_GATE_PROD_UNITS=48 -DPEARL_GATE_PROD_ALU=18"
}
foreach ($a in $arms.Keys) {
  if ($only -and ($only -notcontains $a)) { continue }
  & "$S\gate\build.ps1" $mode "$($env:GATE_PREFIX)$a" "$($arms[$a]) $($env:GATE_EXTRA)"
}
