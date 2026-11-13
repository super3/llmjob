# The Intel (SYCL) Quantus core

`quantus_sycl.node`: a Quantus (QPoW) core for Intel Arc cards, to the contract in `../README.md`.
Plain N-API (no node-addon-api), one SYCL kernel, exact Goldilocks arithmetic, so its hits are
exactly the reference's.

| File | What it is |
|---|---|
| `qtc_poseidon2.h` | The field and the permutation, shared by the host and the kernel |
| `quantus_sycl.cpp` | The addon: device list, cores, the scan as launches, the kernel |
| `build.sh` | Builds it, as `../../sycl/build.sh` builds the Pearl core |
| `qtc_selftest.cpp` | Host-only check of the field code on the five official vectors (`build.sh` runs it) |
| `bench.js` | Back-to-back scans for a rate, the way the miner drives a core |
| `qtc_rates.cpp` | Integer issue rates of a card (timing only, nothing ships from it) |

## Build and check

    . ../../sycl/toolchain-open/envrc.sh      # or any DPC++: see ../../sycl/install-toolchain.sh
    ./build.sh --jit                          # writes ../build/quantus_sycl.node
    node ../verify-core.js --addon ../build/quantus_sycl.node --device 0
    node bench.js --addon ../build/quantus_sycl.node --device 0 --secs 20

`QUANTUS_SYCL_DEVICE=cpu` runs it on the SYCL CPU device (with Intel's OpenCL CPU runtime): all five
checks pass there too, at about 2 MH/s. Tuning settings are listed at the top of `quantus_sycl.cpp`.

## Runtime libraries

The addon links `libsycl.so.9` (open-source DPC++ v7.x) and nothing else beyond the C and C++
runtime. Its run path is `$ORIGIN:$ORIGIN/sycl`, so on Linux the runtime can sit beside it, or in a
`sycl` folder beside it. That second place is where a release already keeps the Pearl Intel core
(`pearl_core_sycl.node`) with the same runtime. Checked on the Arc rig with a staged build:

- the addon in a folder with the runtime beside it (`build.sh --jit --stage DIR`): loads;
- the addon alone in `native/` with the Pearl core's staged folder as `native/sycl`: loads;
- the addon alone, no runtime anywhere: does not load.

So on Linux the loader needs no change: `quantus_sycl.node` can ship in the places
`src/main/quantusCore.js` already looks (beside the CLI, `resources/native`), as long as the Pearl
Intel core's `sycl` folder ships beside it, built with the same DPC++ (`libsycl.so.9`). Windows has
no run path: a DLL is found beside the `.node` that loads it. A Windows build (not made yet) would
have to sit in the `sycl` folder with the runtime, and the loader would then need the folder layout
the Pearl Intel core uses (`native/sycl/quantus_sycl.node`).

## Speed on an Arc A750

See `earn/tuning/quantus-benchmark.md`. In short (13 Nov, card 0 of the rented rig): 106.7 MH/s
(verify-core 106.8) at 2,200 MHz and 190 W (the power limit), up from 103.6; SRBMiner 3.8.6 does
115-117 MH/s on the same card at 2,150-2,200 MHz and the same power.

What a multiply costs on this card: there is no 64-bit integer multiply. A 32 x 32 -> 64 product is
three instructions (`mul`, `mach`, a `mov` from the accumulator), a 64 x 64 product twelve, and the
field multiply with its reduction 37 to 39. Every integer instruction (add, add3, addc, mul, mach,
mov) issues at about one SIMD8 instruction a clock per EU (`qtc_rates.cpp`). Only about a fifth of
the time goes to the products; the rest is carries and reduction.

Kernel forms (`QUANTUS_SYCL_KERNEL=SG,GRF`), all at 190 W: sub-group 8 with 128 registers 102.7
(the default: each lane has twice the registers of sub-group 16 and nothing spills), sub-group 16
92.6-94.1, the 256-register forms about 82. Level with the default: the S-box outputs unfolded through
the linear layers (`QTC_LAZY 1`, 103.2). Lost: more than one counter a work-item (4: -4%, 16: -10%,
sub-group 16).

Two changes since, each raced against the old default run again as its twin (103.60-103.63 over
four runs, all at 2,200 MHz and 190 W):

- The fold's correction (a compare and two selects in each of about 1,700 folds a nonce) is inline
  vISA in the 8-lane kernels, with float-typed selects, so they run on the float pipe, which the
  kernel otherwise leaves idle (`QTC_ASM8`, `QTC_FCORR`): 3.7% fewer int-pipe instructions, 105.0-105.1.
  The other forms and the CPU device keep the plain C++ (a second copy of the field code, `qtc8`).
  IGC declares 32-bit asm operands signed: the compare reads them through unsigned aliases.
- A smaller kernel: one loop over all 16 full rounds of a nonce (`QTC_ONELOOP`), one copy of the 12
  S-boxes, 4,858 instructions in the kernel instead of 17,101, the same count executed: 106.7.
  One copy of the rounds for both permutations (`QTC_SHARE`) gave 106.4-106.6.

`qtc_rates.cpp`'s `fmix` row says the float and int pipes do not simply add up on this card (int
adds alone 7.5 a clock an EU; with a float mad beside each, 3.2 of each), so the selects' gain is
less than their share of the instructions.

A measuring trap: a kernel compiled inside a timed launch makes any launch-size rule read the card
as slow. The core compiles before its first launch for that reason; compare builds only after that.
