# Kernel probes

Standalone CUDA programs that establish a card's denominators. None needs the miner.
Run these before touching the kernel: every ratio in `OPTIMIZATION.md` and the 4090
tuning log is relative to that card's ceiling, and none of them transfer between
architectures.

```
nvcc -arch=sm_120 -O3 -o mmapeak mmapeak.cu && ./mmapeak 512
```

| Probe | Answers |
|---|---|
| `mmapeak.cu` | Pure `mma.m16n8k32.s8` from registers, no memory. Hardware ceiling + power. |

## Measured

| Card | Ceiling @512 thr | Power at ceiling | Miner | % of ceiling |
|---|---|---|---|---|
| RTX 4090 (sm_89) | 338 T-MAC/s | 156 W | 216.0 TH/s | 64% |
| RTX 5090 (sm_120) | **482.6 T-MAC/s** | **153 W @ 2400 MHz** | 138 TH/s | **28.6%** |

The 5090 reaches a 1.43x higher tensor ceiling on the same power as the 4090. The
shipped kernel meanwhile draws 575 W (its full cap) at 1170 MHz -- 38% of the 3090 MHz
max, with `SW Power Cap` active. Pure MMA does not do that, so the power is going to
the memory path, and the clock it costs is what puts a 5090 below a 4090.

## Traps (learned the hard way, see tuning log s6)

- **A store the compiler can prove dead deletes the whole workload.** The first version
  of `mmapeak` guarded its store with `if (threadIdx.x == 1024)`, which is never true;
  it reported 377,140 T-MAC/s in 0.00 ms. Store unconditionally.
- **Check `cudaGetLastError()` after the launch, not just after the sync.** A launch
  that fails on resources reports 0.00 ms rather than an error.
- **Check `nvidia-smi` compute processes first.** A miner or LLM sharing the card
  silently contaminates every reading.

## Power is the binding constraint on sm_120

The 5090 is hard power-capped running this kernel: steady state sits at exactly the
cap with `SW Power Cap: Active` and only 47 C, so there is no thermal headroom to
recover. Hashrate is bought with watts, almost perfectly linearly:

| power limit | TH/s | TH/s per W |
|---|---|---|
| 450 W | 117.7 | 0.262 |
| 525 W | 135.6 | 0.258 |
| 575 W | 145.4 | 0.253 |
| 600 W | 148.7 | 0.248 |

At ~0.25 TH/W, 300 TH/s would need 1200 W. **The only route to the goal is roughly
doubling MACs per joule** -- there is no clock left to find.

Where the power goes, per launch (49.08 ms, from Nsight):

| | |
|---|---|
| tensor instructions | 2,147,483,648 |
| ALU instructions | 6,305,857,536 (2.9 per MMA) |
| total instructions | 13,310,255,104 |
| L2 traffic | 104 GB -> 2.12 TB/s |
| DRAM traffic | 3.18 GB -> 65 GB/s (3.5% of peak) |
| shared loads | 412 GB -> 8.4 TB/s |

The tensor pipe is 69.41% active, essentially identical to Ada's 69.9%, and the
arithmetic closes exactly: 0.694 x 482 x (1095/2400) = 152.6 against 152.5 measured.
Nsight's percentages are per ACTIVE CYCLE, so they hide the clock -- the kernel is
issuing MMA just as densely as on the 4090 and losing purely on frequency.

Ruled out as the power sink: DRAM (3.5%), shared-memory bank conflicts (64,620 of
3.2e9 wavefronts), operand data toggling (a variant with 8 distinct operand sets
measured 156 W against 153 W for constant operands), and codegen fallbacks (sm_120
emits native LDGSTS and LDSM, and its integer op counts match sm_89).

### The memory clock experiment

Locking the memory clock exposes how the budget is split:

| memory clock | SM clock | TH/s |
|---|---|---|
| 14001 MHz | 1322 MHz | 157.9 |
| 7001 MHz | 1353 MHz | 158.4 |
| 810 MHz | **2659 MHz** | 99.7 |

At 810 MHz the SM clock DOUBLES inside the same 600 W -- the memory domain is worth
roughly half the power budget -- but hashrate collapses, because L2 is in that clock
domain and the fold pulls 2.12 TB/s through it. At 2659 MHz with bandwidth intact the
kernel would be around 368 TH/s. So **L2 traffic per MAC is the thing to attack**;
DRAM traffic is already irrelevant.
