"""build75.py: the RTX 20 (sm_75) scan kernel with hand-written partial rounds, for a trial on a card.

  python build75.py <nvcc> <nvdisasm> <out dir> [--opts "key=value ..."] [--defs "-D..."]

  1. nvcc -cubin -arch=sm_75 -DQC_HAND75=1 qc_host.cu -> <out dir>/qc_sm75_raw.cubin (right as compiled)
  2. patch75.py with --opts                           -> <out dir>/qc_sm75.cubin.unverified

The result runs through QUANTUS_CUDA_CUBIN_FILE: the core checks it against its compiled kernel at startup and
falls back to that on any difference. Nothing here is embedded in the core yet.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nvcc')
    ap.add_argument('nvdisasm')
    ap.add_argument('out')
    ap.add_argument('--opts', default='')
    ap.add_argument('--defs', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.out, 'qc_sm75_raw.cubin')
    cmd = [a.nvcc, '-cubin', '-O3', '-std=c++17', '-arch=sm_75', '-DQC_HAND75=1'] + a.defs.split() + \
          ['-I', CUDA_DIR, '-o', raw, os.path.join(CUDA_DIR, 'qc_host.cu')]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        sys.exit('nvcc failed:\n' + r.stdout[-2000:])
    os.environ['NVDISASM'] = a.nvdisasm
    import patch75
    out = os.path.join(a.out, 'qc_sm75.cubin.unverified')
    opts = {k: int(v) for k, v in (x.split('=') for x in a.opts.split())}
    texts, stalls, bl = patch75.run(raw, out, **opts)
    tmp = out + '.iface.json'
    if os.path.exists(tmp):
        os.remove(tmp)
    loop = [s for s, b in zip(stalls, bl) if b == 'loop']
    print('sm_75 hand kernel: %d instructions, loop %d, %d cycles a trip for one warp; opts %s; wrote %s'
          % (len(texts), len(loop), sum(max(s, patch75.OPT['stall']) for s in loop), a.opts or '-', out))


if __name__ == '__main__':
    main()
