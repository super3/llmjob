#!/bin/bash
# Build the NEB CUDA core on Linux: earn/native/neb/build/neb_cuda.node.
#
#   bash earn/native/neb/cuda/build.sh [arch list, default 75,80,86,89,90,120]
#
# Environment: CUDA_HOME (default /usr/local/cuda; 12.8 or newer for sm_120), NODE_INCLUDE (Node's include/node,
# default: beside the node on PATH), OUT (default ../build/neb_cuda.node), DEFS (extra -D flags). The CUDA runtime
# is linked statically, so the addon needs only the NVIDIA driver. N-API symbols come from the node or Electron
# process that loads it.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
NEB=$(dirname "$HERE")
ARCH=${1:-75,80,86,89,90,120}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NODE_INCLUDE=${NODE_INCLUDE:-$(dirname "$(dirname "$(command -v node)")")/include/node}
OUT=${OUT:-$NEB/build/neb_cuda.node}
[ -f "$NODE_INCLUDE/node_api.h" ] || { echo "no node_api.h in $NODE_INCLUDE (set NODE_INCLUDE)" >&2; exit 1; }
GEN=()
for a in ${ARCH//,/ }; do GEN+=("-gencode=arch=compute_$a,code=sm_$a"); done
GEN+=("-gencode=arch=compute_$a,code=compute_$a")  # PTX for cards newer than the list
mkdir -p "$(dirname "$OUT")"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
# -Xptxas -v: each kernel's register count, for the log.
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -Xcompiler -fPIC -Xptxas -v "${GEN[@]}" ${DEFS:-} -I "$NEB" -c "$HERE/neb_cuda.cu" -o "$T/neb_cuda.o"
for f in nebhash2 shares blake3_min chacha8; do
  gcc -O3 -fPIC -I "$NEB/ref" -c "$NEB/ref/$f.c" -o "$T/$f.o"
done
gcc -O2 -fPIC -I "$NODE_INCLUDE" -I "$NEB" -c "$NEB/neb_napi.c" -o "$T/neb_napi.o"
g++ -shared -o "$OUT" "$T/neb_cuda.o" "$T/neb_napi.o" "$T/nebhash2.o" "$T/shares.o" "$T/blake3_min.o" "$T/chacha8.o" \
  -L "$CUDA_HOME/lib64" -lcudart_static -lrt -lpthread -ldl
echo "built $OUT ($ARCH)"
