#!/bin/bash
# Build pearl_core_fp8.node on a Linux box with a CUDA toolkit (12.8 or newer) and Node.
#   bash fp8/build.sh [extra nvcc flags, e.g. -DFP8_MMA_PLAIN]
# CI builds it the same way (native-core.yml, job "fp8"). Run from earn/native.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p cuda-build
[ -d node_modules/node-addon-api ] || npm install --prefix . --no-save --no-audit --no-fund node-addon-api node-gyp
nvcc -O3 -std=c++17 -cudart static -Xcompiler -fPIC -gencode arch=compute_120,code=sm_120 \
  -Xptxas -v "$@" -c pearl_fp8.cu -o cuda-build/pearl_fp8.o 2>&1 | tee cuda-build/ptxas.txt
ar rcs cuda-build/libpearl_fp8.a cuda-build/pearl_fp8.o
npx node-gyp rebuild
ls -l build/Release/pearl_core_fp8.node
