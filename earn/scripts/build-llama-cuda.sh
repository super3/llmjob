#!/usr/bin/env bash
# Build llama.cpp's llama-server for CUDA on Linux, as the archive the CLI
# downloads when an NVIDIA card has no Vulkan driver (shared/llama.js,
# needsCudaServer). llama.cpp publishes no Linux CUDA build, only Vulkan, and
# Salad's NVIDIA PCs run Linux containers under WSL2, which has a CUDA driver
# and no NVIDIA Vulkan one. There the Vulkan build ran Gemma on the CPU.
#
#   scripts/build-llama-cuda.sh OUT_DIR
#
# Writes OUT_DIR/<LLM.cudaServer.asset>: one folder holding llama-server and
# the CUDA runtime libraries it needs (cudart, cuBLAS, cuBLASLt), with its run
# path set to the folder. The driver's libcuda.so.1 is the one thing it takes
# from the system. Needs git, cmake, g++, make, curl, patchelf and a CUDA
# toolkit with cuBLAS (CUDA_HOME, default /usr/local/cuda). CI runs it in
# nvidia/cuda:12.8.1-devel-ubuntu22.04, so the binary needs glibc 2.35, which
# is what Salad's and HiveOS's Ubuntu 22.04 images have.
#
# CUDA_ARCHS overrides the GPUs it builds for (CMAKE_CUDA_ARCHITECTURES; a test
# box builds for its own card only, "86-real", to save time). The default is
# llama.cpp's own list for the toolkit.
set -euo pipefail

# The tag and the archive name are LLM.cudaServer in src/shared/config.js;
# test/llamaCuda.test.js checks they match.
TAG=b9902
ASSET=llama-b9902-bin-ubuntu-cuda-12.8-x64.tar.gz

out=${1:?usage: build-llama-cuda.sh OUT_DIR}
cuda=${CUDA_HOME:-/usr/local/cuda}
work=${WORK_DIR:-$(mktemp -d)}
jobs=${JOBS:-$(nproc)}
mkdir -p "$out"
out=$(cd "$out" && pwd)

rm -rf "$work/llama.cpp"
git clone -q --depth 1 -b "$TAG" https://github.com/ggml-org/llama.cpp "$work/llama.cpp"
cd "$work/llama.cpp"

# The server's web page, from the same release. The build embeds
# tools/ui/dist; without it, and without npm to build one, it fails.
mkdir -p tools/ui/dist
curl -fsSL --retry 3 "https://github.com/ggml-org/llama.cpp/releases/download/$TAG/llama-$TAG-ui.tar.gz" \
  | tar -xz -C tools/ui/dist --strip-components=1

archs=()
[ -n "${CUDA_ARCHS:-}" ] && archs=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS")
# ggml is linked into the binary (BUILD_SHARED_LIBS=OFF), so the folder holds
# one program and NVIDIA's libraries. libcuda comes from the toolkit's stub at
# link time and from the driver at run time. No OpenMP: llama.cpp's Linux
# builds need libgomp.so.1 for it, which a bare container lacks (the Vulkan
# build failed to start for that on a Salad box), and with every layer on the
# GPU its own thread pool does as well. No NCCL either: llama.cpp links it
# when the build machine has it (CI's CUDA image does), and the archive
# doesn't carry it. It only speeds up splitting a model across cards, and
# the miner gives the LLM one card (--split-mode none).
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF "${archs[@]}" \
  -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" -DCUDAToolkit_ROOT="$cuda" \
  -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$cuda/lib64/stubs"
cmake --build build --target llama-server -j "$jobs"

name=${ASSET%.tar.gz}
pkg=$work/$name
rm -rf "$pkg" && mkdir -p "$pkg"
cp build/bin/llama-server "$pkg/"
lib=$cuda/lib64
[ -d "$lib" ] || lib=$cuda/lib
for so in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
  cp -L "$lib/$so" "$pkg/"
done
# DT_RPATH, not RUNPATH: it also covers libcublas finding libcublasLt.
patchelf --force-rpath --set-rpath '$ORIGIN' "$pkg/llama-server" "$pkg"/libcublas*.so.12
cp LICENSE "$pkg/LICENSE-llama.cpp"

# Everything but the driver must resolve inside the folder.
missing=$(cd "$pkg" && ldd ./llama-server | grep 'not found' | grep -v 'libcuda\.so\.1' || true)
if [ -n "$missing" ]; then
  echo "llama-server needs a library the archive does not carry: $missing" >&2
  exit 1
fi
if ldd "$pkg/llama-server" | grep -E 'libcu(dart|blas|blasLt)' | grep -v "$pkg" >&2; then
  echo "llama-server loads a CUDA library from outside its folder" >&2
  exit 1
fi

tar -C "$work" -czf "$out/$ASSET" "$name"
ls -l "$out/$ASSET"
