#!/usr/bin/env bash
# Build the RandomX CPU worker (earn/xmr-worker) and check it.
#
#   scripts/build-xmr-worker.sh            build, then run its self-test
#   scripts/build-xmr-worker.sh --stage    also copy it to vendor/native/,
#                                          where the installer and the CLI
#                                          build pick it up
#
# Linux gives a static x86-64 binary, llmjob-xmr-worker. On Windows, run it
# from Git Bash with Visual Studio's CMake on PATH (CI does): it gives
# llmjob-xmr-worker.exe with the static MSVC runtime.
#
# RandomX is fetched from GitHub at the commit below and nothing else.
# RANDOMX_SOURCE_DIR points at an existing checkout instead (it must be at
# that commit), and BUILD_DIR moves the build out of xmr-worker/build.
set -euo pipefail

RANDOMX_REPO=https://github.com/tevador/RandomX
# master on 2026-09-24: v2.0.1 plus a v2 JIT fix (#340) and build fixes.
RANDOMX_COMMIT=7607fb2faed24d5a679e139a9828d194bbc644a4

# Git Bash paths (/d/a/...) as Windows programs need them (D:/a/...). A no-op
# everywhere else.
native() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
src=$here/xmr-worker
build=${BUILD_DIR:-$src/build}
rx=${RANDOMX_SOURCE_DIR:-$build/randomx-src}
version=$(node -p "require('$(native "$here")/package.json').version" 2>/dev/null || echo 0.0.0)

stage=0
for a in "$@"; do
  case $a in
    --stage) stage=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

if [[ ! -d $rx/.git ]]; then
  git clone --quiet "$RANDOMX_REPO" "$rx"
fi
if [[ $(git -C "$rx" rev-parse HEAD) != "$RANDOMX_COMMIT" ]]; then
  git -C "$rx" fetch --quiet origin "$RANDOMX_COMMIT" 2>/dev/null || git -C "$rx" fetch --quiet origin
  git -C "$rx" -c advice.detachedHead=false checkout --quiet "$RANDOMX_COMMIT"
fi
[[ $(git -C "$rx" rev-parse HEAD) == "$RANDOMX_COMMIT" ]] || { echo "RandomX is not at $RANDOMX_COMMIT" >&2; exit 1; }

cmake -S "$(native "$src")" -B "$(native "$build")" -DCMAKE_BUILD_TYPE=Release \
  -DRANDOMX_SOURCE_DIR="$(native "$rx")" -DRANDOMX_COMMIT="${RANDOMX_COMMIT:0:12}" -DXMR_WORKER_VERSION="$version"
cmake --build "$(native "$build")" --config Release --parallel "${JOBS:-4}" --target llmjob-xmr-worker

exe=$build/llmjob-xmr-worker
[[ -f $exe ]] || exe=$build/Release/llmjob-xmr-worker.exe
[[ -f $exe ]] || exe=$build/llmjob-xmr-worker.exe
[[ -f $exe ]] || { echo "the build produced no llmjob-xmr-worker" >&2; exit 1; }

# The known-answer check against RandomX's own test vectors. A build that
# hashes wrong must not ship.
"$exe" --self-test

if (( stage )); then
  mkdir -p "$here/vendor/native"
  cp "$exe" "$here/vendor/native/"
  echo "staged $(basename "$exe") in vendor/native/"
fi
echo "built $exe"
