#!/usr/bin/env bash
# Build the AMD Pearl core. Nothing here touches the CUDA build: it reads ../src and writes
# only to ./build.
#
#   ./build-amd.sh cpu              HIP-CPU: build/pearl_core_hipcpu.node and
#                                   build/pearl_amd_foldtest_cpu, which run the AMD kernels
#                                   on the CPU, and their _kwrap twins with the wrap readout
#                                   (RDNA 4's, which mines k 4096), and fold checks with the
#                                   RDNA 3 and 4 loops and dot4's other k-steps. For checking
#                                   without a GPU; never shipped.
#   ./build-amd.sh gpu [gfx...]     ROCm: build/pearl_core_hip.node and build/pearl_amd_foldtest
#                                   for the listed targets (default: $DEFAULT_ARCHS below).
#   ./build-amd.sh check [gfx...]   Compile the device code for each target and print each
#                                   fold's registers, scratch and LDS. Fails if a fold
#                                   spills or uses scratch memory, or, with no
#                                   PEARL_AMD_CFLAGS, if a gfx12 target lacks the wrap
#                                   readout (PEARL_AMD_KWRAP) or any other target has it, or
#                                   a target lacks the fold the host runs on it (folds_of).
#                                   On Windows no target has the wrap readout (KWRAP_OS),
#                                   the RDNA 4 cuts of 4 November or tile order, RDNA 3's
#                                   buffer loads or staging cuts (SET_OS), or the hash in
#                                   the fold (FUSE_OS).
#   ./build-amd.sh winlib [gfx...]  Windows: build/win/pearl_hip.lib, the kernels and host
#                                   for the listed targets (default: $WINDOWS_ARCHS below),
#                                   and build/pearl_amd_foldtest.exe. Then, in earn/native,
#                                   PEARL_BACKEND=hip npx node-gyp rebuild links the addon
#                                   (binding.gyp), as the CUDA core is linked.
#
# On Windows this runs in Git Bash with MSVC's environment (vcvars64, or the
# ilammy/msvc-dev-cmd action), and the compiler is the one in AMD's HIP SDK, found
# through HIP_PATH, which its installer sets. check works there too.
#
# Environment:
#   ROCM_PATH       ROCm install with hip-clang (default /opt/rocm). gpu and check.
#   HIP_PATH        Windows: the HIP SDK's folder, e.g. C:\Program Files\AMD\ROCm\7.2\.
#   HIP_CPU_PATH    a checkout of github.com/ROCm/HIP-CPU (its include/ is used). cpu.
#   TBB_ROOT        prefix holding include/tbb and lib/libtbb.so (default: the system's). cpu.
#   CPU_CXX         C++ compiler for the cpu build (default clang++; g++ ignores the vector
#                   types the folds use).
#   NODE_ADDON_API  node-addon-api's include folder (default: require('node-addon-api')).
#   NODE_INCLUDE    Node's headers (default: the running node's include/node).
#   PEARL_AMD_CFLAGS  extra compiler flags for gpu and check, e.g. -DPEARL_AMD_TILED=0 to
#                   build a tuning variant (pearl_amd_fold.h lists the knobs). Refused with
#                   PEARL_AMD_READOUT_DROP, PEARL_AMD_ABLATE or PEARL_AMD_STAMP in winlib, and in
#                   gpu under CI: they are timing-only.
#
# The portable kernels are pearl_kernel.cu's own: this script copies that file up to the
# start of the CUDA folds into build/gen/pearl_kernel_portable.cu, and stops if the line it
# cuts at is gone.
set -euo pipefail

windows() {
  case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) return 0 ;; *) return 1 ;; esac
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The compiler is a Windows program: give it C:/... paths, which Git Bash takes too.
if windows; then HERE="$(cygpath -m "$HERE")"; fi
SRC="$HERE/../src"
OUT="$HERE/build"
GEN="$OUT/gen"
DEFAULT_ARCHS="gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201"
# The Radeon targets only: AMD's Windows runtime has no Instinct (gfx9) cards.
WINDOWS_ARCHS="gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201"
MARKER='^// A 16-byte global->shared copy that does not pass through registers'
# The Windows core keeps today's readout on every target, so RDNA 4 mines k 2048 there. With the wrap
# readout the SDK's clang 21 put 24 bytes of the gfx12 fold in scratch memory (native-core.yml's
# windows-2022-hip, 1 November), and no Windows Radeon has run either readout: the +1.1% at k 4096
# was measured on Linux, built with ROCm's clang 22. It goes before PEARL_AMD_CFLAGS, which can still
# set the readout either way.
KWRAP_OS=
if windows; then KWRAP_OS=-DPEARL_AMD_KWRAP=0; fi
# The RDNA 4 cuts of 4 November (PEARL_AMD_RDNA4_SET in pearl_amd_fold.h) stay out of the Windows core too: they
# were timed only with ROCm's clang 22 on Linux, where one of them alone (CRANK) lost 1.8% to the scheduler, and
# the SDK's clang 21 has never built them. Also before PEARL_AMD_CFLAGS. So do RDNA 3's buffer loads
# (PEARL_AMD_RDNA3_BUFLD, 7 November) and its staging cuts (PEARL_AMD_RDNA3_SET, 7 November), for the same reason.
SET_OS=
if windows; then SET_OS="-DPEARL_AMD_RDNA4_SET=0 -DPEARL_AMD_RDNA3_BUFLD=0 -DPEARL_AMD_RDNA3_SET=0"; fi
# So does the RDNA 4 tile order of 7 November (PEARL_AMD_WALK): it was timed on Linux at k 4096 and 8192, and the
# Windows core mines k 2048.
if windows; then SET_OS="$SET_OS -DPEARL_AMD_WALK=0"; fi
# The hash in the fold (PEARL_AMD_FUSE, on in gfx1200 code since 6 November) likewise: timed only on Linux with
# ROCm's clang 22.
FUSE_OS=
if windows; then FUSE_OS=-DPEARL_AMD_FUSE=0; fi

mode="${1:-}"
shift || true
if windows; then archs="${*:-$WINDOWS_ARCHS}"; else archs="${*:-$DEFAULT_ARCHS}"; fi

# PEARL_AMD_READOUT_DROP and PEARL_AMD_ABLATE (pearl_amd_fold.h) build a core with wrong
# transcripts, to time the RDNA 4 readout or a part of the fold; PEARL_AMD_STAMP one whose folds
# store clock stamps. None goes in a build that can ship: winlib is the Windows package's, and a
# gpu build in CI is the Linux release's (native-core.yml). Their place is a gpu build on a test
# box, from test/sweep.sh.
case "${PEARL_AMD_CFLAGS:-}" in
  *PEARL_AMD_READOUT_DROP*)
    if [ "$mode" = winlib ] || { [ "$mode" = gpu ] && [ -n "${CI:-}${GITHUB_ACTIONS:-}" ]; }; then
      echo "build-amd.sh: readout-drop build is timing-only: PEARL_AMD_READOUT_DROP is refused in a $mode" >&2
      echo "  build that can ship (winlib, or gpu under CI). Build it with gpu on a test box (test/sweep.sh)." >&2
      exit 1
    fi
    ;;
  *PEARL_AMD_ABLATE*)
    if [ "$mode" = winlib ] || { [ "$mode" = gpu ] && [ -n "${CI:-}${GITHUB_ACTIONS:-}" ]; }; then
      echo "build-amd.sh: ablation build is timing-only: PEARL_AMD_ABLATE is refused in a $mode" >&2
      echo "  build that can ship (winlib, or gpu under CI). Build it with gpu on a test box (test/sweep.sh)." >&2
      exit 1
    fi
    ;;
esac
case "${PEARL_AMD_CFLAGS:-}" in
  *PEARL_AMD_STAMP*)
    if [ "$mode" = winlib ] || { [ "$mode" = gpu ] && [ -n "${CI:-}${GITHUB_ACTIONS:-}" ]; }; then
      echo "build-amd.sh: clock-stamp build is timing-only: PEARL_AMD_STAMP is refused in a $mode" >&2
      echo "  build that can ship (winlib, or gpu under CI). Build it with gpu on a test box (test/sweep.sh)." >&2
      exit 1
    fi
    ;;
esac

slice() {
  mkdir -p "$GEN"
  local line
  line="$(grep -n "$MARKER" "$SRC/pearl_kernel.cu" | head -n 1 | cut -d: -f1 || true)"
  if [ -z "$line" ]; then
    echo "build-amd.sh: the cut marker is gone from pearl_kernel.cu; update MARKER to the line" >&2
    echo "  that starts the CUDA folds, so the AMD build takes everything before it." >&2
    exit 1
  fi
  head -n $((line - 1)) "$SRC/pearl_kernel.cu" > "$GEN/pearl_kernel_portable.cu"
}

node_includes() {
  local api="${NODE_ADDON_API:-}" inc="${NODE_INCLUDE:-}"
  [ -n "$api" ] || api="$(node -p "require('node-addon-api').include" 2>/dev/null | tr -d '"' || true)"
  [ -n "$inc" ] || inc="$(node -p "require('path').join(process.execPath, '..', '..', 'include', 'node')")"
  if [ -z "$api" ] || [ ! -f "$api/napi.h" ]; then
    echo "build-amd.sh: node-addon-api not found: run 'npm install --no-save node-addon-api' in" >&2
    echo "  earn/native/amd, or set NODE_ADDON_API to its include folder." >&2
    exit 1
  fi
  if [ ! -f "$inc/node_api.h" ]; then
    echo "build-amd.sh: Node headers not found at $inc: set NODE_INCLUDE." >&2
    exit 1
  fi
  echo "-I$api -I$inc"
}

rocm_root() {
  if windows; then
    if [ -z "${HIP_PATH:-}" ]; then
      echo "build-amd.sh: HIP_PATH is not set: install AMD's HIP SDK, or set HIP_PATH to its folder." >&2
      exit 1
    fi
    cygpath -m "${HIP_PATH%\\}"
  else
    echo "${ROCM_PATH:-/opt/rocm}"
  fi
}

hip_clang() {
  local rocm cc
  rocm="$(rocm_root)"
  # The HIP SDK keeps clang in bin\; ROCm on Linux in lib/llvm/bin.
  if windows; then cc="$rocm/bin/clang++.exe"; else cc="$rocm/lib/llvm/bin/clang++"; fi
  if [ ! -x "$cc" ]; then
    echo "build-amd.sh: no hip-clang at $cc: set ROCM_PATH (HIP_PATH on Windows) to a ROCm install." >&2
    exit 1
  fi
  echo "$cc"
}

offload_flags() {
  local f=""
  for a in $archs; do f="$f --offload-arch=$a"; done
  echo "$f"
}

# The fold a target's code object must have a body for, besides the reference fold: the one the host
# runs on it (pearl_host_hip.cpp's native_fold). check fails a release build that lacks it.
folds_of() {
  case "$1" in
    gfx908 | gfx90a) echo mfma8 ;;
    gfx942 | gfx950) echo mfma16 ;;
    gfx103*) echo dot4 ;;
    gfx11*) echo wmma11 ;;
    gfx12*) echo wmma12 ;;
  esac
}

case "$mode" in
  cpu)
    slice
    : "${HIP_CPU_PATH:?set HIP_CPU_PATH to a HIP-CPU checkout}"
    cxx="${CPU_CXX:-clang++}"
    tbb_inc="" tbb_lib=""
    if [ -n "${TBB_ROOT:-}" ]; then
      tbb_inc="-I$TBB_ROOT/include"
      tbb_dir="$TBB_ROOT/lib"
      [ -f "$tbb_dir/libtbb.so" ] || tbb_dir="$TBB_ROOT/lib/x86_64-linux-gnu"
      tbb_lib="-L$tbb_dir -Wl,-rpath,$tbb_dir"
    fi
    mkdir -p "$OUT/cpu"
    inc="-I$HERE/compat/cpu -I$HERE/src -I$SRC -I$GEN -I$HIP_CPU_PATH/include $tbb_inc"
    # -Wno-psabi: the dot4 fold passes 32-byte vectors by value between inlined functions, and clang
    # notes that without AVX that changes the x86 ABI (3 warnings). Nothing crosses a library boundary.
    flags="-std=c++17 -O2 -fPIC -DNDEBUG -Wno-psabi"
    echo "cpu: kernels and host"
    $cxx $flags $inc -c "$HERE/src/pearl_host_hip.cpp" -o "$OUT/cpu/pearl_host_hip.o"
    echo "cpu: addon"
    $cxx $flags $(node_includes) -I"$SRC" -DNAPI_DISABLE_CPP_EXCEPTIONS -c "$SRC/pearl_core.cc" -o "$OUT/cpu/pearl_core.o"
    $cxx -shared -o "$OUT/pearl_core_hipcpu.node" "$OUT/cpu/pearl_host_hip.o" "$OUT/cpu/pearl_core.o" $tbb_lib -ltbb -lpthread
    echo "cpu: fold check"
    $cxx $flags $inc "$HERE/test/pearl_amd_foldtest.cpp" -o "$OUT/pearl_amd_foldtest_cpu" $tbb_lib -ltbb -lpthread
    # The RDNA 3 and RDNA 4 defaults (PF2, and PF2 + PFX + TID) are on only in a real gfx11 or
    # gfx12 device pass, so the build above runs the one-slab loop. These run their loops too.
    # The DPP readout can't be emulated: a new card's first check is still the foldtest on it.
    echo "cpu: fold check with the RDNA 3 and RDNA 4 loops"
    $cxx $flags -DPEARL_AMD_PF2=1 $inc "$HERE/test/pearl_amd_foldtest.cpp" -o "$OUT/pearl_amd_foldtest_cpu_pf2" \
      $tbb_lib -ltbb -lpthread
    $cxx $flags -DPEARL_AMD_PF2=1 -DPEARL_AMD_PFX=1 -DPEARL_AMD_TID=1 $inc "$HERE/test/pearl_amd_foldtest.cpp" \
      -o "$OUT/pearl_amd_foldtest_cpu_pfx" $tbb_lib -ltbb -lpthread
    # The dot4 fold's k-step (PEARL_AMD_DOT4_K in pearl_amd_fold.h) at the two sizes the release doesn't use,
    # so a tuning build of either is checked too.
    echo "cpu: fold check with dot4's other k-steps"
    for dk in 8 16; do
      $cxx $flags -DPEARL_AMD_DOT4_K=$dk $inc "$HERE/test/pearl_amd_foldtest.cpp" -o "$OUT/pearl_amd_foldtest_cpu_dot4k$dk" \
        $tbb_lib -ltbb -lpthread
    done
    # The wrap readout is on only in a real gfx12 device pass (RDNA 4 mines k 4096), so the builds above
    # have today's. These have it, so the k 4096 path runs without a GPU: the core picks k 4096 when the
    # miner lets it (cardK), as it does on an RDNA 4 card.
    echo "cpu: kernels, host, addon and fold check with the wrap readout"
    $cxx $flags -DPEARL_AMD_KWRAP=1 $inc -c "$HERE/src/pearl_host_hip.cpp" -o "$OUT/cpu/pearl_host_hip_kwrap.o"
    $cxx -shared -o "$OUT/pearl_core_hipcpu_kwrap.node" "$OUT/cpu/pearl_host_hip_kwrap.o" "$OUT/cpu/pearl_core.o" \
      $tbb_lib -ltbb -lpthread
    $cxx $flags -DPEARL_AMD_KWRAP=1 $inc "$HERE/test/pearl_amd_foldtest.cpp" -o "$OUT/pearl_amd_foldtest_cpu_kwrap" \
      $tbb_lib -ltbb -lpthread
    echo "built $OUT/pearl_core_hipcpu{,_kwrap}.node and $OUT/pearl_amd_foldtest_cpu{,_pf2,_pfx,_kwrap,_dot4k8,_dot4k16}"
    ;;
  gpu)
    slice
    cc="$(hip_clang)"
    rocm="${ROCM_PATH:-/opt/rocm}"
    mkdir -p "$OUT/gpu"
    inc="-I$HERE/compat/gpu -I$HERE/src -I$SRC -I$GEN"
    # ROCm marks every HIP call [[nodiscard]]. The host ignores the result of the ones that
    # cannot fail on their own, as pearl_host.cu does, and reads hipGetLastError where it
    # matters.
    flags="-std=c++17 -O3 -fPIC -Wno-unused-result -Wno-unused-value ${PEARL_AMD_CFLAGS:-}"
    # The host names the targets in its errors (a card the build has no code for).
    targets="-DPEARL_AMD_TARGETS=\"$(echo $archs | tr ' ' ',')\""
    echo "gpu: kernels and host for $archs"
    $cc -x hip $(offload_flags) --rocm-path="$rocm" $flags $targets $inc -c "$HERE/src/pearl_host_hip.cpp" -o "$OUT/gpu/pearl_host_hip.o"
    echo "gpu: addon"
    $cc $flags $(node_includes) -I"$SRC" -DNAPI_DISABLE_CPP_EXCEPTIONS -c "$SRC/pearl_core.cc" -o "$OUT/gpu/pearl_core.o"
    # --allow-shlib-undefined: libamdhip64 needs the rest of the ROCm runtime, which a build
    # box without a GPU may not have. The user's machine has it (README.md).
    $cc -shared -fPIC -o "$OUT/pearl_core_hip.node" "$OUT/gpu/pearl_host_hip.o" "$OUT/gpu/pearl_core.o" \
      -L"$rocm/lib" -lamdhip64 -Wl,-rpath,/opt/rocm/lib -Wl,--allow-shlib-undefined
    echo "gpu: fold check"
    $cc -x hip $(offload_flags) --rocm-path="$rocm" $flags $inc -c "$HERE/test/pearl_amd_foldtest.cpp" -o "$OUT/gpu/pearl_amd_foldtest.o"
    $cc -o "$OUT/pearl_amd_foldtest" "$OUT/gpu/pearl_amd_foldtest.o" \
      -L"$rocm/lib" -lamdhip64 -Wl,-rpath,/opt/rocm/lib -Wl,--allow-shlib-undefined
    echo "built $OUT/pearl_core_hip.node and $OUT/pearl_amd_foldtest for: $archs"
    ;;
  winlib)
    if ! windows; then
      echo "build-amd.sh: winlib is the Windows build; on Linux use gpu." >&2
      exit 2
    fi
    slice
    cc="$(hip_clang)"
    rocm="$(rocm_root)"
    mkdir -p "$OUT/win"
    inc="-I$HERE/compat/gpu -I$HERE/src -I$SRC -I$GEN"
    # The gpu build's flags, less -fPIC, and with the static C runtime: node-gyp builds the
    # rest of the addon with /MT, and both halves must use the same one. MSVC's headers warn
    # about getenv and friends unless told not to.
    flags="-std=c++17 -O3 -fms-runtime-lib=static -D_CRT_SECURE_NO_WARNINGS -Wno-unused-result -Wno-unused-value $KWRAP_OS $SET_OS $FUSE_OS ${PEARL_AMD_CFLAGS:-}"
    targets="-DPEARL_AMD_TARGETS=\"$(echo $archs | tr ' ' ',')\""
    echo "winlib: kernels and host for $archs"
    "$cc" -x hip $(offload_flags) --rocm-path="$rocm" $flags $targets $inc -c "$HERE/src/pearl_host_hip.cpp" \
      -o "$OUT/win/pearl_host_hip.obj"
    # MSVC's librarian, since MSVC's linker is the one that reads it (node-gyp).
    lib -nologo -out:"$OUT/win/pearl_hip.lib" "$OUT/win/pearl_host_hip.obj"
    echo "winlib: fold check"
    "$cc" -x hip $(offload_flags) --rocm-path="$rocm" $flags $inc -c "$HERE/test/pearl_amd_foldtest.cpp" \
      -o "$OUT/win/pearl_amd_foldtest.obj"
    "$cc" -fms-runtime-lib=static -o "$OUT/pearl_amd_foldtest.exe" "$OUT/win/pearl_amd_foldtest.obj" \
      -L"$rocm/lib" -lamdhip64
    echo "built $OUT/win/pearl_hip.lib and $OUT/pearl_amd_foldtest.exe for: $archs"
    echo "next, in earn/native: PEARL_BACKEND=hip npx node-gyp rebuild"
    ;;
  check)
    slice
    cc="$(hip_clang)"
    rocm="$(rocm_root)"
    mkdir -p "$OUT/check"
    inc="-I$HERE/compat/gpu -I$HERE/src -I$SRC -I$GEN"
    bad=0 wrapbad=0 foldbad=0
    where="on exactly the gfx12 targets"
    [ -n "$KWRAP_OS" ] && where="on no target (the Windows core mines k 2048)"
    printf '%-9s %-22s %6s %6s %8s %7s %8s %10s %5s\n' target kernel VGPRs AGPRs scratch spills LDS "waves/SIMD" wrap
    for a in $archs; do
      log="$OUT/check/$a.txt"
      if ! "$cc" -x hip --offload-arch="$a" --rocm-path="$rocm" -std=c++17 -O3 $KWRAP_OS $SET_OS $FUSE_OS ${PEARL_AMD_CFLAGS:-} $inc --cuda-device-only \
          -c "$HERE/src/pearl_host_hip.cpp" -o "$OUT/check/$a.o" -Rpass-analysis=kernel-resource-usage > "$log" 2>&1; then
        echo "$a: compile failed, see $log" >&2
        bad=1
        continue
      fi
      # The wrap readout (PEARL_AMD_KWRAP), as this target's device pass sets it: on in every gfx12
      # code object, since RDNA 4 mines k 4096, and in no other, which mine k 2048 with today's
      # readout; on Windows in none (KWRAP_OS). The release builds with no PEARL_AMD_CFLAGS, and
      # then a target that differs fails the check. A tuning build may set it either way, so there
      # it is only shown. tr: clang on Windows can end the lines with CR LF, and "1\r" is not 1.
      kw=$("$cc" -x hip --offload-arch="$a" --rocm-path="$rocm" -std=c++17 $KWRAP_OS ${PEARL_AMD_CFLAGS:-} $inc --cuda-device-only \
          -E -dM "$HERE/src/pearl_host_hip.cpp" 2>> "$log" | tr -d '\r' | sed -n 's/^#define PEARL_AMD_KWRAP //p' \
          | head -n 1 || true)
      case "$a" in gfx12*) want=1 ;; *) want=0 ;; esac
      [ -n "$KWRAP_OS" ] && want=0
      if [ -z "${PEARL_AMD_CFLAGS:-}" ] && [ "$kw" != "$want" ]; then
        if [ -n "$KWRAP_OS" ]; then
          echo "$a: the wrap readout (PEARL_AMD_KWRAP) is ${kw:-not set}, not 0: the Windows core" \
            "mines k 2048 on every card" >&2
        else
          echo "$a: the wrap readout (PEARL_AMD_KWRAP) is ${kw:-not set}, not $want: RDNA 4 (gfx12)" \
            "has it and mines k 4096, every other target mines k 2048 without it" >&2
        fi
        wrapbad=1
      fi
      # One row a kernel: the fold kernels with a body on this target, the hash and the
      # reference fold. A fold without a body here shows 0 VGPRs and is left out. In a release
      # build, the fold folds_of names without one fails the check (awk exit bit 2; bit 1: a spill).
      need=""
      [ -z "${PEARL_AMD_CFLAGS:-}" ] && need="$(folds_of "$a")"
      awk -v arch="$a" -v kw="${kw:-?}" -v need="$need" '
        /remark: Function Name: / { name = $(NF-1) }
        /remark:     VGPRs: / { v[name] = $(NF-1) }
        /remark:     AGPRs: / { g[name] = $(NF-1) }
        /remark:     ScratchSize/ { s[name] = $(NF-1) }
        /remark:     VGPRs Spill: / { sp[name] = $(NF-1) }
        /remark:     SGPRs Spill: / { ss[name] = $(NF-1) }
        /remark:     Occupancy/ { o[name] = $(NF-1) }
        /remark:     LDS Size/ { l[name] = $(NF-1) }
        END {
          for (n in v) {
            if (n !~ /^pearl_amd_(fold|hash)/ || v[n] == 0) continue
            printf "%-9s %-22s %6s %6s %8s %7s %8s %10s %5s\n", arch, n, v[n], (n in g ? g[n] : "-"), s[n], sp[n] + ss[n], l[n], o[n], kw
            if (s[n] + 0 > 0 || sp[n] + ss[n] > 0) bad = 1
          }
          nf = split(need, f, " ")
          for (i = 1; i <= nf; i++)
            if (!(("pearl_amd_fold_" f[i]) in v) || v["pearl_amd_fold_" f[i]] == 0) {
              print arch ": no body for pearl_amd_fold_" f[i] > "/dev/stderr"
              bad += 2
              break
            }
          exit bad
        }' "$log" | sort || { rc=$?; [ $((rc & 1)) -eq 0 ] || bad=1; [ $((rc & 2)) -eq 0 ] || foldbad=1; }
    done
    if [ "$bad" -ne 0 ] || [ "$wrapbad" -ne 0 ] || [ "$foldbad" -ne 0 ]; then
      [ "$bad" -ne 0 ] && echo "check: a fold failed to compile, spills, or uses scratch memory" >&2
      [ "$wrapbad" -ne 0 ] && echo "check: the wrap readout should be $where" >&2
      [ "$foldbad" -ne 0 ] && echo "check: a target lacks the fold the host runs on it (folds_of in build-amd.sh)" >&2
      exit 1
    fi
    echo "check: every fold compiled for $archs with no spills and no scratch$([ -z "${PEARL_AMD_CFLAGS:-}" ] \
      && echo ", every target has its fold, and the wrap readout $where")"
    ;;
  *)
    sed -n '2,/^# The portable kernels/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
