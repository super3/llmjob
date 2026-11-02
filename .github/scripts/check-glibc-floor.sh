#!/usr/bin/env bash
# Fails when a Linux addon needs a newer glibc or libstdc++ than an Ubuntu release has.
#
#   check-glibc-floor.sh [--ubuntu 20.04|22.04] <file.node>...
#
# A build can pick up a symbol that only a newer system has without anyone noticing. Under
# glibc 2.38's headers strtoul is __isoc23_strtoul and atoi calls __isoc23_strtol, so one
# strtoul in the RTX 20 launch sweep made v0.6.7's pearl_core.node need glibc 2.38: it would
# not load on Ubuntu 22.04 (2.35) or Debian 12 (2.36), and the app said only that the core was
# missing. v0.5.7 to v0.6.6 needed 2.34, built on Ubuntu 24.04, so they did not load on
# HiveOS's Ubuntu 20.04 image.
#
# The CUDA cores are built on Rocky Linux 8 (glibc 2.28) and held to Ubuntu 20.04's limits,
# so they load on every HiveOS image from 20.04 up. The AMD core is held to 22.04's: ROCm 7,
# which it needs, supports no older Ubuntu. 22.04 is the default.
set -euo pipefail

release=22.04
if [ "${1:-}" = --ubuntu ]; then
  release=${2:-}
  shift 2
fi
case $release in
  # glibc, its libstdc++ (GCC 10 on 20.04, GCC 12 on 22.04), and its libgcc_s.
  20.04) GLIBC_MAX=2.31 GLIBCXX_MAX=3.4.28 CXXABI_MAX=1.3.12 GCC_MAX=10.0.0 ;;
  22.04) GLIBC_MAX=2.35 GLIBCXX_MAX=3.4.30 CXXABI_MAX=1.3.13 GCC_MAX=12.0.0 ;;
  *) echo "::error::no limits for Ubuntu '$release' (use 20.04 or 22.04)"; exit 2 ;;
esac

# The newest of two dotted versions.
newest() { printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n 1; }

bad=0
for f in "$@"; do
  if ! syms=$(objdump -T "$f" 2>&1); then
    echo "::error::objdump could not read $f: $syms"
    bad=1
    continue
  fi
  for pair in GLIBC:$GLIBC_MAX GLIBCXX:$GLIBCXX_MAX CXXABI:$CXXABI_MAX GCC:$GCC_MAX; do
    name=${pair%%:*}
    max=${pair##*:}
    # Each version tag once, as name_x.y.z. GLIBC_ never matches GLIBCXX_: the underscore
    # must follow the name.
    tags=$(printf '%s\n' "$syms" | grep -oE "\b${name}_[0-9][0-9.]*" | sort -u || true)
    [ -n "$tags" ] || continue
    got=$(printf '%s\n' "$tags" | sed "s/^${name}_//" | sort -V | tail -n 1)
    if [ "$(newest "$got" "$max")" != "$max" ]; then
      echo "::error::$f needs ${name}_$got, newer than Ubuntu $release's ${name}_$max, so it would not load there. The symbols:"
      while read -r tag; do
        v=${tag#"${name}_"}
        [ "$(newest "$v" "$max")" = "$max" ] && continue
        printf '%s\n' "$syms" | grep -E "\b${tag}\b" | awk '{print "  " $NF " (" t ")"}' t="$tag"
      done <<< "$tags"
      bad=1
    else
      echo "$f: newest $name it needs is $got (Ubuntu $release has $max)"
    fi
  done
done
exit $bad
