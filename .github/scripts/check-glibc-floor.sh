#!/usr/bin/env bash
# Fails when a Linux addon needs a newer glibc or libstdc++ than Ubuntu 22.04 has.
#
#   check-glibc-floor.sh <file.node>...
#
# The cores are built on ubuntu-latest (Ubuntu 24.04, glibc 2.39), and a build there can pick
# up a symbol that only a newer system has without anyone noticing. Under glibc 2.38's headers
# strtoul is __isoc23_strtoul and atoi calls __isoc23_strtol, so one strtoul in the RTX 20
# launch sweep made v0.6.7's pearl_core.node need glibc 2.38: it would not load on Ubuntu 22.04
# (2.35) or Debian 12 (2.36), and the app said only that the core was missing. v0.5.7 to v0.6.6
# needed 2.34. The limits below are Ubuntu 22.04's, which ROCm 7.2 also supports.
set -euo pipefail

GLIBC_MAX=2.35       # Ubuntu 22.04's glibc
GLIBCXX_MAX=3.4.30   # its libstdc++ (GCC 12)
CXXABI_MAX=1.3.13
GCC_MAX=12.0.0       # its libgcc_s

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
      echo "::error::$f needs ${name}_$got, newer than Ubuntu 22.04's ${name}_$max, so it would not load there. The symbols:"
      while read -r tag; do
        v=${tag#"${name}_"}
        [ "$(newest "$v" "$max")" = "$max" ] && continue
        printf '%s\n' "$syms" | grep -E "\b${tag}\b" | awk '{print "  " $NF " (" t ")"}' t="$tag"
      done <<< "$tags"
      bad=1
    else
      echo "$f: newest $name it needs is $got (Ubuntu 22.04 has $max)"
    fi
  done
done
exit $bad
