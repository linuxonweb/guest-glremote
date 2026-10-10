#!/bin/bash
# Builds glremote.o and the glrtest guest check (wasm32 musl, release, -g0).
# usage: build.sh OUT_DIR
set -euo pipefail
OUT=$1; mkdir -p "$OUT/obj"
export TMPDIR=$(dirname "$OUT")/t TMP=$(dirname "$OUT")/t TEMP=$(dirname "$OUT")/t; mkdir -p "$TMPDIR"
W=/nix/store/j8mdl69qafyijllv7s703219nb5fnrdp-wasm32-unknown-linux-musl-llvm-toolchain-wrapper-22.1.8/bin
MESA=${MESA_INCLUDE:-/mnt/p6/build/glgears-kopper-20261006/mesa-26.2.3/include}
HERE=$(cd "$(dirname "$0")" && pwd)
CFLAGS="-std=c11 -O2 -g0 -Wall -Wextra -Werror -pthread -matomics -mbulk-memory -ffunction-sections -fdata-sections -I$MESA"
node "$HERE/check-ops.mjs"
# SIMD (required by the host page anyway) for the frame compare and swizzle.
$W/wasm32-unknown-linux-musl-clang $CFLAGS -msimd128 -c "$HERE/glremote.c" -o "$OUT/obj/glremote.o"
$W/wasm32-unknown-linux-musl-clang $CFLAGS -Wno-unused-result -c "$HERE/glrtest.c" -o "$OUT/obj/glrtest.o"
# No -O at link: the clang driver would run wasm-opt implicitly.
$W/wasm32-unknown-linux-musl-clang -pthread -matomics -mbulk-memory -static "$OUT/obj/glrtest.o" "$OUT/obj/glremote.o" -lm \
  -Wl,--gc-sections,--import-memory,--export-table,--shared-memory \
  -Wl,--initial-memory=33554432,--max-memory=4294967296,-z,stack-size=1048576 -o "$OUT/glrtest.wasm"
ls -la "$OUT/glrtest.wasm"
