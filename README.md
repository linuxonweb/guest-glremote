# guest-glremote (linuxonweb)

GL bridge for the Firefox wasm guest: `glremote.c` / `glremote.h` and the
`ops.def` operation table, plus a host-side checker (`check-ops.mjs`,
`swizzle-check.mjs`) and a native test (`glrtest.c`).

Build for the guest with `build.sh OUTDIR` from the linuxonweb build recipe
(`firefox/linuxonweb-build/release/build-firefox-release.sh` calls it).

Pinned to linuxonweb project commit 3de18d3.
