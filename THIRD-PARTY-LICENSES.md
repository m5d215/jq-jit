# Third-Party Licenses

jq-jit is distributed under the [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) license (at your option). However, the compiled
binary statically links or dynamically links against third-party code whose
own licenses must be preserved when distributing the binary.

This file enumerates those third-party components and their licenses. Full
license texts for each component are available from the upstream source
repositories linked below, and (for Rust crates) inside each crate's
distribution on [crates.io](https://crates.io).

---

## System libraries (dynamically linked at runtime)

| Component | Version | License | Upstream |
|-----------|---------|---------|----------|
| [jq](https://github.com/jqlang/jq) | ≥ 1.7 | MIT | <https://github.com/jqlang/jq/blob/master/COPYING> |
| [Oniguruma (libonig)](https://github.com/kkos/oniguruma) | runtime dep of jq | BSD-2-Clause | <https://github.com/kkos/oniguruma/blob/master/COPYING> |

### jq

Copyright (C) 2012 Stephen Dolan.
Licensed under the MIT License. jq-jit's `src/jq_ffi.rs` and `src/bytecode.rs`
include Rust FFI declarations that mirror data structures from jq's public
headers (`jv.h`, `jq.h`) and internal bytecode layout. Those portions are
derivative works of jq and are redistributed under the MIT License.

### Oniguruma

Copyright (c) 2002-2024 K.Kosako.
Licensed under the BSD 2-Clause License. Oniguruma is used transitively via
jq's regex support; jq-jit does not call Oniguruma directly.

---

## Rust crate dependencies

The following crates are compiled into the jq-jit binary. Each crate's
license text is included in its source distribution on crates.io.

### Cranelift / Wasmtime (Apache-2.0 WITH LLVM-exception)

Copyright Bytecode Alliance contributors. Licensed under the Apache License,
Version 2.0, with the LLVM exception. See
<https://github.com/bytecodealliance/wasmtime/blob/main/LICENSE>.

- `cranelift-assembler-x64`, `cranelift-assembler-x64-meta`
- `cranelift-bforest`, `cranelift-bitset`
- `cranelift-codegen`, `cranelift-codegen-meta`, `cranelift-codegen-shared`
- `cranelift-control`, `cranelift-entity`, `cranelift-frontend`
- `cranelift-isle`, `cranelift-jit`, `cranelift-module`
- `cranelift-native`, `cranelift-srcgen`
- `regalloc2`
- `target-lexicon`
- `wasmtime-internal-core`, `wasmtime-internal-jit-icache-coherence`
- `ar_archive_writer`

### Dual-licensed MIT OR Apache-2.0

- `allocator-api2`, `anyhow`, `arbitrary`, `autocfg`
- `bumpalo`, `cc`, `cfg-if`, `chrono`, `core-foundation-sys`
- `equivalent`, `fast-float`, `find-msvc-tools`, `fnv`, `gimli`
- `hashbrown`, `heck`, `iana-time-zone`, `indexmap`, `itoa`
- `libc`, `log`, `memmap2`, `num-traits`, `object`, `psm`
- `regex`, `regex-automata`, `regex-syntax`
- `rustc-hash`, `rustversion`, `serde_core`, `serde_json`
- `shlex`, `smallvec`, `stacker`, `static_assertions`
- `bitflags` (MIT / Apache-2.0)

### MIT only

- `castaway`
- `compact_str`
- `libm`
- `libmimalloc-sys`, `mimalloc` (© Microsoft Corporation)
- `region`
- `zmij`

### Unlicense OR MIT

- `aho-corasick`
- `csv`, `csv-core`
- `memchr`

### Multi-license / other

- `mach2` — BSD-2-Clause OR MIT OR Apache-2.0
- `ryu` — Apache-2.0 OR BSL-1.0
- `foldhash` — Zlib

---

## Apache-2.0 attribution

Per Section 4 of the Apache License 2.0, this distribution notes that it
contains code originally distributed by the Bytecode Alliance (Cranelift /
Wasmtime) and other upstream authors listed above. No `NOTICE` file was
identified in the upstream crates as of this writing; if one is added
upstream, it will be mirrored here on the next update.

---

## How this file is maintained

The list above was generated from `cargo tree --prefix none --format "{p} {l}"`
against the `Cargo.lock` committed to this repository. When dependencies
change, regenerate the listing with:

```bash
cargo tree --prefix none --format "{p} | {l}" | sort -u
```

and update this file accordingly.
