# Phase 1 Blocker: Design Decisions

> Phase 0 research findings ([jq-bytecode.md](jq-bytecode.md), [backtracking.md](backtracking.md), [cranelift.md](cranelift.md), [existing-impls.md](existing-impls.md), [jit-backends.md](jit-backends.md)) to inform the five design decisions identified in [plan/roadmap.md](../plan/roadmap.md).

---

## 1. Bytecode Acquisition Strategy

How does the JIT obtain jq bytecode to compile?

### Options

#### (a) Subprocess + `--debug-dump-disasm` parsing

Invoke `jq --debug-dump-disasm '<filter>' < /dev/null` and parse the text output.

| Aspect | Assessment |
|---|---|
| Pros | No build-time dependency on jq internals. Easy to prototype |
| Cons | Output format is undocumented and fragile. `dump_operation()` only prints opcode name + immediate -- **constant pool values, subfunction trees, closure metadata, and variable binding info are incomplete or absent**. Debug format has changed across jq versions |
| Impl difficulty | Low (initial), High (completeness) |

The disassembly output (jq-bytecode.md, section 3) lacks:
- The full `struct bytecode` tree structure (subfunctions, parent pointers)
- `nclosures` / `nlocals` counts
- Closure reference resolution details
- `CALL_JQ` / `TAIL_CALL_JQ` immediate encoding (nargs, level, fidx)

These are all required for correct JIT compilation. Reconstructing them from text output is error-prone and version-dependent.

#### (b) Link `libjq` and read `struct bytecode` directly via C FFI

Use jq's compile pipeline (`jq_compile()`) to produce a `struct bytecode*`, then walk the tree from Rust via FFI.

| Aspect | Assessment |
|---|---|
| Pros | **Complete access** to all bytecode metadata: constants, subfunctions, closures, locals, debug info. Same compilation result as jq itself -- guaranteed semantic equivalence |
| Cons | Requires C FFI (`bindgen` or manual). Must link against jq's build artifacts (`libjq.a` / `libjq.so`). jq's build system uses autotools, which complicates cross-compilation |
| Impl difficulty | Medium |

Key implementation considerations:
- `struct bytecode` is defined in `src/bytecode.h` (jq-bytecode.md, section 2.5) -- the struct is relatively stable
- `jv` type (16 bytes, tagged union) needs careful FFI mapping
- jq's `jq_compile_args()` is the public entry point; returns the bytecode tree root
- `bindgen` can auto-generate bindings from jq headers

#### (c) Custom parser + compiler in Rust

Write a jq parser (from `parser.y`) and compiler (from `compile.c`) in pure Rust, producing a custom IR optimized for JIT.

| Aspect | Assessment |
|---|---|
| Pros | Full control over IR design. No C dependencies. Can produce JIT-friendly IR directly (e.g., SSA or CPS form). Long-term maintainability |
| Cons | **Massive development cost.** jq's grammar is non-trivial (operator precedence, string interpolation, `as` patterns, `try-catch`, `?//`, `@format`). Compatibility with jq is hard to guarantee and harder to maintain as jq evolves |
| Impl difficulty | Very High |

jaq took this approach and achieved good results, but jaq is a full language reimplementation, not a JIT for existing jq. The compatibility gap is significant -- jaq deliberately deviates from jq semantics in several areas (e.g., `nan` handling, `limit` behavior, error propagation).

### Recommendation: (b) `libjq` FFI

**Rationale:**

1. **Completeness**: Only option that provides the full `struct bytecode` tree without reconstruction
2. **Semantic fidelity**: Bytecode is produced by jq's own compiler -- no compatibility risk
3. **Phase 1 focus**: The goal is to validate the "bytecode -> native code" pipeline. Investing in a custom parser/compiler is premature
4. **Escape hatch**: If `libjq` integration proves too burdensome, option (c) can be pursued later -- but only after the JIT pipeline is proven

**Phase 1 approach:**

1. Use `bindgen` to generate Rust bindings for `bytecode.h` and `jv.h`
2. Write a safe Rust wrapper around `struct bytecode` traversal
3. Link against a system-installed or vendored `libjq`
4. Accept the autotools dependency for now; consider a `cc`-based build or `cmake` port later

**What to verify in Phase 1:**

- Can `bindgen` handle jq's headers cleanly? (jq uses some macro-heavy patterns in `opcode_list.h`)
- Is `struct bytecode` layout stable across jq 1.7.x releases?
- What is the overhead of the compile step relative to JIT compilation?

---

## 2. Value Representation

How are jq values represented in JIT-compiled code?

### Options

#### (a) 16-byte enum (jaq-style)

```rust
enum Val {
    Null,
    Bool(bool),
    Num(f64),       // or custom Num type
    Str(Rc<...>),
    Arr(Rc<Vec<Val>>),
    Obj(Rc<Map<...>>),
}
```

| Aspect | Assessment |
|---|---|
| JIT optimization | Tag check + branch for each operation. Cranelift can generate efficient `brif` chains. **Number operations require unboxing (load f64 from enum payload)**. Not ideal for tight arithmetic loops |
| Memory | 16 bytes per value (tag + 8-byte payload padded). Same as jq and jaq |
| Ease of impl | **Highest.** Idiomatic Rust. No unsafe code for the value type itself |

jaq uses this approach and achieves 2-5x speedup over jq on most benchmarks (existing-impls.md, section 1). The overhead is primarily in dynamic dispatch (`Box<dyn Iterator>`), not in the value representation.

#### (b) NaN boxing (64-bit, 1 word)

Encode all values in a single 64-bit word by exploiting IEEE 754 NaN payload bits:
- Quiet NaN with specific bit patterns encodes null, bool, pointers to heap objects
- Non-NaN bit patterns are raw `f64` values

| Aspect | Assessment |
|---|---|
| JIT optimization | **Best for number-heavy workloads.** Numbers are already f64 -- no unboxing needed. Cranelift operates directly on the 64-bit value. Type checks are bitwise operations (`and`, `cmp`) |
| Memory | **8 bytes per value** -- 50% reduction vs 16-byte representations. Better cache utilization. Stack slots are half the size |
| Ease of impl | **Medium.** Requires careful bit manipulation. All heap pointers must fit in 48 bits (true on current x86_64 and AArch64). Debugging is harder (values look like NaN in debuggers) |

Payload encoding scheme (example):
```
f64 value:         raw IEEE 754 bits (if not NaN)
null:              0x7FF8_0000_0000_0001
true:              0x7FF8_0000_0000_0003
false:             0x7FF8_0000_0000_0002
string ptr:        0x7FFA_xxxx_xxxx_xxxx  (48-bit pointer)
array ptr:         0x7FFB_xxxx_xxxx_xxxx
object ptr:        0x7FFC_xxxx_xxxx_xxxx
```

Used by: LuaJIT, SpiderMonkey, JSC (JavaScriptCore).

#### (c) Tagged pointer (64-bit)

Use the low 3 bits of a 64-bit word as a type tag (exploiting pointer alignment). Integers are stored shifted, pointers are stored with the tag in the low bits.

| Aspect | Assessment |
|---|---|
| JIT optimization | Good for integer-heavy workloads. **Does not help with f64 (jq's default number type is double)**. Tag extraction is cheap (mask + shift), but f64 requires heap allocation or a separate "boxed float" path |
| Memory | 8 bytes per value, but f64 requires heap boxing (16+ bytes total for floats) |
| Ease of impl | Medium. Common in Lisp/Scheme/Ruby implementations |

This is suboptimal for jq because jq's number type is `double` (f64). Tagged pointers excel when the primary numeric type is integer (e.g., Ruby's Fixnum), but jq uses f64 pervasively.

#### (d) Use jq's `jv` via FFI

Pass `jv` values between JIT code and the jq runtime. JIT-compiled functions operate on `jv` directly.

| Aspect | Assessment |
|---|---|
| JIT optimization | **Minimal.** Every operation goes through `jv_*` C functions. The JIT eliminates dispatch overhead but cannot optimize value operations themselves |
| Memory | 16 bytes (jq's native format). No conversion overhead |
| Ease of impl | **Lowest.** All value operations delegate to jq's C runtime. No need to reimplement string/array/object |

### Recommendation: (a) 16-byte enum for Phase 1, with (b) NaN boxing as Phase 2+ upgrade path

**Rationale:**

1. **Phase 1 goal is pipeline validation**, not peak performance. The 16-byte enum is the simplest correct implementation
2. **jaq has proven** that a 16-byte enum achieves excellent performance (2-5x over jq) without NaN boxing
3. **NaN boxing's benefit is concentrated in tight numeric loops** -- these are not the dominant jq use case (JSON transformation is)
4. **Upgrade path is clean**: the JIT's internal value representation can be swapped without changing the bytecode-to-IR translation logic, because the IR operates on abstract "Value" types

**Phase 1 approach:**

```rust
#[repr(C)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(Rc<String>),
    Arr(Rc<Vec<Value>>),
    Obj(Rc<BTreeMap<String, Value>>),
}
// size: 16 bytes on x86_64 (tag: 8 bytes due to alignment, payload: 8 bytes)
```

- All heap types use `Rc` (single-threaded reference counting)
- `Num` stores f64 inline (no heap allocation for numbers)
- Cranelift-generated code treats values as `i128` (two i64 words) and calls runtime helper functions for type checks and operations

**What to verify in Phase 1:**

- Is 16-byte `Value` passable in Cranelift's calling convention? (Cranelift supports multi-value returns, but large struct passing may require pointer indirection)
- What is the overhead of Rc increment/decrement in JIT-generated code vs interpreter?
- Does the 16-byte representation cause register pressure issues in Cranelift's register allocator?

---

## 3. Memory Management

How are heap-allocated values (strings, arrays, objects) managed?

### Options

#### (a) Reference counting (like jq)

Every heap value has an embedded reference count. `copy` increments, `free` decrements, deallocation when count reaches zero.

| Aspect | Assessment |
|---|---|
| Latency | **Deterministic.** No GC pauses. Deallocation happens immediately |
| Overhead | `incref`/`decref` on every copy/drop. jq's profiling shows this is a significant cost (existing-impls.md, section 3) |
| Cycles | Cannot collect cyclic structures. **Not a concern for jq** -- jq values are acyclic by construction (JSON has no cycles) |
| JIT integration | Cranelift generates `iadd` for incref, `isub` + `brif` for decref. Can be inlined |

#### (b) Tracing GC

Periodically trace all live objects from roots and collect unreachable ones.

| Aspect | Assessment |
|---|---|
| Latency | **Non-deterministic.** GC pauses vary. Unpredictable for streaming workloads |
| Overhead | No per-copy overhead. But root scanning and tracing add complexity |
| JIT integration | **Difficult.** GC needs to know the stack layout to find roots. Cranelift does not provide stack maps for precise GC (Wasmtime uses its own stack map mechanism tied to Wasm semantics). Safepoints must be inserted into generated code |
| Precedent | gojq uses Go's tracing GC, but Go provides the infrastructure. Building a tracing GC from scratch is a major undertaking |

#### (c) Arena allocation

Allocate all values in a region (arena). Free the entire arena at once when a pipeline stage completes.

| Aspect | Assessment |
|---|---|
| Latency | **Best-case deterministic.** Allocation is bump-pointer fast. Deallocation is a single `free()` call |
| Overhead | Almost zero per-allocation. But **values that outlive the arena must be copied out**, which complicates the API |
| JIT integration | **Excellent.** Arena pointer is passed as a context parameter. Allocation is `iadd` on the bump pointer |
| Limitation | jq's `reduce`, `foreach`, and recursive functions accumulate values across iterations. Pure arena allocation would require escape analysis to determine which values outlive the current scope |

### Recommendation: Reference counting (Rc) for Phase 1, with arena allocation for hot paths in later phases

**Rationale:**

1. **Correctness first.** Reference counting is the simplest model that handles jq's value semantics correctly
2. **jq values are acyclic** -- the main weakness of reference counting (cycles) does not apply
3. **jaq validates the approach** -- `Rc<T>` with clone avoidance (`next_if_one()`) achieves excellent performance
4. **Arena allocation is an optimization**, not a necessity. It can be layered on top of Rc for specific patterns (e.g., array construction in `[.[] | f]`) where lifetime analysis is straightforward
5. **Tracing GC is inappropriate** for a JIT targeting Cranelift. The stack map infrastructure does not exist, and building it from scratch is disproportionate effort

**Phase 1 approach:**

- Use `Rc<T>` for all heap-allocated types (strings, arrays, objects)
- Generate Cranelift code that calls `extern "C"` helper functions for refcount management:
  ```rust
  extern "C" fn value_clone(v: *const Value) -> Value;  // Rc::clone
  extern "C" fn value_drop(v: *mut Value);               // Rc::drop
  ```
- No COW optimization initially (jq has COW, but it adds complexity)

**Phase 2+ upgrade path:**

- **Elide redundant refcount operations**: Static analysis can prove that many `clone`/`drop` pairs cancel out (e.g., when a value is consumed exactly once in a pipe)
- **Arena for array builders**: `[.[] | f]` allocates a temporary arena for intermediate values, then copies the final array to Rc-managed storage
- **COW for objects**: When `jvp_refcnt_unshared()` equivalent shows refcount == 1, mutate in place

**What to verify in Phase 1:**

- What is the refcount overhead as a percentage of total JIT execution time? Profile against jq's interpreter
- Can Cranelift inline the fast path of refcount operations (increment is just `iadd`, decrement is `isub` + branch)?
- Does the lack of COW cause pathological performance on object-heavy workloads (e.g., `walk(...)`, repeated object update)?

---

## 4. Backtracking Strategy

How does the JIT implement jq's generator/backtracking semantics?

### Analysis Summary

backtracking.md evaluated seven strategies. The key findings:

| Strategy | Cranelift fit | Performance | Impl cost | Phase 1 viable |
|---|---|---|---|---|
| CPS + callbacks | High | High | Medium-High | Yes |
| Stack copy/switch | Low | Highest | Very High | No |
| Coroutines/fibers | Low | Medium | Medium | No |
| Trampoline | High | Low | Low | Yes (baseline) |
| Iterator transform | Low (for JIT) | Medium | Low (interp) | Yes (interp only) |
| Segmented stacks | Low | Medium | High | No |
| Delimited continuations | Low | High | Very High | No |

Cranelift does not provide direct stack manipulation, coroutine support, or first-class continuations. This eliminates stack copy, coroutines, segmented stacks, and delimited continuations as primary strategies.

### Recommendation: CPS + callback compilation to Cranelift

**Core insight (from backtracking.md, section 4.2):**

CPS transformation converts every jq filter from "a function that may produce multiple values via backtracking" into "a function that calls a callback for each value it produces." This **eliminates backtracking entirely** -- generators become ordinary nested function calls.

```
Semantic translation:

  jq:    .[] | . + 1
  CPS:   each(input, |elem| {
             let result = add(elem, 1);
             callback(result);
         })
```

**Why CPS + callbacks is optimal for Cranelift:**

1. **Natural IR mapping**: CPS-transformed code is a tree of function calls. Cranelift's `FunctionBuilder` directly supports this -- basic blocks, call instructions, and control flow
2. **Backtracking disappears**: No fork stack, no state save/restore, no ON_BACKTRACK handlers. `empty` = "don't call the callback." Comma = "call callback for left, then for right." Pipe = "left's callback calls right"
3. **Proven approach**: CapyScheme (Scheme -> CPS -> Cranelift -> native) demonstrates the pattern works
4. **Inline-friendly**: Small callbacks (e.g., `. + 1`) can be inlined into the caller, eliminating call overhead

**CPS interface for JIT-compiled filters:**

```rust
/// A JIT-compiled jq filter.
/// Takes an input value and a callback; calls the callback for each output value.
type JittedFilter = extern "C" fn(
    input: *const Value,
    callback: extern "C" fn(*const Value, *mut Context),
    ctx: *mut Context,
);
```

Translation patterns:

| jq construct | CPS translation |
|---|---|
| `f \| g` | `f(input, \|v, ctx\| g(v, callback, ctx), ctx)` |
| `f, g` | `f(input, callback, ctx); g(input, callback, ctx)` |
| `empty` | `/* don't call callback */` |
| `if c then t else e` | `if eval(c, input) then t(input, callback, ctx) else e(input, callback, ctx)` |
| `.[]` | `for elem in input.iter() { callback(elem, ctx); }` |
| `reduce .[] as $x (init; update)` | `let mut acc = init; each(input, \|x\| { acc = update(acc, x); }); callback(acc, ctx)` |
| `try f catch g` | `let ok = false; f(input, \|v, ctx\| { ok = true; callback(v, ctx); }, ctx); if !ok { g(input, callback, ctx); }` |

**Phase 1 approach:**

Phase 1 targets arithmetic filters only (no generators). CPS is still the right foundation because:
- Even `. + 1` is compiled as `fn(input, callback, ctx) { callback(add(input, 1), ctx); }`
- The interface is established from day one
- Adding generators later requires no architectural change

**What to verify in Phase 1:**

- Can Cranelift efficiently compile nested callback calls? (Concern: deep nesting may cause excessive stack usage)
- What is the overhead of `extern "C"` calling convention for callbacks that could be inlined?
- Does Cranelift's inliner handle cross-function optimization, or must inlining be done at the IR level before Cranelift?

**Known risk: Stack depth for deeply nested generators.**

A filter like `.[][] | .[][] | .[][] | ...` produces deeply nested CPS callbacks. Unlike the interpreter (which uses a flat fork stack), CPS uses the native call stack. Mitigation options:
1. Stack depth monitoring + fallback to interpreter
2. Trampoline for depth > N (hybrid approach)
3. Cranelift's `try_call` for stack overflow detection

---

## 5. Compilation Unit Granularity

What is the scope of each Cranelift function that the JIT produces?

### Options

#### (a) Whole top-level filter as one function

The entire jq program (from command-line `-f` or inline expression) is compiled into a single Cranelift function.

| Aspect | Assessment |
|---|---|
| Pros | Maximum optimization opportunity (inlining, constant propagation, dead code elimination all within one function). No inter-function call overhead |
| Cons | Large functions stress Cranelift's register allocator and increase compile time. **Subfunctions (closures passed to `map`, `select`, etc.) are conceptually separate** -- forcing them into one function requires flattening the call graph |
| Cranelift fit | `FunctionBuilder` is designed for single-function construction. Very large functions (100+ basic blocks) may hit quadratic behavior in register allocation |

#### (b) Per-subfunction compilation

Each `struct bytecode` node in jq's bytecode tree (top-level + each subfunction) becomes one Cranelift function.

| Aspect | Assessment |
|---|---|
| Pros | **Natural mapping to jq's bytecode structure.** Each subfunction is already identified by `bytecode->subfunctions[]`. Compilation of each function is independent -- good for incremental and parallel compilation. Manageable function size |
| Cons | Cross-function optimization (inlining) must be done manually. Call overhead for small subfunctions (e.g., `select(. > 3)` is a 5-instruction subfunction) |
| Cranelift fit | **Ideal.** `FunctionBuilder` handles one function at a time. `Module::declare_function` / `Module::define_function` support multiple function declarations with cross-references |

#### (c) Trace-based (hot path detection)

Instrument the interpreter to detect hot bytecode sequences. Compile only the hot traces into Cranelift functions.

| Aspect | Assessment |
|---|---|
| Pros | Only compiles what matters. Adapts to actual workload |
| Cons | **Phase 1 incompatible** -- requires a working interpreter with profiling infrastructure first. Trace selection heuristics are complex. Side exits from compiled traces need handling |
| Cranelift fit | Possible but non-trivial. Each trace becomes a function with guard checks and side exit stubs |

### Recommendation: (b) Per-subfunction, with selective inlining

**Rationale:**

1. **Direct mapping**: jq's `struct bytecode` tree directly defines the compilation units. No heuristics needed
2. **Cranelift alignment**: `FunctionBuilder` + `Module` are designed for this exact granularity
3. **CPS compatibility**: In the CPS model, each subfunction becomes a separate compiled function that takes `(input, callback, ctx)`. The top-level filter calls into subfunctions via function pointers or direct calls
4. **Incremental path**: Can start by compiling only the top-level function and interpreting subfunctions, then gradually compile more

**Phase 1 approach:**

Phase 1 targets arithmetic filters (`+`, `-`, `*`, `/`, `.` identity) with no subfunctions. The compilation unit is the single top-level bytecode. This is effectively option (a) for the degenerate case.

As subfunctions are introduced (Phase 3+):

```
struct bytecode (root: ". + 1")          -> fn jq_main(input, callback, ctx)
  subfunctions: []                       -> (none)

struct bytecode (root: "map(. * 2)")     -> fn jq_main(input, callback, ctx)
  subfunctions[0]: ". * 2"              -> fn jq_sub_0(input, callback, ctx)
```

**Inlining strategy (Phase 3+):**

Small subfunctions (estimated < 20 CLIF instructions after CPS transform) are inlined into the caller at the IR level, before passing to Cranelift. This recovers the optimization benefits of option (a) for common patterns:

- `select(f)` -- always inline (just a branch)
- `map(f)` where `f` is simple -- inline `f` into the loop body
- User-defined `def` with > 20 instructions -- compile separately, call via function pointer

**Why not trace-based (option c):**

Trace-based JIT is the right approach for a mature system with a working interpreter baseline. For Phase 1, we do not yet have an interpreter to profile. Trace-based compilation is a Phase 6 concern (per PLAN.md: "Tiered compilation -- interpreter -> JIT switching").

**What to verify in Phase 1:**

- What is the compilation time for a single arithmetic function in Cranelift? (Target: < 1ms for a 10-instruction function)
- Can cross-function calls between JIT-compiled subfunctions use direct call instructions (not indirect)? (Cranelift's `Module` supports this via `declare_function` + `Linkage::Local`)
- What is the call overhead for the CPS callback pattern compared to direct inlining?

---

## Summary: Phase 1 Baseline Decisions

| Decision | Phase 1 Choice | Rationale | Upgrade Path |
|---|---|---|---|
| Bytecode acquisition | libjq FFI (bindgen) | Complete bytecode access, semantic fidelity | Custom Rust parser if libjq proves too burdensome |
| Value representation | 16-byte Rust enum | Simplest correct implementation. jaq validates performance | NaN boxing for numeric hot paths |
| Memory management | Rc (reference counting) | Deterministic, no GC infra needed, jaq-proven | Arena for array builders, refcount elision |
| Backtracking | CPS + callbacks | Eliminates backtracking, natural Cranelift fit | Stack depth guard + trampoline for deep nesting |
| Compilation unit | Per-subfunction (trivial in Phase 1: one function) | Maps to bytecode tree, FunctionBuilder-aligned | Selective inlining for small subfunctions |

### Open Questions for Phase 1 Validation

1. **libjq build integration**: Can `build.rs` + `cc` crate handle jq's autotools build, or do we need to vendor a pre-built `libjq`?
2. **Value passing convention**: How does Cranelift handle 16-byte struct arguments/returns? Pointer indirection or register pairs?
3. **Callback inlining**: Does Cranelift optimize through indirect calls (`call_indirect`), or must we inline at the IR level?
4. **Compilation latency budget**: For `jq '.foo + 1' < small.json`, what is the acceptable JIT compilation overhead vs just interpreting?
5. **Correctness oracle**: Can we use jq itself (via libjq) as the reference implementation for differential testing?
