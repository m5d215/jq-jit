# jq Official Test Suite Failure Analysis

**Date**: 2026-02-28
**Results**: 149 PASS, 181 FAIL, 180 SKIP, 0 ERROR / 510 total (29.2% PASS rate)

## Summary by Category

| # | Category | Count | Severity | Fix Effort |
|---|----------|-------|----------|------------|
| 1 | [Compilation panics (exit 101)](#1-compilation-panics-exit-101) | 33 | Critical | Medium |
| 2 | [JSON output formatting (spaces)](#2-json-output-formatting-spaces) | 27 | Low | Easy |
| 3 | [Slice/index with negative, float, or NaN](#3-sliceindex-with-negative-float-or-nan) | 15 | High | Medium |
| 4 | [try/catch/error semantics](#4-trycatcherror-semantics) | 12 | High | Hard |
| 5 | [Number formatting (scientific notation)](#5-number-formatting-scientific-notation) | 8 | Medium | Medium |
| 6 | [Miscellaneous bugs](#6-miscellaneous-bugs) | 8 | Medium | Varies |
| 7 | [Object key insertion order](#7-object-key-insertion-order) | 7 | Low | Medium |
| 8 | [path() operations](#8-path-operations) | 7 | High | Hard |
| 9 | [String operations bugs](#9-string-operations-bugs) | 6 | Medium | Easy-Medium |
| 10 | [Try-operator (?) semantics](#10-try-operator--semantics) | 5 | High | Medium |
| 11 | [range/while/until return boolean instead of values](#11-rangewhileuntil-return-boolean-instead-of-values) | 5 | Critical | Medium |
| 12 | [Update/assignment bugs](#12-updateassignment-bugs) | 5 | High | Medium |
| 13 | [Division by zero handling](#13-division-by-zero-handling) | 5 | Medium | Easy |
| 14 | [Negative array index assignment](#14-negative-array-index-assignment) | 4 | High | Easy |
| 15 | [Unicode string index (byte vs codepoint)](#15-unicode-string-index-byte-vs-codepoint) | 4 | Medium | Medium |
| 16 | [join() with null elements](#16-join-with-null-elements) | 4 | Medium | Easy |
| 17 | [Parse errors (exit 2)](#17-parse-errors-exit-2) | 3 | Medium | Easy |
| 18 | [Unicode escape output format](#18-unicode-escape-output-format) | 3 | Low | Easy |
| 19 | [contains() with null bytes](#19-contains-with-null-bytes) | 3 | Medium | Medium |
| 20 | [flatten(depth) broken](#20-flattendepth-broken) | 3 | High | Easy |
| 21 | [implode/explode invalid codepoints](#21-implodeexplode-invalid-codepoints) | 3 | Medium | Easy |
| 22 | [String repetition (string * number)](#22-string-repetition-string--number) | 3 | Medium | Easy |
| 23 | [limit/skip/first builtins](#23-limitskipfirst-builtins) | 3 | Medium | Easy |
| 24 | [sort/min/max comparison bugs](#24-sortminmax-comparison-bugs) | 2 | Medium | Medium |
| 25 | [NaN handling](#25-nan-handling) | 1 | Low | Easy |
| 26 | [abs on string](#26-abs-on-string) | 1 | Low | Easy |
| 27 | [Object add](#27-object-add) | 1 | High | Easy |
| **Total** | | **181** | | |

---

## Detailed Analysis

### 1. Compilation panics (exit 101)

**Count**: 33 failures
**Tests**: #54, #58, #64, #70, #72, #74-76, #82, #85-88, #90, #94, #134, #145, #165, #243-244, #256, #267, #288, #291-292, #336, #455, #467-469, #472, #474, #476

**Root cause**: The JIT compiler panics with `codegen_expr: generator expression ... in scalar context -- use codegen_generator instead` when a generator (multi-value-producing expression) appears in a position where a scalar is expected. This is the single largest category of failures.

**Sub-patterns**:

**(a) Multiple arguments to builtins** (11 tests: #54, #58, #85-88, #90, #94, #134, #336)
```
Filter:  [range(0,1;3,4)]
Input:   null
Expected: [0,1,2,3,0,1,2,3]
Actual:   panic: codegen_expr: generator expression in scalar context
```
When builtins like `range`, `index`, `join`, `flatten`, `sort_by`, `group_by` receive multiple values via `,` in their arguments, the compiler cannot handle the generator context.

**(b) Generator in control flow positions** (8 tests: #64, #70, #72, #74-76, #82, #256)
```
Filter:  [limit(3; .[])]
Input:   [11,22,33,44,55,66,77,88,99]
Expected: [11,22,33]
Actual:   panic: generator expression Each in scalar context
```
`foreach`, `limit`, `skip`, `first` with expression arguments that produce generators.

**(c) try/catch with generators and complex expressions** (8 tests: #267, #288, #291-292, #467-469, #472)
```
Filter:  [-try .]
Input:   null
Expected: [0]  (or similar)
Actual:   panic
```
`try` in expression positions like `-try .`, `try` in update context, and complex try/catch nesting.

**(d) update-assignment with generators** (5 tests: #165, #243-244, #474, #476)
```
Filter:  (.[] | select(. >= 2)) |= empty
Input:   [1,5,3,0,7]
Expected: [1,0]
Actual:   panic
```
`|=` with `select`, `empty`, `try`, or user-defined functions that produce generators.

**(e) label/break** (1 test: #455)
```
Filter:  [ label $if | range(10) | ., (select(. == 5) | break $if) ]
Input:   null
Expected: [0,1,2,3,4,5]
Actual:   panic
```

**(f) map_values** (1 test: #145)
```
Filter:  map_values(.+1)
Input:   {"a": 1}
Expected: {"a":2}
Actual:   panic: generator expression Each in scalar context
```

---

### 2. JSON output formatting (spaces)

**Count**: 27 failures
**Tests**: #21, #103, #123, #125, #139, #142, #159, #222-223, #234, #236, #246, #278-279, #282, #297-298, #304, #330-331, #343-344, #347-348, #364-365, #435

**Root cause**: jq-jit uses `-c` (compact) mode and outputs `[1,2,3]` and `{"a":1}`, while the test file expects jq's compact output with spaces after `:` and `,` in some contexts: `[1, 2, 3]` and `{"a": 1}`.

**Examples**:
```
Filter:  [10 * 20, 20 / .]
Input:   4
Expected: [200, 5]
Actual:   [200,5]

Filter:  {"message": "goodbye"}
Input:   {"message": "hello"}
Expected: {"message": "goodbye"}
Actual:   {"message":"goodbye"}
```

**Fix**: The test file was likely generated with a non-compact jq output mode. Either update the test expectations to match `-c` mode, or make the test runner normalize whitespace. This is a **test harness issue**, not a jq-jit bug. The computed values are correct.

---

### 3. Slice/index with negative, float, or NaN

**Count**: 15 failures
**Tests**: #91-92, #352, #485-489, #492-498

**Root cause**: jq-jit does not support slicing with:
- Negative indices: `.[:-2]`, `.[-5:4]`, `.[-2:]`
- Float indices: `.[1.2:3.5]`, `.[1.5:3.5]`
- NaN indices: `.[nan:1]`, `.[1:nan]`, `.[nan]`
- Dynamic expression slicing: `[1,2][0:.]`

All of these return `"Cannot index array/string with object"`, meaning the slice bounds are being treated as an object type instead of being coerced to integers.

**Examples**:
```
Filter:  [.[3:2], .[-5:4], .[:-2], .[-2:], .[3:3][1:], .[10:]]
Input:   [0,1,2,3,4,5,6]
Expected: [[], [2,3], [0,1,2,3,4], [5,6], [], []]
Actual:   ["Cannot index array with object", ...]

Filter:  [range(10)] | .[1.2:3.5]
Input:   null
Expected: [1,2,3]
Actual:   (empty - error)
```

**Fix**: The slice codegen needs to handle negative indices (wrap around from end), truncate float indices to integers, and treat NaN as a special case (NaN start = 0, NaN end = length).

---

### 4. try/catch/error semantics

**Count**: 12 failures
**Tests**: #36, #284, #286-287, #290, #293, #295, #400-402, #470-471

**Root cause**: Multiple interrelated issues with error handling:

**(a) `error` builtin ignores its argument** (3 tests: #36, #284, #295)
```
Filter:  .[] | try error catch .
Input:   [1,null,2]
Expected: 1 / null / 2
Actual:   (empty - errors go uncaught)
```
`error` should throw its input (`.`) as the error value, which `catch` should receive. Instead, `error` seems to either not throw or throw a generic "error" string, and `catch` doesn't capture it properly.

```
Filter:  try error("foo") catch .
Actual:  "error"  (should be "foo")
```

**(b) `?` (try-operator) on expressions doesn't suppress errors properly** (2 tests: #286-287)
```
Filter:  [[.[]|[.a,.a]]?]
Input:   [null,true,{"a":1}]
Expected: []  (entire expression fails on `true`, ? suppresses all)
Actual:   [[[null,null],["Cannot index boolean with string",...],[1,1]]]
```

**(c) `try` expression precedence/parsing** (2 tests: #290, #293)
```
Filter:  1 + try 2 catch 3 + 4
Expected: 7  (parsed as: 1 + (try 2 catch (3 + 4)), result = 1 + 2 = 3... wait)
Actual:   3
```
jq parses `try` with specific precedence rules. `1 + try 2 catch 3 + 4` should yield `1 + 2 = 3`... actually expected 7 means it's `(1 + try 2 catch 3) + 4 = (1+2)+4 = 7`. The parser has wrong precedence for `try`.

**(d) Type error messages missing value context** (3 tests: #400-402)
```
Filter:  try -. catch .
Input:   "very-long-long-long-long-string"
Expected: "string (\"very-long-long-long-long...\") cannot be negated"
Actual:   (empty)
```
When type errors occur, the error message should include the value that caused the error, but jq-jit either produces no output or a generic message.

**(e) Nested try/catch** (2 tests: #470-471)
```
Filter:  try (try error catch "inner catch \(.)") catch "outer catch \(.)"
Input:   "foo"
Expected: "inner catch foo"
Actual:   (empty)
```
Nested try/catch chains don't propagate error values correctly.

---

### 5. Number formatting (scientific notation)

**Count**: 8 failures
**Tests**: #108, #111, #112, #126-128, #149, #451

**Root cause**: jq-jit formats numbers differently from jq. jq uses a specific algorithm that:
- Outputs `2.0` for integer results of float operations (e.g., `1+1` when input is a string)
- Uses scientific notation for very small/large numbers: `1.05e-19`, `1e+17`
- Preserves high-precision large integers: `1000000000000000002`
- Handles extreme exponents: `9E+999999999`

jq-jit always outputs plain decimal or standard double formatting.

**Examples**:
```
Filter:  1+1
Input:   "wtasdf"
Expected: 2.0
Actual:   2

Filter:  1e-19 + 1e-20 - 5e-21
Expected: 1.05e-19
Actual:   0.000000000000000000105

Filter:  9E999999999, 9999999999E999999990, 1E-999999999
Expected: 9E+999999999 / 9.999999999E+999999999 / 1E-999999999
Actual:   1.7976931348623157e+308 / 1.7976931348623157e+308 / 0
```

**Fix**: Implement jq-compatible number formatting (shortest representation, scientific notation when appropriate). The extreme exponent case (#128) also indicates lack of arbitrary-precision number support (clamped to f64 range).

---

### 6. Miscellaneous bugs

**Count**: 8 failures
**Tests**: #13, #237, #463, #499-500, #501-502, #505

**(a) @html, @urid, @sh format functions** (#13)
```
Filter:  @text,@json,([1,.]|@csv,@tsv),@html,(@uri|.,@urid),@sh,(@base64|.,@base64d)
```
- `@html`: uses `&#39;` instead of `&apos;` for apostrophe, doesn't escape `"` to `&quot;`
- `@urid`: doesn't decode URI-encoded strings (returns input unchanged)
- `@sh`: shell quoting not implemented correctly

**(b) Comma in update expression only produces first branch** (#237)
```
Filter:  .[] += 2, .[] *= 2, .[] -= 2, .[] /= 2, .[] %=2
Expected: 5 lines of output
Actual:   5 lines (correct values, but line 4 has [0.5,1.5,2.5] instead of [0.5, 1.5, 2.5])
```
This is actually a JSON formatting issue (spaces after commas in arrays).

**(c) fromjson error handling** (#463, #500)
```
Filter:  .[] | try (fromjson | isnan) catch .
Input:   ["NaN","-NaN","NaN1",...]
Expected: true / true / "Invalid numeric literal..."
Actual:   true / true / false
```
`fromjson` accepts invalid strings like "NaN1" instead of rejecting them. Also, `fromjson` error messages don't match jq's format.

**(d) setpath type checking** (#499, #505)
```
Filter:  try ["ok", setpath([1]; 1)] catch ["ko", .]
Input:   {"hi":"hello"}
Expected: ["ko","Cannot index object with number (1)"]
Actual:   ["ok",[null,1]]
```
`setpath` doesn't validate that the path is compatible with the value type.

**(e) ltrimstr/rtrimstr missing output for second comma branch** (#501-502)
```
Filter:  try ltrimstr(1) catch "x", try rtrimstr(1) catch "x" | "ok"
Expected: "ok" / "ok" (two lines)
Actual:   "ok" (one line)
```
The comma operator only produces output from the first branch. This appears related to the generator-in-scalar-context issue but doesn't cause a panic.

---

### 7. Object key insertion order

**Count**: 7 failures
**Tests**: #22, #240, #247, #342, #345, #457, #460

**Root cause**: jq-jit outputs object keys in a different order than jq. jq preserves insertion order of keys; jq-jit appears to sort them alphabetically or use hash-map ordering.

**Examples**:
```
Filter:  .foo[2].bar = 1
Input:   {"foo":[11], "bar":42}
Expected: {"foo":[11,null,{"bar":1}], "bar":42}
Actual:   {"bar":42,"foo":[11,null,{"bar":1}]}

Filter:  1 as $foreach | 2 as $and | 3 as $or | { $foreach, $and, $or, a }
Input:   {"a":4,"b":5}
Expected: {"foreach":1,"and":2,"or":3,"a":4}
Actual:   {"a":4,"and":2,"foreach":1,"or":3}
```

**Note**: The test runner (run.sh) already has a `sort`-based normalization that handles simple reordering. These tests fail because sorting the lines alphabetically doesn't match when the key differences are within a single JSON line, OR because the expected value includes keys like `"foo"` before `"bar"` in insertion order and sorting by string content doesn't resolve the difference.

**Fix**: Use an insertion-ordered map (e.g., `IndexMap`) for JSON objects to preserve key order.

---

### 8. path() operations

**Count**: 7 failures
**Tests**: #213, #215-219, #226

**Root cause**: The `path()` builtin has multiple issues:

**(a) path() with select doesn't filter** (#213)
```
Filter:  path(.[] | select(.>3))
Input:   [1,5,3]
Expected: [1]
Actual:   [0] / [1] / [2]  (returns all paths, ignoring select)
```

**(b) path() with non-path expressions should error** (#215-218)
```
Filter:  try path(.a | map(select(.b == 0))) catch .
Expected: "Invalid path expression with result [...]"
Actual:   (empty)
```

**(c) path() with nested path expression** (#219)
```
Filter:  path(.a[path(.b)[0]])
Input:   {"a":{"b":0}}
Expected: ["a","b"]
Actual:   (empty)
```

**(d) delpaths type checking** (#226)
```
Filter:  try delpaths(0) catch .
Expected: "Paths must be specified as an array"
Actual:   {}
```

---

### 9. String operations bugs

**Count**: 6 failures
**Tests**: #17, #300, #305-306, #313, #430

**(a) @urid not implemented** (#17)
```
Filter:  @urid
Input:   "%CE%BC"
Expected: "\u03bc" (μ)
Actual:   "%CE%BC" (returned unchanged)
```

**(b) split("") includes empty strings at boundaries** (#300)
```
Filter:  split("")
Input:   "abc"
Expected: ["a","b","c"]
Actual:   ["","a","b","c",""]
```

**(c) rtrimstr("") and trimstr("") broken** (#305-306)
```
Filter:  [.[]|rtrimstr("")]
Input:   ["a", "xx", ""]
Expected: ["a", "xx", ""]
Actual:   ["","",""]   (returns empty string for all inputs)
```

**(d) indices() for array subsequences not implemented** (#313)
```
Filter:  indices([1,2])
Input:   [0,1,2,3,1,4,2,5,1,2,6,7]
Expected: [1,8]
Actual:   []
```

**(e) index("") should return null** (#430)
```
Filter:  index("")
Input:   ""
Expected: null
Actual:   0
```

---

### 10. Try-operator (?) semantics

**Count**: 5 failures
**Tests**: #30-31, #34-35, #417

**Root cause**: The `?` operator (try without catch) should suppress errors and skip values that cause errors. Instead, jq-jit produces `null` for error cases.

**Examples**:
```
Filter:  [.[]|.foo?]
Input:   [1,[2],{"foo":3,"bar":4},{},{"foo":5}]
Expected: [3,null,5]  (only objects produce values; 1 and [2] are suppressed)
Actual:   [null,null,3,null,5]  (1 and [2] produce null instead of being suppressed)

Filter:  [.[]|.[1:3]?]
Input:   [1,null,true,false,"abcdef",{},{"a":1,"b":2},[],[1,2,3,4,5],[1,2]]
Expected: [null,"bc",[],[2,3],[2]]
Actual:   [null,null,null,null,null,null,null,null,null,null]
```
The `?` operator on slicing returns null for all inputs instead of computing the slice for types that support it (strings, arrays, null) and suppressing errors for types that don't (numbers, booleans, objects).

---

### 11. range/while/until return boolean instead of values

**Count**: 5 failures
**Tests**: #55-57, #59, #62

**Root cause**: `range(from;to;step)`, `while`, and `until` return the comparison result (boolean) instead of the actual computed values. This strongly suggests the 3-argument `range`, `while`, and `until` are implemented incorrectly -- they appear to be returning the loop condition rather than the loop variable.

**Examples**:
```
Filter:  [range(0;10;3)]
Expected: [0,3,6,9]
Actual:   [true]  (returns the comparison 0<10, not the sequence)

Filter:  [while(.<100; .*2)]
Input:   1
Expected: [1,2,4,8,16,32,64]
Actual:   [true]  (returns the predicate result)

Filter:  [.[]|[.,1]|until(.[0] < 1; [.[0] - 1, .[1] * .[0]])|.[1]]
Input:   [1,2,3,4,5]
Expected: [1,2,6,24,120]
Actual:   ["Cannot index boolean with number",...]  (tries to index the boolean result)
```

**Fix**: The loop builtins need to yield the accumulator/counter value, not the loop predicate result.

---

### 12. Update/assignment bugs

**Count**: 5 failures
**Tests**: #235, #242, #268, #425, #426

**(a) Assignment doesn't read .bar before setting .foo** (#235)
```
Filter:  .foo = .bar
Input:   {"bar":42}
Expected: {"foo":42, "bar":42}
Actual:   {"bar":42,"foo":null}
```
The RHS `.bar` should be evaluated before the assignment, but `.foo` gets `null` instead of `42`.

**(b) Complex path update with |=** (#242)
```
Filter:  .[] | try (getpath(["a",0,"b"]) |= 5) catch .
Input:   [null,{"b":0},...]
Expected: {"a":[{"b":5}]} / {"b":0,"a":[{"b":5}]} / error message
Actual:   null / {"b":0} / {"a":0}  (update not applied)
```

**(c) Alternative-assignment operator //=** (#268)
```
Filter:  .[] //= .[0]
Input:   ["hello",true,false,[false],null]
Expected: ["hello",true,"hello",[false],"hello"]
Actual:   ["hello",true,false,[false],false]
```
`//=` should replace falsy values (false, null) with the RHS. Instead, false values are kept.

**(d) Assignment through variable binding** (#425)
```
Filter:  (.a as $x | .b) = "b"
Input:   {"a":null,"b":null}
Expected: {"a":null,"b":"b"}
Actual:   {"a":null,"b":null}
```

**(e) Recursive descent in update** (#426)
```
Filter:  (.. | select(...))|.b) |= .[0]
Expected: {"a": {"b": 1}}
Actual:   {"a":{"b":[null,{"b":null}]}}
```

---

### 13. Division by zero handling

**Count**: 5 failures
**Tests**: #410-414

**Root cause**: jq-jit doesn't raise errors on division by zero. Instead, it returns IEEE 754 infinity or null. jq (since 1.6) treats division by zero as an error.

**Examples**:
```
Filter:  try (1/.) catch .
Input:   0
Expected: "number (1) and number (0) cannot be divided because the divisor is zero"
Actual:   1.7976931348623157e+308  (infinity-like value)

Filter:  try (1%0) catch .
Expected: "number (1) and number (0) cannot be divided (remainder) because the divisor is zero"
Actual:   (empty)
```

---

### 14. Negative array index assignment

**Count**: 4 failures
**Tests**: #37-40

**Root cause**: Setting array elements via negative indices doesn't work. Negative indices should count from the end of the array.

**Examples**:
```
Filter:  .[-1] = 5
Input:   [0,1,2]
Expected: [0,1,5]
Actual:   [0,1,2]  (assignment silently ignored)

Filter:  .[-2] = 5
Input:   [0,1,2]
Expected: [0,5,2]
Actual:   [0,1,2]  (assignment silently ignored)

Filter:  try (.foo[-1] = 0) catch .
Input:   null
Expected: "Out of bounds negative array index"
Actual:   null  (no error raised)
```

---

### 15. Unicode string index (byte vs codepoint)

**Count**: 4 failures
**Tests**: #316-319

**Root cause**: `index()`, `rindex()`, and `indices()` on strings return byte offsets instead of Unicode codepoint offsets.

**Examples**:
```
Filter:  index("!")
Input:   "здравствуй мир!"
Expected: 14  (14th codepoint)
Actual:   27  (27th byte - Cyrillic chars are 2 bytes each in UTF-8)

Filter:  indices("o")
Input:   "🇬🇧oo"
Expected: [2,3]  (codepoint positions)
Actual:   [8,9]  (byte positions - flag emoji is 8 bytes)

Filter:  indices("o")
Input:   "ƒoo"
Expected: [1,2]
Actual:   [2,3]  (ƒ is 2 bytes in UTF-8)
```

---

### 16. join() with null elements

**Count**: 4 failures
**Tests**: #405-408

**Root cause**: `join()` skips null elements instead of treating them as empty strings. In jq, `null` elements in `join` produce the separator without a value.

**Examples**:
```
Filter:  .[] | join(",")
Input:   [[], [null], [null,null], [null,null,null]]
Expected: "" / "" / "," / ",,"
Actual:   "" / "" / "" / ""  (nulls are completely ignored)

Filter:  .[] | join(",")
Input:   [["a",null], [null,"a"]]
Expected: "a," / ",a"
Actual:   "a" / "a"  (null elements dropped entirely)
```

Also, `join` should error on non-string/non-null elements (#407-408) but instead stringifies them:
```
Filter:  try join(",") catch .
Input:   ["1","2",{"a":{"b":{"c":33}}}]
Expected: "string (\"1,2,\") and object (...) cannot be added"
Actual:   "1,2,{\"a\":{\"b\":{\"c\":33}}}"  (object is stringified instead of error)
```

---

### 17. Parse errors (exit 2)

**Count**: 3 failures
**Tests**: #5, #442, #443

**(a) Unary minus on literal** (#5)
```
Filter:  -1
Input:   null
Expected: -1
Actual:   exit 2: "Unknown option: -1"
```
The parser treats `-1` as a command-line flag instead of a jq filter.

**(b) Unary minus on input with have_decnum** (#442-443)
```
Filter:  -. | tojson == if have_decnum then "..." else "..." end
```
The parser fails on the `have_decnum` builtin (exit 2). This is likely an unsupported builtin rather than a true parse error.

---

### 18. Unicode escape output format

**Count**: 3 failures
**Tests**: #10-11, #119

**Root cause**: jq-jit outputs `\r`, `\n`, `\t`, and literal Unicode characters (like `μ`) instead of `\u000d`, `\u000a`, `\u0009`, and `\u03bc`.

**Examples**:
```
Filter:  "Aa\r\n\t\b\f\u03bc"
Expected: "Aa\u000d\u000a\u0009\u0008\u000c\u03bc"
Actual:   "Aa\r\n\t\u0008\u000cμ"
```

jq uses `\uXXXX` escapes for control characters (CR, LF, TAB) and non-ASCII Unicode in compact mode. jq-jit uses the short escape forms (`\r`, `\n`, `\t`) and outputs literal UTF-8 for non-ASCII.

**(b) Null byte in string concatenation** (#119)
```
Filter:  "\u0000\u0020\u0000" + .
Input:   "\u0000\u0020\u0000"
Expected: "\u0000 \u0000\u0000 \u0000"
Actual:   ""
```
Strings containing null bytes (`\u0000`) are truncated or handled incorrectly.

---

### 19. contains() with null bytes

**Count**: 3 failures
**Tests**: #280-281, #283

**Root cause**: `contains()` on strings with embedded null bytes (`\u0000`) gives wrong results, likely because the underlying string comparison uses C-style null-terminated string functions.

**Examples**:
```
Filter:  [contains("c"), contains("d")]
Input:   "ab\u0000cd"
Expected: [true, true]
Actual:   [false, false]  (can't see past the null byte)

Filter:  [contains("\u0000@"), contains("\u0000what")]
Input:   "ab\u0000cd"
Expected: [false, false]
Actual:   [true, true]  (incorrectly matches after null byte)
```

---

### 20. flatten(depth) broken

**Count**: 3 failures
**Tests**: #366-368

**Root cause**: `flatten(n)` with a depth argument doesn't actually flatten. It returns the input unchanged regardless of depth.

**Examples**:
```
Filter:  flatten(2)
Input:   [0, [1], [[2]], [[[3]]]]
Expected: [0, 1, 2, [3]]
Actual:   [0,[1],[[2]],[[[3]]]]  (no flattening occurred)

Filter:  try flatten(-1) catch .
Expected: "flatten depth must not be negative"
Actual:   [0,[1],[[2]],[[[3]]]]  (no error, no flattening)
```

Note: `flatten` (without argument) works correctly (test #364 passes with formatting normalization). The bug is specifically in the depth-limited variant.

---

### 21. implode/explode invalid codepoints

**Count**: 3 failures
**Tests**: #478-480

**Root cause**: `implode` doesn't validate or replace invalid Unicode codepoints. jq replaces invalid codepoints (negative, >1114111, surrogates 55296-57343) with U+FFFD (65533).

**Examples**:
```
Filter:  implode|explode
Input:   [-1,0,1,2,3,1114111,1114112,55295,55296,57343,57344,1.1,1.9]
Expected: [65533,0,1,2,3,1114111,65533,55295,65533,65533,57344,1,1]
Actual:   [0,0,1,2,3,1114111,55295,57344,1,1]
```
Missing entries for invalid codepoints (-1, 1114112, 55296, 57343) which should become 65533 (U+FFFD).

Also, `implode` on non-array or non-numeric elements should error but doesn't (#479).

---

### 22. String repetition (string * number)

**Count**: 3 failures
**Tests**: #325-327

**Root cause**: String repetition (`string * number`) is not implemented.

**Examples**:
```
Filter:  . * 100000 | [.[:10],.[-10:]]
Input:   "abc"
Expected: ["abcabcabca","cabcabcabc"]
Actual:   "string and number cannot be multiplied"

Filter:  . * 1000000000
Input:   ""
Expected: ""
Actual:   (empty/error)
```

---

### 23. limit/skip/first builtins

**Count**: 3 failures
**Tests**: #71, #73, #77

**Root cause**: Edge cases in `limit` and `skip` builtins.

```
Filter:  [limit(0; error)]
Expected: []  (0 elements, don't evaluate the expression)
Actual:   [0]

Filter:  try limit(-1; error) catch .
Expected: "limit doesn't support negative count"
Actual:   (empty)

Filter:  try skip(-1; error) catch .
Expected: "skip doesn't support negative count"
Actual:   (empty)
```

---

### 24. sort/min/max comparison bugs

**Count**: 2 failures
**Tests**: #335, #339

**Root cause**: jq's comparison/ordering for arrays and objects differs from jq-jit's implementation.

```
Filter:  sort
Input:   [42,[2,5,3,11],10,...,{},{"a":42},{"a":42,"b":2},{"a":[],"b":1}]
```
Arrays: jq compares element-by-element. jq-jit appears to compare by length first:
- Expected: `[2,5,3,11],[2,5,6],[2,6]`
- Actual: `[2,5,3,11],[2,6],[2,5,6]` (4-element array before 3-element)

Objects: jq compares by number of keys, then key names, then values. jq-jit has a different order:
- Expected: `{},{"a":42},{"a":42,"b":2},{"a":[],"b":1}`
- Actual: `{"a":42,"b":2},{"a":42},{"a":[],"b":1},{}`

`min`/`max` also wrong (#339): `min` returns `[4,2,"a"]` instead of `[1,3,"a"]`.

---

### 25. NaN handling

**Count**: 1 failure
**Test**: #349

```
Filter:  has(nan)
Input:   [0,1,2]
Expected: false
Actual:   true
```
`has(nan)` on an array should return false (NaN is not a valid index), but returns true.

---

### 26. abs on string

**Count**: 1 failure
**Test**: #447

```
Filter:  abs
Input:   "abc"
Expected: "abc"  (abs is identity for strings in jq)
Actual:   (empty - error)
```
`abs` on a string should return the string unchanged, but jq-jit raises an error.

---

### 27. Object add

**Count**: 1 failure
**Test**: #144

```
Filter:  map(add)
Input:   [[], [1,2,3], ["a","b","c"], [[3],[4,5],[6]], [{"a":1}, {"b":2}, {"a":3}]]
Expected: [null, 6, "abc", [3,4,5,6], {"a":3, "b": 2}]
Actual:   [null,6,"abc",[3,4,5,6],"object and object cannot be added"]
```
Adding two objects together (merging) is not implemented. `{"a":1} + {"b":2}` should produce `{"a":1,"b":2}`.

---

## Recommended Fix Priority

### Tier 1 — High impact, fixes many tests
1. **JSON output formatting** (27 tests) — Fix test runner to normalize spaces, or match jq's compact output exactly
2. **Compilation panics / generator-in-scalar-context** (33 tests) — Core codegen issue preventing many features from working
3. **Slice with negative/float indices** (15 tests) — Implement proper index coercion
4. **range/while/until returning boolean** (5 tests) — Fix loop builtins to return values

### Tier 2 — Important correctness fixes
5. **try/catch/error semantics** (12 tests) — Error value propagation
6. **Object key insertion order** (7 tests) — Use ordered map
7. **Negative index assignment** (4 tests) — Implement negative index resolution
8. **path() operations** (7 tests) — Fix path tracking with select/map
9. **Update/assignment** (5 tests) — Fix RHS evaluation order

### Tier 3 — Specific feature fixes
10. **Number formatting** (8 tests) — Match jq's number output format
11. **Division by zero** (5 tests) — Raise errors instead of returning infinity
12. **String operations** (6 tests) — Fix split(""), rtrimstr(""), indices([])
13. **flatten(depth)** (3 tests) — Implement depth parameter
14. **join() with nulls** (4 tests) — Treat null as empty string
15. **Unicode index** (4 tests) — Use codepoint offsets instead of byte offsets
16. **String * number** (3 tests) — Implement string repetition
17. **Object add** (1 test) — Implement object merge
18. **Unicode escape format** (3 tests) — Use \uXXXX for control chars

### Tier 4 — Edge cases
19. **Parse `-1` as filter** (1 test) — CLI argument parsing
20. **implode/explode validation** (3 tests)
21. **contains with null bytes** (3 tests)
22. **NaN handling** (1 test)
23. **abs on string** (1 test)
24. **sort comparison** (2 tests)
25. **limit/skip edge cases** (3 tests)
