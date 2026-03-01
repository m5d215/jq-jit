//! CPS Intermediate Representation.
//!
//! The CPS IR is a tree-structured expression representation converted from
//! jq's stack-based bytecode.  The bytecode→IR translation in [`crate::compiler`]
//! simulates the stack machine with `Expr` nodes instead of `jv` values, producing
//! a single `Expr` tree that captures the computation.
//!
//! Phase 1: arithmetic operations and field access.
//! Phase 2: comparison operators, unary builtins, type extensions.

use std::fmt;

/// A CPS IR expression node.
///
/// This is the output of the bytecode→IR translation pass.  Each node represents
/// a computation that, given an input value, produces an output value.
#[derive(Debug, Clone)]
pub enum Expr {
    /// The identity filter `.` — references the input value.
    Input,

    /// A constant literal value.
    Literal(Literal),

    /// A binary arithmetic operation.
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    /// Field/index access: `expr[key]` (e.g., `.foo` = `Input["foo"]`).
    Index {
        expr: Box<Expr>,
        key: Box<Expr>,
    },

    /// Optional field/index access: `expr[key]?` (e.g., `.foo?` = `Input["foo"]?`).
    /// Returns null instead of error for incompatible types.
    /// (Phase 8-2: INDEX_OPT)
    IndexOpt {
        expr: Box<Expr>,
        key: Box<Expr>,
    },

    /// A unary operation (Phase 2: length, type, tostring, tonumber, keys).
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
    },

    /// Conditional expression: `if cond then then_branch else else_branch end`.
    /// (Phase 3: control flow)
    IfThenElse {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },

    /// Try-catch expression: `try try_expr catch catch_expr`.
    /// (Phase 3: error handling)
    ///
    /// Evaluates `try_expr`.  If the result is an error, evaluates `catch_expr`
    /// with the error message as input.  Otherwise returns the try result.
    TryCatch {
        try_expr: Box<Expr>,
        catch_expr: Box<Expr>,
    },

    /// Comma expression: `left, right` — generator that produces outputs
    /// from both left and right in sequence.
    /// (Phase 4: generators)
    Comma {
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Empty expression: produces zero outputs (callback is never called).
    /// Corresponds to jq's `empty` / `BACKTRACK` opcode.
    /// (Phase 4: generators)
    Empty,

    /// Each expression: iterates over all elements of an array or all values
    /// of an object, applying `body` to each element.
    ///
    /// Corresponds to jq's `.[]` / `EACH` opcode.
    /// `input_expr` evaluates to the array/object to iterate over.
    /// `body` is evaluated with each element as `Input`.
    /// (Phase 4-3: generators)
    Each {
        input_expr: Box<Expr>,
        body: Box<Expr>,
    },

    /// Optional each expression: iterates like Each but silently produces
    /// zero outputs for non-iterable inputs instead of erroring.
    ///
    /// Corresponds to jq's `.[]?` / `EACH_OPT` opcode.
    /// (Phase 8-3: optional iteration)
    EachOpt {
        input_expr: Box<Expr>,
        body: Box<Expr>,
    },

    /// Local variable binding: `. as $x | body` or `expr as $x | body`.
    ///
    /// Evaluates `value`, stores it in variable slot `var_index`, then evaluates
    /// `body` which may reference the variable via `LoadVar`.
    /// (Phase 4-4: variables)
    LetBinding {
        var_index: u16,
        value: Box<Expr>,
        body: Box<Expr>,
    },

    /// Local variable reference: loads the value from variable slot `var_index`.
    ///
    /// Corresponds to jq's `LOADV` / `LOADVN` opcodes.
    /// (Phase 4-4: variables)
    LoadVar {
        var_index: u16,
    },

    /// Reduce expression: `reduce source as $var (init; update)`.
    ///
    /// Iterates over `source`, applying `update` to the accumulator for each
    /// element.  Returns the final accumulator value (1→1).
    ///
    /// The accumulator is stored in variable slot `acc_index`, and each element
    /// is bound to `var_index`.  The `update` expression receives the current
    /// accumulator as `Input` and can reference `$var` via `LoadVar`.
    /// (Phase 4-5: reduce/foreach)
    Reduce {
        source: Box<Expr>,
        init: Box<Expr>,
        var_index: u16,
        acc_index: u16,
        update: Box<Expr>,
    },

    /// Foreach expression: `foreach source as $var (init; update)`.
    ///
    /// Like Reduce, but yields each intermediate accumulator value (1→N).
    /// After each `update`, the current accumulator is output via callback.
    ///
    /// The accumulator is stored in variable slot `acc_index`, and each element
    /// is bound to `var_index`.  The `update` expression receives the current
    /// accumulator as `Input` and can reference `$var` via `LoadVar`.
    /// (Phase 4-5: reduce/foreach)
    Foreach {
        source: Box<Expr>,
        init: Box<Expr>,
        var_index: u16,
        acc_index: u16,
        update: Box<Expr>,
    },

    /// Array constructor: `[expr]` — collects all outputs from a generator
    /// into an array, producing a single array value (1→1).
    ///
    /// The `generator` is an expression that may produce 0 or more outputs
    /// (e.g., `.[]`, `, `, `select(f)`, `empty`).  Each output is appended
    /// to a result array.  The `acc_index` is the variable slot used for
    /// the accumulator array.
    /// (Phase 5-1: array constructor)
    Collect {
        generator: Box<Expr>,
        acc_index: u16,
    },

    /// Alternative operator: `primary // fallback`.
    ///
    /// Evaluates `primary`.  If the result is neither `null` nor `false`
    /// (i.e., truthy), returns it.  Otherwise evaluates `fallback` and
    /// returns that result.
    /// (Phase 5-2: alternative operator)
    Alternative {
        primary: Box<Expr>,
        fallback: Box<Expr>,
    },

    /// Object insert: inserts a key-value pair into an object.
    ///
    /// Corresponds to jq's `INSERT` opcode used in object construction
    /// patterns like `{a: .foo, b: .bar}`.
    ///
    /// Stack semantics: [input, obj, key, value] → [input, updated_obj]
    /// In IR: ObjectInsert { obj, key, value } → new object with key-value inserted.
    /// (Phase 8-4: object construction)
    ObjectInsert {
        obj: Box<Expr>,
        key: Box<Expr>,
        value: Box<Expr>,
    },

    /// Range generator: yields numeric values from `from` up to (but not including) `to`.
    ///
    /// Corresponds to jq's `range(n)` / `range(from;to)` / `range(from;to;step)` via the RANGE opcode.
    /// The counter starts at `from` and increments by `step` (default 1) until it reaches `to`.
    /// For negative step, counts down until counter <= to.
    /// (Phase 9-2: range)
    Range {
        from: Box<Expr>,
        to: Box<Expr>,
        step: Option<Box<Expr>>,
    },

    /// While generator: `while(cond; update)`.
    ///
    /// Starting with `input_expr`, while `cond` is truthy, yields the current
    /// value then applies `update` to get the next value.
    /// `cond` and `update` use Input to reference the current accumulator.
    /// (Phase 11: while/until)
    While {
        input_expr: Box<Expr>,
        cond: Box<Expr>,
        update: Box<Expr>,
    },

    /// Until expression: `until(cond; update)`.
    ///
    /// Starting with `input_expr`, while `cond` is falsy, applies `update`.
    /// Returns the first value where `cond` is truthy.
    /// `cond` and `update` use Input to reference the current accumulator.
    /// (Phase 11: while/until)
    Until {
        input_expr: Box<Expr>,
        cond: Box<Expr>,
        update: Box<Expr>,
    },

    /// Recursive descent: `..` — yields the input itself and all sub-values
    /// recursively (depth-first). For arrays, recurses into each element.
    /// For objects, recurses into each value. Scalars yield only themselves.
    ///
    /// When `body` is `Input`, yields each value directly.
    /// When `body` is something else (e.g., a select/pipe), applies body
    /// to each recursed value (modeling `.. | body`).
    ///
    /// Corresponds to jq's `..` / `recurse` builtin.
    /// (Phase 9-4: recursive descent)
    Recurse {
        input_expr: Box<Expr>,
        body: Box<Expr>,
    },

    /// Closure-based array operation: applies `key_expr` to each element
    /// of the input array to compute keys, then performs `op` using those keys.
    ///
    /// `input_expr` evaluates to the array to operate on.
    /// `key_expr` is evaluated with each element as `Input`, producing a key.
    ///
    /// Corresponds to jq's sort_by(f), group_by(f), unique_by(f), min_by(f), max_by(f).
    /// (Phase 9-1: closure builtins)
    ClosureApply {
        op: ClosureOp,
        input_expr: Box<Expr>,
        key_expr: Box<Expr>,
    },

    /// Set a value at a given path: `setpath(path; value)`.
    ///
    /// `input_expr` evaluates to the input value to modify.
    /// `path` evaluates to the path array (e.g., ["a", "b"]).
    /// `value` evaluates to the value to set.
    ///
    /// (Phase 9-6: path operations)
    SetPath {
        input_expr: Box<Expr>,
        path: Box<Expr>,
        value: Box<Expr>,
    },

    /// Regex test: `test(re)` / `test(re; flags)`.
    /// Returns true if the input string matches the regex.
    /// (Phase 10-2: regex)
    RegexTest {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        flags: Box<Expr>,
    },

    /// Regex match: `match(re)` / `match(re; flags)`.
    /// Returns match object(s). When flags contain "g", this is a generator.
    /// (Phase 10-2: regex)
    RegexMatch {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        flags: Box<Expr>,
    },

    /// Regex capture: `capture(re)` / `capture(re; flags)`.
    /// Returns object of named captures: {"name1": "val1", ...}.
    /// (Phase 10-2: regex)
    RegexCapture {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        flags: Box<Expr>,
    },

    /// Regex scan: `scan(re)` / `scan(re; flags)`.
    /// Generator: yields each match as a string (no groups) or array (with groups).
    /// (Phase 10-2: regex)
    RegexScan {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        flags: Box<Expr>,
    },

    /// Regex sub: `sub(re; tostr)` / `sub(re; tostr; flags)`.
    /// Replaces the first match of re with tostr.
    /// (Phase 10-2: regex)
    RegexSub {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        tostr: Box<Expr>,
        flags: Box<Expr>,
    },

    /// Regex gsub: `gsub(re; tostr)` / `gsub(re; tostr; flags)`.
    /// Replaces all matches of re with tostr.
    /// (Phase 10-2: regex)
    RegexGsub {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        tostr: Box<Expr>,
        flags: Box<Expr>,
    },

    /// Path extraction: `path(expr)` — generator that yields all paths
    /// accessed by `expr` on the input value.
    ///
    /// `input_expr` evaluates to the value to inspect.
    /// `path_expr` is the expression whose access paths are extracted.
    ///
    /// For `path(.a.b)` → yields `["a","b"]`
    /// For `path(.a)` → yields `["a"]`
    /// For `path(.[])` → yields `[0]`, `[1]`, ... for each element
    /// For `path(..)` → yields all recursive paths
    ///
    /// (Phase 9-5: path operations)
    PathOf {
        input_expr: Box<Expr>,
        path_expr: Box<Expr>,
    },

    /// Update expression: `path_expr |= update_expr`
    ///
    /// For each path that `path_expr` accesses on the input, applies `update_expr`
    /// to the value at that path and sets the result back.
    ///
    /// `input_expr` evaluates to the value to update.
    /// `path_expr` is the expression defining which paths to update (e.g., `.a`, `.[]`).
    /// `update_expr` is applied to the current value at each path.
    ///
    /// (Phase 10-3: assignment operators)
    Update {
        input_expr: Box<Expr>,
        path_expr: Box<Expr>,
        update_expr: Box<Expr>,
        /// If true, this is a plain assignment (=) where update_expr should be
        /// evaluated against the original input, not the getpath result.
        /// If false, this is an update assignment (|=) where update_expr is
        /// applied to each getpath result.
        #[allow(dead_code)]
        is_plain_assign: bool,
    },

    /// Limit generator: `limit(n; gen)` — yields at most `count` outputs from `generator`.
    ///
    /// `count` evaluates to the maximum number of outputs (scalar expression).
    /// `generator` is the expression whose outputs are limited.
    ///
    /// (Phase 12: limit/first/skip)
    Limit {
        count: Box<Expr>,
        generator: Box<Expr>,
    },

    /// Skip generator: `skip(n; gen)` — skips the first `count` outputs from `generator`.
    ///
    /// `count` evaluates to the number of outputs to skip (scalar expression).
    /// `generator` is the expression whose initial outputs are skipped.
    ///
    /// (Phase 12: limit/first/skip)
    Skip {
        count: Box<Expr>,
        generator: Box<Expr>,
    },
}

/// A constant literal value in the IR.
#[derive(Debug, Clone)]
pub enum Literal {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    /// An empty array literal `[]`.
    /// Used as the initial accumulator for array constructors `[expr]`.
    /// (Phase 5-1)
    EmptyArr,
    /// A pre-compiled non-empty array literal (e.g., `[1,2,3]`).
    /// LOADK loads these directly as constants.
    /// (Phase 8-8)
    Arr(Vec<crate::value::Value>),
    /// An empty object literal `{}`.
    /// (Phase 8-8)
    EmptyObj,
    /// A pre-compiled non-empty object literal.
    Obj(std::collections::BTreeMap<String, crate::value::Value>),
    /// An error literal with a message string.
    /// Used for ERRORK opcode.
    /// (Phase 10-4)
    Error(String),
}

/// Binary operators (arithmetic + comparison + nargs=2 builtins).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    // Arithmetic (Phase 1)
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // Comparison (Phase 2)
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    // Phase 5-2: nargs=2 builtins (input, arg → result)
    Split,
    Has,
    StartsWith,
    EndsWith,
    Join,
    Contains,
    Ltrimstr,
    Rtrimstr,
    In,
    // Phase 9-6: Remaining builtins
    Inside,
    Indices,
    StrIndex,
    StrRindex,
    GetPath,
    DelPaths,
    FlattenDepth,
    // Math binary functions (2-arg)
    Pow,
    Atan2,
    Drem,
    Ldexp,
    Scalb,
    Scalbln,
    // Binary search
    Bsearch,
    // Date/time with format argument
    Strftime,
    Strptime,
    Strflocaltime,
}

/// Unary operators (Phase 2+).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Length,
    Type,
    ToString,
    ToNumber,
    Keys,
    Negate,
    // Phase 5-2: additional unary builtins
    Sort,
    KeysUnsorted,
    Floor,
    Ceil,
    Round,
    Fabs,
    Explode,
    Implode,
    Reverse,
    Unique,
    Flatten,
    Min,
    Max,
    AsciiDowncase,
    AsciiUpcase,
    ToEntries,
    FromEntries,
    Values,
    Add,
    Not,
    Ascii,
    // Phase 9-3: Format string operations (@base64, @html, etc.)
    FormatBase64,
    FormatBase64d,
    FormatHtml,
    FormatUri,
    FormatUrid,
    FormatCsv,
    FormatTsv,
    FormatJson,
    FormatSh,
    // Phase 9-6: Remaining builtins
    Any,
    All,
    ToJson,
    FromJson,
    Debug,
    Env,
    Builtins,
    Infinite,
    Nan,
    IsInfinite,
    IsNan,
    IsNormal,
    // Phase 11: Make error value from input
    MakeError,
    // Math functions
    Sqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Exp,
    Exp2,
    Exp10,
    Log,
    Log2,
    Log10,
    Cbrt,
    Significand,
    Exponent,
    Logb,
    NearbyInt,
    Trunc,
    Rint,
    J0,
    J1,
    // Transpose (2D array)
    Transpose,
    // UTF-8 byte length
    Utf8ByteLength,
    // toboolean
    ToBoolean,
    // String trimming
    Trim,
    Ltrim,
    Rtrim,
    // Date/time
    Gmtime,
    Mktime,
    Now,
}

/// Closure-based array operations (Phase 9-1).
///
/// These operations take an array and a key function (closure) as arguments.
/// The key function is applied to each element to compute a sort/grouping key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureOp {
    SortBy,
    GroupBy,
    UniqueBy,
    MinBy,
    MaxBy,
}

// ---------------------------------------------------------------------------
// Display implementations for human-readable IR output
// ---------------------------------------------------------------------------

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Input => write!(f, "."),
            Expr::Literal(lit) => write!(f, "{}", lit),
            Expr::BinOp { op, lhs, rhs } => {
                write!(f, "({} {} {})", lhs, op, rhs)
            }
            Expr::Index { expr, key } => {
                write!(f, "{}[{}]", expr, key)
            }
            Expr::IndexOpt { expr, key } => {
                write!(f, "{}[{}]?", expr, key)
            }
            Expr::UnaryOp { op, operand } => {
                write!(f, "{}({})", op, operand)
            }
            Expr::IfThenElse {
                cond,
                then_branch,
                else_branch,
            } => {
                write!(
                    f,
                    "if {} then {} else {} end",
                    cond, then_branch, else_branch
                )
            }
            Expr::TryCatch {
                try_expr,
                catch_expr,
            } => {
                write!(f, "try {} catch {}", try_expr, catch_expr)
            }
            Expr::Comma { left, right } => {
                write!(f, "({}, {})", left, right)
            }
            Expr::Empty => {
                write!(f, "empty")
            }
            Expr::Each { input_expr, body } => {
                write!(f, "each({}, {})", input_expr, body)
            }
            Expr::EachOpt { input_expr, body } => {
                write!(f, "each_opt({}, {})", input_expr, body)
            }
            Expr::LetBinding { var_index, value, body } => {
                write!(f, "let $v{} = {} in {}", var_index, value, body)
            }
            Expr::LoadVar { var_index } => {
                write!(f, "$v{}", var_index)
            }
            Expr::Reduce { source, init, var_index, acc_index, update } => {
                write!(
                    f,
                    "reduce {} as $v{} ({}; $v{} | {})",
                    source, var_index, init, acc_index, update
                )
            }
            Expr::Foreach { source, init, var_index, acc_index, update } => {
                write!(
                    f,
                    "foreach {} as $v{} ({}; $v{} | {})",
                    source, var_index, init, acc_index, update
                )
            }
            Expr::Collect { generator, acc_index } => {
                write!(f, "collect($v{}, {})", acc_index, generator)
            }
            Expr::Alternative { primary, fallback } => {
                write!(f, "({} // {})", primary, fallback)
            }
            Expr::ObjectInsert { obj, key, value } => {
                write!(f, "obj_insert({}, {}, {})", obj, key, value)
            }
            Expr::Range { from, to, step } => {
                if let Some(step) = step {
                    write!(f, "range({}, {}, {})", from, to, step)
                } else {
                    write!(f, "range({}, {})", from, to)
                }
            }
            Expr::While { input_expr, cond, update } => {
                write!(f, "while({}, {}, {})", input_expr, cond, update)
            }
            Expr::Until { input_expr, cond, update } => {
                write!(f, "until({}, {}, {})", input_expr, cond, update)
            }
            Expr::Recurse { input_expr, body } => {
                if matches!(body.as_ref(), Expr::Input) {
                    write!(f, "recurse({})", input_expr)
                } else {
                    write!(f, "recurse({} | {})", input_expr, body)
                }
            }
            Expr::ClosureApply { op, input_expr, key_expr } => {
                write!(f, "{}({}, {})", op, input_expr, key_expr)
            }
            Expr::SetPath { input_expr, path, value } => {
                write!(f, "setpath({}, {}, {})", input_expr, path, value)
            }
            Expr::RegexTest { input_expr, re, flags } => {
                write!(f, "test({}, {}, {})", input_expr, re, flags)
            }
            Expr::RegexMatch { input_expr, re, flags } => {
                write!(f, "match({}, {}, {})", input_expr, re, flags)
            }
            Expr::RegexCapture { input_expr, re, flags } => {
                write!(f, "capture({}, {}, {})", input_expr, re, flags)
            }
            Expr::RegexScan { input_expr, re, flags } => {
                write!(f, "scan({}, {}, {})", input_expr, re, flags)
            }
            Expr::RegexSub { input_expr, re, tostr, flags } => {
                write!(f, "sub({}, {}, {}, {})", input_expr, re, tostr, flags)
            }
            Expr::RegexGsub { input_expr, re, tostr, flags } => {
                write!(f, "gsub({}, {}, {}, {})", input_expr, re, tostr, flags)
            }
            Expr::PathOf { input_expr, path_expr } => {
                write!(f, "path({}, {})", input_expr, path_expr)
            }
            Expr::Update { input_expr, path_expr, update_expr, is_plain_assign } => {
                if *is_plain_assign {
                    write!(f, "assign({}, {}, {})", input_expr, path_expr, update_expr)
                } else {
                    write!(f, "update({}, {}, {})", input_expr, path_expr, update_expr)
                }
            }
            Expr::Limit { count, generator } => {
                write!(f, "limit({}, {})", count, generator)
            }
            Expr::Skip { count, generator } => {
                write!(f, "skip({}, {})", count, generator)
            }
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Null => write!(f, "null"),
            Literal::Bool(b) => write!(f, "{}", b),
            Literal::Num(n) => {
                // Format integer-valued doubles without decimal point
                if *n == n.trunc() && n.abs() < 1e15 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            Literal::Str(s) => write!(f, "\"{}\"", s),
            Literal::EmptyArr => write!(f, "[]"),
            Literal::Arr(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Literal::EmptyObj => write!(f, "{{}}"),
            Literal::Obj(m) => {
                write!(f, "{{")?;
                for (i, (k, _v)) in m.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "\"{}\":...", k)?;
                }
                write!(f, "}}")
            }
            Literal::Error(msg) => write!(f, "error(\"{}\")", msg),
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Ne => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Le => write!(f, "<="),
            BinOp::Ge => write!(f, ">="),
            BinOp::Split => write!(f, "split"),
            BinOp::Has => write!(f, "has"),
            BinOp::StartsWith => write!(f, "startswith"),
            BinOp::EndsWith => write!(f, "endswith"),
            BinOp::Join => write!(f, "join"),
            BinOp::Contains => write!(f, "contains"),
            BinOp::Ltrimstr => write!(f, "ltrimstr"),
            BinOp::Rtrimstr => write!(f, "rtrimstr"),
            BinOp::In => write!(f, "in"),
            BinOp::Inside => write!(f, "inside"),
            BinOp::Indices => write!(f, "indices"),
            BinOp::StrIndex => write!(f, "index"),
            BinOp::StrRindex => write!(f, "rindex"),
            BinOp::GetPath => write!(f, "getpath"),
            BinOp::DelPaths => write!(f, "delpaths"),
            BinOp::FlattenDepth => write!(f, "flatten"),
            BinOp::Pow => write!(f, "pow"),
            BinOp::Atan2 => write!(f, "atan2"),
            BinOp::Drem => write!(f, "drem"),
            BinOp::Ldexp => write!(f, "ldexp"),
            BinOp::Scalb => write!(f, "scalb"),
            BinOp::Scalbln => write!(f, "scalbln"),
            BinOp::Bsearch => write!(f, "bsearch"),
            BinOp::Strftime => write!(f, "strftime"),
            BinOp::Strptime => write!(f, "strptime"),
            BinOp::Strflocaltime => write!(f, "strflocaltime"),
        }
    }
}

impl fmt::Display for ClosureOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClosureOp::SortBy => write!(f, "sort_by"),
            ClosureOp::GroupBy => write!(f, "group_by"),
            ClosureOp::UniqueBy => write!(f, "unique_by"),
            ClosureOp::MinBy => write!(f, "min_by"),
            ClosureOp::MaxBy => write!(f, "max_by"),
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Length => write!(f, "length"),
            UnaryOp::Type => write!(f, "type"),
            UnaryOp::ToString => write!(f, "tostring"),
            UnaryOp::ToNumber => write!(f, "tonumber"),
            UnaryOp::Keys => write!(f, "keys"),
            UnaryOp::Negate => write!(f, "negate"),
            UnaryOp::Sort => write!(f, "sort"),
            UnaryOp::KeysUnsorted => write!(f, "keys_unsorted"),
            UnaryOp::Floor => write!(f, "floor"),
            UnaryOp::Ceil => write!(f, "ceil"),
            UnaryOp::Round => write!(f, "round"),
            UnaryOp::Fabs => write!(f, "fabs"),
            UnaryOp::Explode => write!(f, "explode"),
            UnaryOp::Implode => write!(f, "implode"),
            UnaryOp::Reverse => write!(f, "reverse"),
            UnaryOp::Unique => write!(f, "unique"),
            UnaryOp::Flatten => write!(f, "flatten"),
            UnaryOp::Min => write!(f, "min"),
            UnaryOp::Max => write!(f, "max"),
            UnaryOp::AsciiDowncase => write!(f, "ascii_downcase"),
            UnaryOp::AsciiUpcase => write!(f, "ascii_upcase"),
            UnaryOp::ToEntries => write!(f, "to_entries"),
            UnaryOp::FromEntries => write!(f, "from_entries"),
            UnaryOp::Values => write!(f, "values"),
            UnaryOp::Add => write!(f, "add"),
            UnaryOp::Not => write!(f, "not"),
            UnaryOp::Ascii => write!(f, "ascii"),
            UnaryOp::FormatBase64 => write!(f, "@base64"),
            UnaryOp::FormatBase64d => write!(f, "@base64d"),
            UnaryOp::FormatHtml => write!(f, "@html"),
            UnaryOp::FormatUri => write!(f, "@uri"),
            UnaryOp::FormatUrid => write!(f, "@urid"),
            UnaryOp::FormatCsv => write!(f, "@csv"),
            UnaryOp::FormatTsv => write!(f, "@tsv"),
            UnaryOp::FormatJson => write!(f, "@json"),
            UnaryOp::FormatSh => write!(f, "@sh"),
            UnaryOp::Any => write!(f, "any"),
            UnaryOp::All => write!(f, "all"),
            UnaryOp::ToJson => write!(f, "tojson"),
            UnaryOp::FromJson => write!(f, "fromjson"),
            UnaryOp::Debug => write!(f, "debug"),
            UnaryOp::Env => write!(f, "env"),
            UnaryOp::Builtins => write!(f, "builtins"),
            UnaryOp::Infinite => write!(f, "infinite"),
            UnaryOp::Nan => write!(f, "nan"),
            UnaryOp::IsInfinite => write!(f, "isinfinite"),
            UnaryOp::IsNan => write!(f, "isnan"),
            UnaryOp::IsNormal => write!(f, "isnormal"),
            UnaryOp::MakeError => write!(f, "error"),
            UnaryOp::Sqrt => write!(f, "sqrt"),
            UnaryOp::Sin => write!(f, "sin"),
            UnaryOp::Cos => write!(f, "cos"),
            UnaryOp::Tan => write!(f, "tan"),
            UnaryOp::Asin => write!(f, "asin"),
            UnaryOp::Acos => write!(f, "acos"),
            UnaryOp::Atan => write!(f, "atan"),
            UnaryOp::Exp => write!(f, "exp"),
            UnaryOp::Exp2 => write!(f, "exp2"),
            UnaryOp::Exp10 => write!(f, "exp10"),
            UnaryOp::Log => write!(f, "log"),
            UnaryOp::Log2 => write!(f, "log2"),
            UnaryOp::Log10 => write!(f, "log10"),
            UnaryOp::Cbrt => write!(f, "cbrt"),
            UnaryOp::Significand => write!(f, "significand"),
            UnaryOp::Exponent => write!(f, "exponent"),
            UnaryOp::Logb => write!(f, "logb"),
            UnaryOp::NearbyInt => write!(f, "nearbyint"),
            UnaryOp::Trunc => write!(f, "trunc"),
            UnaryOp::Rint => write!(f, "rint"),
            UnaryOp::J0 => write!(f, "j0"),
            UnaryOp::J1 => write!(f, "j1"),
            UnaryOp::Transpose => write!(f, "transpose"),
            UnaryOp::Utf8ByteLength => write!(f, "utf8bytelength"),
            UnaryOp::ToBoolean => write!(f, "toboolean"),
            UnaryOp::Trim => write!(f, "trim"),
            UnaryOp::Ltrim => write!(f, "ltrim"),
            UnaryOp::Rtrim => write!(f, "rtrim"),
            UnaryOp::Gmtime => write!(f, "gmtime"),
            UnaryOp::Mktime => write!(f, "mktime"),
            UnaryOp::Now => write!(f, "now"),
        }
    }
}

// ---------------------------------------------------------------------------
// IR transformations
// ---------------------------------------------------------------------------

/// Replace all `Expr::Input` nodes in `expr` with `replacement`.
///
/// Used for inlining subfunctions: the subfunction's IR uses `Expr::Input`
/// to refer to "the input value passed to this subfunction".  When inlining,
/// we replace it with the actual expression from the caller's stack.
pub fn substitute_input(expr: &Expr, replacement: &Expr) -> Expr {
    match expr {
        Expr::Input => replacement.clone(),
        Expr::Literal(_) => expr.clone(),
        Expr::BinOp { op, lhs, rhs } => Expr::BinOp {
            op: *op,
            lhs: Box::new(substitute_input(lhs, replacement)),
            rhs: Box::new(substitute_input(rhs, replacement)),
        },
        Expr::Index { expr: e, key } => Expr::Index {
            expr: Box::new(substitute_input(e, replacement)),
            key: Box::new(substitute_input(key, replacement)),
        },
        Expr::IndexOpt { expr: e, key } => Expr::IndexOpt {
            expr: Box::new(substitute_input(e, replacement)),
            key: Box::new(substitute_input(key, replacement)),
        },
        Expr::UnaryOp { op, operand } => Expr::UnaryOp {
            op: *op,
            operand: Box::new(substitute_input(operand, replacement)),
        },
        Expr::IfThenElse {
            cond,
            then_branch,
            else_branch,
        } => Expr::IfThenElse {
            cond: Box::new(substitute_input(cond, replacement)),
            then_branch: Box::new(substitute_input(then_branch, replacement)),
            else_branch: Box::new(substitute_input(else_branch, replacement)),
        },
        Expr::TryCatch {
            try_expr,
            catch_expr,
        } => Expr::TryCatch {
            try_expr: Box::new(substitute_input(try_expr, replacement)),
            catch_expr: Box::new(substitute_input(catch_expr, replacement)),
        },
        Expr::Comma { left, right } => Expr::Comma {
            left: Box::new(substitute_input(left, replacement)),
            right: Box::new(substitute_input(right, replacement)),
        },
        Expr::Empty => Expr::Empty,
        Expr::Each { input_expr, body } => Expr::Each {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            // body's Input refers to each element of the iterable, NOT the outer input.
            // Do NOT substitute into body — it has its own Input scope.
            body: body.clone(),
        },
        Expr::EachOpt { input_expr, body } => Expr::EachOpt {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            body: body.clone(),
        },
        Expr::LetBinding { var_index, value, body } => Expr::LetBinding {
            var_index: *var_index,
            value: Box::new(substitute_input(value, replacement)),
            body: Box::new(substitute_input(body, replacement)),
        },
        Expr::LoadVar { var_index } => Expr::LoadVar {
            var_index: *var_index,
        },
        Expr::Reduce { source, init, var_index, acc_index, update } => Expr::Reduce {
            source: Box::new(substitute_input(source, replacement)),
            init: Box::new(substitute_input(init, replacement)),
            var_index: *var_index,
            acc_index: *acc_index,
            // update's Input refers to the accumulator, NOT the outer input.
            update: update.clone(),
        },
        Expr::Foreach { source, init, var_index, acc_index, update } => Expr::Foreach {
            source: Box::new(substitute_input(source, replacement)),
            init: Box::new(substitute_input(init, replacement)),
            var_index: *var_index,
            acc_index: *acc_index,
            // update's Input refers to the accumulator, NOT the outer input.
            update: update.clone(),
        },
        Expr::Collect { generator, acc_index } => Expr::Collect {
            // generator's body contains Each whose body has its own Input scope.
            // However, the generator itself may reference outer Input (e.g., the
            // iterable). We need to substitute into generator, but Each's body
            // inside will be protected by the Each case above.
            generator: Box::new(substitute_input(generator, replacement)),
            acc_index: *acc_index,
        },
        Expr::Alternative { primary, fallback } => Expr::Alternative {
            primary: Box::new(substitute_input(primary, replacement)),
            fallback: Box::new(substitute_input(fallback, replacement)),
        },
        Expr::ObjectInsert { obj, key, value } => Expr::ObjectInsert {
            obj: Box::new(substitute_input(obj, replacement)),
            key: Box::new(substitute_input(key, replacement)),
            value: Box::new(substitute_input(value, replacement)),
        },
        Expr::Range { from, to, step } => Expr::Range {
            from: Box::new(substitute_input(from, replacement)),
            to: Box::new(substitute_input(to, replacement)),
            step: step.as_ref().map(|s| Box::new(substitute_input(s, replacement))),
        },
        Expr::While { input_expr, cond, update } => Expr::While {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            // cond and update have Input referring to the current accumulator, NOT the outer input
            cond: cond.clone(),
            update: update.clone(),
        },
        Expr::Until { input_expr, cond, update } => Expr::Until {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            cond: cond.clone(),
            update: update.clone(),
        },
        Expr::Recurse { input_expr, body } => Expr::Recurse {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            body: Box::new(substitute_input(body, replacement)),
        },
        Expr::ClosureApply { op, input_expr, key_expr } => Expr::ClosureApply {
            op: *op,
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            // key_expr's Input refers to each element, NOT the outer input.
            // Do NOT substitute into key_expr — it has its own Input scope.
            key_expr: key_expr.clone(),
        },
        Expr::SetPath { input_expr, path, value } => Expr::SetPath {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            path: Box::new(substitute_input(path, replacement)),
            value: Box::new(substitute_input(value, replacement)),
        },
        Expr::RegexTest { input_expr, re, flags } => Expr::RegexTest {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            re: Box::new(substitute_input(re, replacement)),
            flags: Box::new(substitute_input(flags, replacement)),
        },
        Expr::RegexMatch { input_expr, re, flags } => Expr::RegexMatch {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            re: Box::new(substitute_input(re, replacement)),
            flags: Box::new(substitute_input(flags, replacement)),
        },
        Expr::RegexCapture { input_expr, re, flags } => Expr::RegexCapture {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            re: Box::new(substitute_input(re, replacement)),
            flags: Box::new(substitute_input(flags, replacement)),
        },
        Expr::RegexScan { input_expr, re, flags } => Expr::RegexScan {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            re: Box::new(substitute_input(re, replacement)),
            flags: Box::new(substitute_input(flags, replacement)),
        },
        Expr::RegexSub { input_expr, re, tostr, flags } => Expr::RegexSub {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            re: Box::new(substitute_input(re, replacement)),
            tostr: Box::new(substitute_input(tostr, replacement)),
            flags: Box::new(substitute_input(flags, replacement)),
        },
        Expr::RegexGsub { input_expr, re, tostr, flags } => Expr::RegexGsub {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            re: Box::new(substitute_input(re, replacement)),
            tostr: Box::new(substitute_input(tostr, replacement)),
            flags: Box::new(substitute_input(flags, replacement)),
        },
        Expr::PathOf { input_expr, path_expr } => Expr::PathOf {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            // path_expr's Input refers to the value being path-inspected, not the outer input
            path_expr: path_expr.clone(),
        },
        Expr::Update { input_expr, path_expr, update_expr, is_plain_assign } => Expr::Update {
            input_expr: Box::new(substitute_input(input_expr, replacement)),
            // path_expr and update_expr have their own Input scope
            path_expr: path_expr.clone(),
            update_expr: update_expr.clone(),
            is_plain_assign: *is_plain_assign,
        },
        Expr::Limit { count, generator } => Expr::Limit {
            count: Box::new(substitute_input(count, replacement)),
            // generator's Input refers to the generator's own input scope
            generator: generator.clone(),
        },
        Expr::Skip { count, generator } => Expr::Skip {
            count: Box::new(substitute_input(count, replacement)),
            // generator's Input refers to the generator's own input scope
            generator: generator.clone(),
        },
    }
}

// ---------------------------------------------------------------------------
// Structural equality for testing
// ---------------------------------------------------------------------------

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Expr::Input, Expr::Input) => true,
            (Expr::Literal(a), Expr::Literal(b)) => a == b,
            (
                Expr::BinOp {
                    op: op1,
                    lhs: l1,
                    rhs: r1,
                },
                Expr::BinOp {
                    op: op2,
                    lhs: l2,
                    rhs: r2,
                },
            ) => op1 == op2 && l1 == l2 && r1 == r2,
            (
                Expr::Index {
                    expr: e1,
                    key: k1,
                },
                Expr::Index {
                    expr: e2,
                    key: k2,
                },
            ) => e1 == e2 && k1 == k2,
            (
                Expr::IndexOpt {
                    expr: e1,
                    key: k1,
                },
                Expr::IndexOpt {
                    expr: e2,
                    key: k2,
                },
            ) => e1 == e2 && k1 == k2,
            (
                Expr::UnaryOp {
                    op: op1,
                    operand: o1,
                },
                Expr::UnaryOp {
                    op: op2,
                    operand: o2,
                },
            ) => op1 == op2 && o1 == o2,
            (
                Expr::IfThenElse {
                    cond: c1,
                    then_branch: t1,
                    else_branch: e1,
                },
                Expr::IfThenElse {
                    cond: c2,
                    then_branch: t2,
                    else_branch: e2,
                },
            ) => c1 == c2 && t1 == t2 && e1 == e2,
            (
                Expr::TryCatch {
                    try_expr: t1,
                    catch_expr: c1,
                },
                Expr::TryCatch {
                    try_expr: t2,
                    catch_expr: c2,
                },
            ) => t1 == t2 && c1 == c2,
            (
                Expr::Comma {
                    left: l1,
                    right: r1,
                },
                Expr::Comma {
                    left: l2,
                    right: r2,
                },
            ) => l1 == l2 && r1 == r2,
            (Expr::Empty, Expr::Empty) => true,
            (
                Expr::Each {
                    input_expr: i1,
                    body: b1,
                },
                Expr::Each {
                    input_expr: i2,
                    body: b2,
                },
            ) => i1 == i2 && b1 == b2,
            (
                Expr::EachOpt {
                    input_expr: i1,
                    body: b1,
                },
                Expr::EachOpt {
                    input_expr: i2,
                    body: b2,
                },
            ) => i1 == i2 && b1 == b2,
            (
                Expr::LetBinding {
                    var_index: v1,
                    value: val1,
                    body: b1,
                },
                Expr::LetBinding {
                    var_index: v2,
                    value: val2,
                    body: b2,
                },
            ) => v1 == v2 && val1 == val2 && b1 == b2,
            (
                Expr::LoadVar { var_index: v1 },
                Expr::LoadVar { var_index: v2 },
            ) => v1 == v2,
            (
                Expr::Reduce {
                    source: s1,
                    init: i1,
                    var_index: v1,
                    acc_index: a1,
                    update: u1,
                },
                Expr::Reduce {
                    source: s2,
                    init: i2,
                    var_index: v2,
                    acc_index: a2,
                    update: u2,
                },
            ) => s1 == s2 && i1 == i2 && v1 == v2 && a1 == a2 && u1 == u2,
            (
                Expr::Foreach {
                    source: s1,
                    init: i1,
                    var_index: v1,
                    acc_index: a1,
                    update: u1,
                },
                Expr::Foreach {
                    source: s2,
                    init: i2,
                    var_index: v2,
                    acc_index: a2,
                    update: u2,
                },
            ) => s1 == s2 && i1 == i2 && v1 == v2 && a1 == a2 && u1 == u2,
            (
                Expr::Collect {
                    generator: g1,
                    acc_index: a1,
                },
                Expr::Collect {
                    generator: g2,
                    acc_index: a2,
                },
            ) => g1 == g2 && a1 == a2,
            (
                Expr::Alternative {
                    primary: p1,
                    fallback: f1,
                },
                Expr::Alternative {
                    primary: p2,
                    fallback: f2,
                },
            ) => p1 == p2 && f1 == f2,
            (
                Expr::ObjectInsert {
                    obj: o1,
                    key: k1,
                    value: v1,
                },
                Expr::ObjectInsert {
                    obj: o2,
                    key: k2,
                    value: v2,
                },
            ) => o1 == o2 && k1 == k2 && v1 == v2,
            (
                Expr::Range { from: f1, to: t1, step: s1 },
                Expr::Range { from: f2, to: t2, step: s2 },
            ) => f1 == f2 && t1 == t2 && s1 == s2,
            (
                Expr::While { input_expr: i1, cond: c1, update: u1 },
                Expr::While { input_expr: i2, cond: c2, update: u2 },
            ) => i1 == i2 && c1 == c2 && u1 == u2,
            (
                Expr::Until { input_expr: i1, cond: c1, update: u1 },
                Expr::Until { input_expr: i2, cond: c2, update: u2 },
            ) => i1 == i2 && c1 == c2 && u1 == u2,
            (
                Expr::Recurse { input_expr: i1, body: b1 },
                Expr::Recurse { input_expr: i2, body: b2 },
            ) => i1 == i2 && b1 == b2,
            (
                Expr::ClosureApply { op: o1, input_expr: i1, key_expr: k1 },
                Expr::ClosureApply { op: o2, input_expr: i2, key_expr: k2 },
            ) => o1 == o2 && i1 == i2 && k1 == k2,
            (
                Expr::SetPath { input_expr: i1, path: p1, value: v1 },
                Expr::SetPath { input_expr: i2, path: p2, value: v2 },
            ) => i1 == i2 && p1 == p2 && v1 == v2,
            (
                Expr::RegexTest { input_expr: i1, re: r1, flags: f1 },
                Expr::RegexTest { input_expr: i2, re: r2, flags: f2 },
            ) => i1 == i2 && r1 == r2 && f1 == f2,
            (
                Expr::RegexMatch { input_expr: i1, re: r1, flags: f1 },
                Expr::RegexMatch { input_expr: i2, re: r2, flags: f2 },
            ) => i1 == i2 && r1 == r2 && f1 == f2,
            (
                Expr::RegexCapture { input_expr: i1, re: r1, flags: f1 },
                Expr::RegexCapture { input_expr: i2, re: r2, flags: f2 },
            ) => i1 == i2 && r1 == r2 && f1 == f2,
            (
                Expr::RegexScan { input_expr: i1, re: r1, flags: f1 },
                Expr::RegexScan { input_expr: i2, re: r2, flags: f2 },
            ) => i1 == i2 && r1 == r2 && f1 == f2,
            (
                Expr::RegexSub { input_expr: i1, re: r1, tostr: t1, flags: f1 },
                Expr::RegexSub { input_expr: i2, re: r2, tostr: t2, flags: f2 },
            ) => i1 == i2 && r1 == r2 && t1 == t2 && f1 == f2,
            (
                Expr::RegexGsub { input_expr: i1, re: r1, tostr: t1, flags: f1 },
                Expr::RegexGsub { input_expr: i2, re: r2, tostr: t2, flags: f2 },
            ) => i1 == i2 && r1 == r2 && t1 == t2 && f1 == f2,
            (
                Expr::PathOf { input_expr: i1, path_expr: p1 },
                Expr::PathOf { input_expr: i2, path_expr: p2 },
            ) => i1 == i2 && p1 == p2,
            (
                Expr::Update { input_expr: i1, path_expr: p1, update_expr: u1, is_plain_assign: a1 },
                Expr::Update { input_expr: i2, path_expr: p2, update_expr: u2, is_plain_assign: a2 },
            ) => i1 == i2 && p1 == p2 && u1 == u2 && a1 == a2,
            (
                Expr::Limit { count: c1, generator: g1 },
                Expr::Limit { count: c2, generator: g2 },
            ) => c1 == c2 && g1 == g2,
            (
                Expr::Skip { count: c1, generator: g1 },
                Expr::Skip { count: c2, generator: g2 },
            ) => c1 == c2 && g1 == g2,
            _ => false,
        }
    }
}

impl Eq for Expr {}

impl PartialEq for Literal {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Literal::Null, Literal::Null) => true,
            (Literal::Bool(a), Literal::Bool(b)) => a == b,
            (Literal::Num(a), Literal::Num(b)) => a.to_bits() == b.to_bits(),
            (Literal::Str(a), Literal::Str(b)) => a == b,
            (Literal::EmptyArr, Literal::EmptyArr) => true,
            (Literal::Arr(a), Literal::Arr(b)) => a == b,
            (Literal::EmptyObj, Literal::EmptyObj) => true,
            (Literal::Error(a), Literal::Error(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Literal {}
