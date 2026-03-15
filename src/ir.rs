//! Intermediate representation for jq filters.
//!
//! Every IR node is a generator: it takes an input Value and produces zero or more
//! output Values. This unified model eliminates the generator-in-scalar-context problem.

/// A filter expression in the IR.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Identity filter `.` - yields the input.
    Input,

    /// Literal constant.
    Literal(Literal),

    /// Binary operation: `lhs op rhs`
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    /// Unary operation (builtins with 1 arg)
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
    },

    /// Field/array index: `expr[key]`
    Index {
        expr: Box<Expr>,
        key: Box<Expr>,
    },

    /// Optional index: `expr[key]?` (no error on type mismatch)
    IndexOpt {
        expr: Box<Expr>,
        key: Box<Expr>,
    },

    /// Pipe: `left | right`
    /// Equivalent to: for each output of left, run right
    Pipe {
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Comma: `left, right` - yields all of left, then all of right
    Comma {
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Empty - yields nothing.
    Empty,

    /// If-then-else: `if cond then t else f end`
    IfThenElse {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },

    /// Try-catch: `try expr catch handler`
    TryCatch {
        try_expr: Box<Expr>,
        catch_expr: Box<Expr>,
    },

    /// Array iteration `.[]`
    Each {
        input_expr: Box<Expr>,
    },

    /// Optional array iteration `.[]?`
    EachOpt {
        input_expr: Box<Expr>,
    },

    /// Variable binding: `expr as $var | body`
    LetBinding {
        var_index: u16,
        value: Box<Expr>,
        body: Box<Expr>,
    },

    /// Variable reference: `$var`
    LoadVar {
        var_index: u16,
    },

    /// Reduce: `reduce source as $var (init; update)`
    Reduce {
        source: Box<Expr>,
        init: Box<Expr>,
        var_index: u16,
        acc_index: u16,
        update: Box<Expr>,
    },

    /// Foreach: `foreach source as $var (init; update; extract)`
    Foreach {
        source: Box<Expr>,
        init: Box<Expr>,
        var_index: u16,
        acc_index: u16,
        update: Box<Expr>,
        extract: Option<Box<Expr>>,
    },

    /// Array constructor: `[expr]`
    Collect {
        generator: Box<Expr>,
    },

    /// Object insert: `{key: value}`
    ObjectConstruct {
        pairs: Vec<(Expr, Expr)>,
    },

    /// Alternative operator: `expr // fallback`
    Alternative {
        primary: Box<Expr>,
        fallback: Box<Expr>,
    },

    /// Negation: `-expr`
    Negate {
        operand: Box<Expr>,
    },

    /// Recursive descent: `..`
    Recurse {
        input_expr: Box<Expr>,
    },

    /// Range: `range(from; to)` or `range(from; to; step)`
    Range {
        from: Box<Expr>,
        to: Box<Expr>,
        step: Option<Box<Expr>>,
    },

    /// Label-break: `label $name | body`
    Label {
        var_index: u16,
        body: Box<Expr>,
    },

    /// Break out of label
    Break {
        var_index: u16,
        value: Box<Expr>,
    },

    /// Update assignment: `path |= update`
    Update {
        path_expr: Box<Expr>,
        update_expr: Box<Expr>,
    },

    /// Plain assignment: `path = value`
    Assign {
        path_expr: Box<Expr>,
        value_expr: Box<Expr>,
    },

    /// Path expression: `path(expr)`
    PathExpr {
        expr: Box<Expr>,
    },

    /// Setpath: `setpath(path; value)`
    SetPath {
        path: Box<Expr>,
        value: Box<Expr>,
    },

    /// Getpath: `getpath(path)`
    GetPath {
        path: Box<Expr>,
    },

    /// Delpaths: `delpaths(paths)`
    DelPaths {
        paths: Box<Expr>,
    },

    /// User-defined function call
    FuncCall {
        func_id: usize,
        args: Vec<Expr>,
    },

    /// String interpolation
    StringInterpolation {
        parts: Vec<StringPart>,
    },

    /// Limit: `limit(n; generator)`
    Limit {
        count: Box<Expr>,
        generator: Box<Expr>,
    },

    /// While: `while(cond; update)`
    While {
        cond: Box<Expr>,
        update: Box<Expr>,
    },

    /// Until: `until(cond; update)`
    Until {
        cond: Box<Expr>,
        update: Box<Expr>,
    },

    /// Repeat: `repeat(update)`
    Repeat {
        update: Box<Expr>,
    },

    /// Short-circuit all(gen; cond)
    AllShort {
        generator: Box<Expr>,
        predicate: Box<Expr>,
    },

    /// Short-circuit any(gen; cond)
    AnyShort {
        generator: Box<Expr>,
        predicate: Box<Expr>,
    },

    /// Error: `error` or `error(msg)`
    Error {
        msg: Option<Box<Expr>>,
    },

    /// Format string: `@base64`, `@uri`, etc.
    Format {
        name: String,
        expr: Box<Expr>,
    },

    /// Closure operation: sort_by, group_by, etc.
    ClosureOp {
        op: ClosureOpKind,
        input_expr: Box<Expr>,
        key_expr: Box<Expr>,
    },

    /// Regex operations
    RegexTest {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        flags: Box<Expr>,
    },
    RegexMatch {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        flags: Box<Expr>,
    },
    RegexCapture {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        flags: Box<Expr>,
    },
    RegexScan {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        flags: Box<Expr>,
    },
    RegexSub {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        tostr: Box<Expr>,
        flags: Box<Expr>,
    },
    RegexGsub {
        input_expr: Box<Expr>,
        re: Box<Expr>,
        tostr: Box<Expr>,
        flags: Box<Expr>,
    },

    /// `?//` alternative destructuring operator
    AlternativeDestructure {
        alternatives: Vec<Expr>,
    },

    /// Slice: `.[from:to]`
    Slice {
        expr: Box<Expr>,
        from: Option<Box<Expr>>,
        to: Option<Box<Expr>>,
    },

    /// `not` operator
    Not,

    /// `$__loc__` special variable
    Loc {
        file: String,
        line: i64,
    },

    /// `env` / `$ENV`
    Env,

    /// `builtins` - list all builtins
    Builtins,

    /// `input` - read next JSON input
    ReadInput,

    /// `inputs` - read all remaining JSON inputs
    ReadInputs,

    /// `debug` - passthrough with stderr output
    Debug {
        expr: Box<Expr>,
    },

    /// `stderr` - output to stderr
    Stderr {
        expr: Box<Expr>,
    },

    /// `modulemeta` - module metadata
    ModuleMeta,

    /// `genlabel` - generate unique label (internal)
    GenLabel,

    /// Runtime builtin call with evaluated args: contains, startswith, etc.
    CallBuiltin {
        name: String,
        args: Vec<Expr>,
    },
}

/// String interpolation part.
#[derive(Debug, Clone)]
pub enum StringPart {
    Literal(String),
    Expr(Expr),
}

/// Literal values.
#[derive(Debug, Clone)]
pub enum Literal {
    Null,
    True,
    False,
    Num(f64, Option<std::rc::Rc<str>>),
    Str(String),
}

/// Binary operators.
#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

impl BinOp {
    /// Invert a comparison operator (for `cmp | not` → inverted cmp).
    /// Returns None for non-comparison operators.
    pub fn invert_cmp(self) -> Option<BinOp> {
        match self {
            BinOp::Eq => Some(BinOp::Ne),
            BinOp::Ne => Some(BinOp::Eq),
            BinOp::Lt => Some(BinOp::Ge),
            BinOp::Ge => Some(BinOp::Lt),
            BinOp::Gt => Some(BinOp::Le),
            BinOp::Le => Some(BinOp::Gt),
            _ => None,
        }
    }
}

/// Unary operators (builtins with 1 argument).
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    // Type
    Length,
    Type,
    TypeOf,
    Infinite,
    Nan,
    IsInfinite,
    IsNan,
    IsNormal,
    IsFinite,

    // Conversion
    ToString,
    ToNumber,
    ToJson,
    FromJson,
    Ascii,
    Explode,
    Implode,

    // String
    AsciiDowncase,
    AsciiUpcase,
    Trim,
    Ltrim,
    Rtrim,
    Utf8ByteLength,

    // Math
    Floor,
    Ceil,
    Round,
    Fabs,
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

    // Array/Object
    Keys,
    KeysUnsorted,
    Values,
    Sort,
    Reverse,
    Unique,
    Flatten,
    Min,
    Max,
    Add,
    Any,
    All,
    Transpose,
    ToEntries,
    FromEntries,

    // Date/Time
    Gmtime,
    Mktime,
    Now,

    // Special
    Abs,
    Not,
    GetModuleMeta,
}

/// Closure operations (operations that take a key expression).
#[derive(Debug, Clone, Copy)]
pub enum ClosureOpKind {
    SortBy,
    GroupBy,
    UniqueBy,
    MinBy,
    MaxBy,
}

/// A compiled function (subfunction from jq bytecode).
#[derive(Debug, Clone)]
pub struct CompiledFunc {
    pub name: Option<String>,
    pub nargs: usize,
    pub body: Expr,
    pub param_vars: Vec<u16>,
}

// ============================================================================
// Pipe beta-reduction helpers
// ============================================================================

impl Expr {
    /// Check if this expression is a simple scalar — always produces exactly one output,
    /// cheap and side-effect-free, safe to substitute (even if duplicated).
    pub fn is_simple_scalar(&self) -> bool {
        match self {
            Expr::Input | Expr::Literal(_) | Expr::LoadVar { .. } => true,
            Expr::Index { expr, key } | Expr::IndexOpt { expr, key } => {
                expr.is_simple_scalar() && matches!(key.as_ref(), Expr::Literal(_))
            }
            Expr::BinOp { lhs, rhs, .. } => lhs.is_simple_scalar() && rhs.is_simple_scalar(),
            Expr::UnaryOp { operand, .. } => operand.is_simple_scalar(),
            Expr::Negate { operand } => operand.is_simple_scalar(),
            Expr::Alternative { primary, fallback } => primary.is_simple_scalar() && fallback.is_simple_scalar(),
            Expr::IfThenElse { cond, then_branch, else_branch } =>
                cond.is_simple_scalar() && then_branch.is_simple_scalar() && else_branch.is_simple_scalar(),
            _ => false,
        }
    }

    /// Check if all Input references in this expression are "free" — not bound by
    /// Pipe, Reduce, Foreach, LetBinding, TryCatch, or other binding constructs.
    /// When true, it is safe to substitute a replacement for all Input nodes.
    pub fn is_input_free(&self) -> bool {
        match self {
            Expr::Input | Expr::Literal(_) | Expr::LoadVar { .. }
            | Expr::Empty | Expr::Env | Expr::Builtins
            | Expr::ReadInput | Expr::ReadInputs | Expr::ModuleMeta
            | Expr::GenLabel | Expr::Loc { .. } => true,
            Expr::BinOp { lhs, rhs, .. } => lhs.is_input_free() && rhs.is_input_free(),
            Expr::UnaryOp { operand, .. } => operand.is_input_free(),
            Expr::Index { expr, key } => expr.is_input_free() && key.is_input_free(),
            Expr::IndexOpt { expr, key } => expr.is_input_free() && key.is_input_free(),
            Expr::Negate { operand } => operand.is_input_free(),
            Expr::Collect { generator } => generator.is_input_free(),
            Expr::Comma { left, right } => left.is_input_free() && right.is_input_free(),
            Expr::Each { input_expr } | Expr::EachOpt { input_expr } => input_expr.is_input_free(),
            Expr::IfThenElse { cond, then_branch, else_branch } => {
                cond.is_input_free() && then_branch.is_input_free() && else_branch.is_input_free()
            }
            Expr::ObjectConstruct { pairs } => {
                pairs.iter().all(|(k, v)| k.is_input_free() && v.is_input_free())
            }
            Expr::Alternative { primary, fallback } => {
                primary.is_input_free() && fallback.is_input_free()
            }
            Expr::Format { expr, .. } => expr.is_input_free(),
            // Debug/Stderr use implicit input (pass through current input)
            Expr::Debug { .. } | Expr::Stderr { .. } => false,
            Expr::Slice { expr, from, to } => {
                expr.is_input_free()
                    && from.as_ref().map_or(true, |e| e.is_input_free())
                    && to.as_ref().map_or(true, |e| e.is_input_free())
            }
            Expr::StringInterpolation { parts } => {
                parts.iter().all(|p| match p {
                    StringPart::Literal(_) => true,
                    StringPart::Expr(e) => e.is_input_free(),
                })
            }
            // CallBuiltin uses the current input implicitly (passed as first arg by JIT)
            Expr::CallBuiltin { .. } => false,
            Expr::Error { msg } => msg.as_ref().map_or(true, |e| e.is_input_free()),
            // Not uses implicit input (negates truthiness of current input)
            Expr::Not => false,
            // Binding constructs — Input in sub-exprs may refer to different values
            Expr::Pipe { .. } | Expr::Reduce { .. } | Expr::Foreach { .. }
            | Expr::LetBinding { .. } | Expr::TryCatch { .. } | Expr::While { .. }
            | Expr::Until { .. } | Expr::Repeat { .. } | Expr::Label { .. }
            | Expr::Update { .. } | Expr::Assign { .. } | Expr::AllShort { .. }
            | Expr::AnyShort { .. } | Expr::Limit { .. } => false,
            // Regex/closure ops — conservatively refuse
            Expr::RegexTest { .. } | Expr::RegexMatch { .. } | Expr::RegexCapture { .. }
            | Expr::RegexScan { .. } | Expr::RegexSub { .. } | Expr::RegexGsub { .. }
            | Expr::ClosureOp { .. } | Expr::AlternativeDestructure { .. } => false,
            // Complex constructs
            Expr::Recurse { .. } | Expr::Range { .. } | Expr::PathExpr { .. }
            | Expr::SetPath { .. } | Expr::GetPath { .. } | Expr::DelPaths { .. }
            | Expr::Break { .. } | Expr::FuncCall { .. } => false,
        }
    }

    /// Conservative check: returns true if this expression is guaranteed to produce
    /// exactly one output value (not a generator). Used to guard beta-reductions
    /// that would be unsound with generators (e.g., range, each, comma).
    pub fn is_single_output(&self) -> bool {
        match self {
            Expr::Input | Expr::Literal(_) | Expr::LoadVar { .. }
            | Expr::Env | Expr::Builtins | Expr::Not | Expr::Loc { .. } => true,
            Expr::Index { expr, key } => expr.is_single_output() && key.is_single_output(),
            Expr::UnaryOp { operand, .. } => operand.is_single_output(),
            Expr::BinOp { lhs, rhs, .. } => lhs.is_single_output() && rhs.is_single_output(),
            Expr::Negate { operand } => operand.is_single_output(),
            Expr::Format { expr, .. } => expr.is_single_output(),
            Expr::Slice { expr, from, to } => {
                expr.is_single_output()
                    && from.as_ref().map_or(true, |e| e.is_single_output())
                    && to.as_ref().map_or(true, |e| e.is_single_output())
            }
            // Conservative: everything else might be a generator
            _ => false,
        }
    }

    /// Substitute all Input nodes with `replacement`.
    /// Only valid when `is_input_free()` is true.
    pub fn substitute_input(&self, replacement: &Expr) -> Expr {
        match self {
            Expr::Input => replacement.clone(),
            Expr::Literal(_) | Expr::LoadVar { .. } | Expr::Empty
            | Expr::Not | Expr::Env | Expr::Builtins | Expr::ReadInput
            | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
            | Expr::Loc { .. } => self.clone(),
            Expr::BinOp { op, lhs, rhs } => Expr::BinOp {
                op: *op,
                lhs: Box::new(lhs.substitute_input(replacement)),
                rhs: Box::new(rhs.substitute_input(replacement)),
            },
            Expr::UnaryOp { op, operand } => Expr::UnaryOp {
                op: *op,
                operand: Box::new(operand.substitute_input(replacement)),
            },
            Expr::Index { expr, key } => Expr::Index {
                expr: Box::new(expr.substitute_input(replacement)),
                key: Box::new(key.substitute_input(replacement)),
            },
            Expr::IndexOpt { expr, key } => Expr::IndexOpt {
                expr: Box::new(expr.substitute_input(replacement)),
                key: Box::new(key.substitute_input(replacement)),
            },
            Expr::Negate { operand } => Expr::Negate {
                operand: Box::new(operand.substitute_input(replacement)),
            },
            Expr::Collect { generator } => Expr::Collect {
                generator: Box::new(generator.substitute_input(replacement)),
            },
            Expr::Comma { left, right } => Expr::Comma {
                left: Box::new(left.substitute_input(replacement)),
                right: Box::new(right.substitute_input(replacement)),
            },
            Expr::Each { input_expr } => Expr::Each {
                input_expr: Box::new(input_expr.substitute_input(replacement)),
            },
            Expr::EachOpt { input_expr } => Expr::EachOpt {
                input_expr: Box::new(input_expr.substitute_input(replacement)),
            },
            Expr::IfThenElse { cond, then_branch, else_branch } => Expr::IfThenElse {
                cond: Box::new(cond.substitute_input(replacement)),
                then_branch: Box::new(then_branch.substitute_input(replacement)),
                else_branch: Box::new(else_branch.substitute_input(replacement)),
            },
            Expr::ObjectConstruct { pairs } => Expr::ObjectConstruct {
                pairs: pairs.iter().map(|(k, v)| {
                    (k.substitute_input(replacement), v.substitute_input(replacement))
                }).collect(),
            },
            Expr::Alternative { primary, fallback } => Expr::Alternative {
                primary: Box::new(primary.substitute_input(replacement)),
                fallback: Box::new(fallback.substitute_input(replacement)),
            },
            Expr::Format { name, expr } => Expr::Format {
                name: name.clone(),
                expr: Box::new(expr.substitute_input(replacement)),
            },
            Expr::Debug { expr } => Expr::Debug {
                expr: Box::new(expr.substitute_input(replacement)),
            },
            Expr::Stderr { expr } => Expr::Stderr {
                expr: Box::new(expr.substitute_input(replacement)),
            },
            Expr::Slice { expr, from, to } => Expr::Slice {
                expr: Box::new(expr.substitute_input(replacement)),
                from: from.as_ref().map(|e| Box::new(e.substitute_input(replacement))),
                to: to.as_ref().map(|e| Box::new(e.substitute_input(replacement))),
            },
            Expr::StringInterpolation { parts } => Expr::StringInterpolation {
                parts: parts.iter().map(|p| match p {
                    StringPart::Literal(s) => StringPart::Literal(s.clone()),
                    StringPart::Expr(e) => StringPart::Expr(e.substitute_input(replacement)),
                }).collect(),
            },
            Expr::CallBuiltin { name, args } => Expr::CallBuiltin {
                name: name.clone(),
                args: args.iter().map(|a| a.substitute_input(replacement)).collect(),
            },
            Expr::Error { msg } => Expr::Error {
                msg: msg.as_ref().map(|e| Box::new(e.substitute_input(replacement))),
            },
            // Safety fallback — should not reach here if is_input_free() was true
            _ => self.clone(),
        }
    }

    /// Substitute all LoadVar references with the given var_index with the replacement expression.
    pub fn substitute_var(&self, var_index: u16, replacement: &Expr) -> Expr {
        match self {
            Expr::LoadVar { var_index: idx } if *idx == var_index => replacement.clone(),
            Expr::LoadVar { .. } | Expr::Literal(_) | Expr::Input | Expr::Empty
            | Expr::Not | Expr::Env | Expr::Builtins | Expr::ReadInput
            | Expr::ReadInputs | Expr::ModuleMeta | Expr::GenLabel
            | Expr::Loc { .. } => self.clone(),
            Expr::BinOp { op, lhs, rhs } => Expr::BinOp {
                op: *op,
                lhs: Box::new(lhs.substitute_var(var_index, replacement)),
                rhs: Box::new(rhs.substitute_var(var_index, replacement)),
            },
            Expr::UnaryOp { op, operand } => Expr::UnaryOp {
                op: *op,
                operand: Box::new(operand.substitute_var(var_index, replacement)),
            },
            Expr::Index { expr, key } => Expr::Index {
                expr: Box::new(expr.substitute_var(var_index, replacement)),
                key: Box::new(key.substitute_var(var_index, replacement)),
            },
            Expr::IndexOpt { expr, key } => Expr::IndexOpt {
                expr: Box::new(expr.substitute_var(var_index, replacement)),
                key: Box::new(key.substitute_var(var_index, replacement)),
            },
            Expr::Negate { operand } => Expr::Negate {
                operand: Box::new(operand.substitute_var(var_index, replacement)),
            },
            Expr::Collect { generator } => Expr::Collect {
                generator: Box::new(generator.substitute_var(var_index, replacement)),
            },
            Expr::Comma { left, right } => Expr::Comma {
                left: Box::new(left.substitute_var(var_index, replacement)),
                right: Box::new(right.substitute_var(var_index, replacement)),
            },
            Expr::Pipe { left, right } => Expr::Pipe {
                left: Box::new(left.substitute_var(var_index, replacement)),
                right: Box::new(right.substitute_var(var_index, replacement)),
            },
            Expr::IfThenElse { cond, then_branch, else_branch } => Expr::IfThenElse {
                cond: Box::new(cond.substitute_var(var_index, replacement)),
                then_branch: Box::new(then_branch.substitute_var(var_index, replacement)),
                else_branch: Box::new(else_branch.substitute_var(var_index, replacement)),
            },
            Expr::ObjectConstruct { pairs } => Expr::ObjectConstruct {
                pairs: pairs.iter().map(|(k, v)| {
                    (k.substitute_var(var_index, replacement), v.substitute_var(var_index, replacement))
                }).collect(),
            },
            Expr::Alternative { primary, fallback } => Expr::Alternative {
                primary: Box::new(primary.substitute_var(var_index, replacement)),
                fallback: Box::new(fallback.substitute_var(var_index, replacement)),
            },
            Expr::Each { input_expr } => Expr::Each {
                input_expr: Box::new(input_expr.substitute_var(var_index, replacement)),
            },
            Expr::EachOpt { input_expr } => Expr::EachOpt {
                input_expr: Box::new(input_expr.substitute_var(var_index, replacement)),
            },
            // Don't substitute into LetBinding that rebinds the same variable
            Expr::LetBinding { var_index: vi, value, body } => {
                let new_value = Box::new(value.substitute_var(var_index, replacement));
                if *vi == var_index {
                    // Inner binding shadows; don't substitute in body
                    Expr::LetBinding { var_index: *vi, value: new_value, body: body.clone() }
                } else {
                    Expr::LetBinding { var_index: *vi, value: new_value, body: Box::new(body.substitute_var(var_index, replacement)) }
                }
            },
            Expr::CallBuiltin { name, args } => Expr::CallBuiltin {
                name: name.clone(),
                args: args.iter().map(|a| a.substitute_var(var_index, replacement)).collect(),
            },
            Expr::Update { path_expr, update_expr } => Expr::Update {
                path_expr: Box::new(path_expr.substitute_var(var_index, replacement)),
                update_expr: Box::new(update_expr.substitute_var(var_index, replacement)),
            },
            Expr::Assign { path_expr, value_expr } => Expr::Assign {
                path_expr: Box::new(path_expr.substitute_var(var_index, replacement)),
                value_expr: Box::new(value_expr.substitute_var(var_index, replacement)),
            },
            Expr::TryCatch { try_expr, catch_expr } => Expr::TryCatch {
                try_expr: Box::new(try_expr.substitute_var(var_index, replacement)),
                catch_expr: Box::new(catch_expr.substitute_var(var_index, replacement)),
            },
            Expr::StringInterpolation { parts } => Expr::StringInterpolation {
                parts: parts.iter().map(|p| match p {
                    StringPart::Literal(s) => StringPart::Literal(s.clone()),
                    StringPart::Expr(e) => StringPart::Expr(e.substitute_var(var_index, replacement)),
                }).collect(),
            },
            Expr::Slice { expr, from, to } => Expr::Slice {
                expr: Box::new(expr.substitute_var(var_index, replacement)),
                from: from.as_ref().map(|e| Box::new(e.substitute_var(var_index, replacement))),
                to: to.as_ref().map(|e| Box::new(e.substitute_var(var_index, replacement))),
            },
            Expr::Reduce { source, init, var_index: vi, acc_index, update } => {
                let new_source = Box::new(source.substitute_var(var_index, replacement));
                let new_init = Box::new(init.substitute_var(var_index, replacement));
                if *vi == var_index || *acc_index == var_index {
                    Expr::Reduce { source: new_source, init: new_init, var_index: *vi, acc_index: *acc_index, update: update.clone() }
                } else {
                    Expr::Reduce { source: new_source, init: new_init, var_index: *vi, acc_index: *acc_index, update: Box::new(update.substitute_var(var_index, replacement)) }
                }
            },
            Expr::Foreach { source, init, var_index: vi, acc_index, update, extract } => {
                let new_source = Box::new(source.substitute_var(var_index, replacement));
                let new_init = Box::new(init.substitute_var(var_index, replacement));
                if *vi == var_index || *acc_index == var_index {
                    Expr::Foreach { source: new_source, init: new_init, var_index: *vi, acc_index: *acc_index, update: update.clone(), extract: extract.clone() }
                } else {
                    Expr::Foreach {
                        source: new_source, init: new_init, var_index: *vi, acc_index: *acc_index,
                        update: Box::new(update.substitute_var(var_index, replacement)),
                        extract: extract.as_ref().map(|e| Box::new(e.substitute_var(var_index, replacement))),
                    }
                }
            },
            Expr::Error { msg } => Expr::Error {
                msg: msg.as_ref().map(|e| Box::new(e.substitute_var(var_index, replacement))),
            },
            _ => self.clone(),
        }
    }
}
