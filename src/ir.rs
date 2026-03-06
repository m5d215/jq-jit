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
}
