//! Structured parse errors (#1037). Every error raised while lexing or
//! parsing a program carries a typed [`ParseErrorKind`] plus, where the
//! source position is known, a [`SourceLoc`] — so callers can inspect
//! errors programmatically instead of scraping message strings.
//!
//! The [`std::fmt::Display`] impl reproduces the historical ad-hoc message
//! wording byte-for-byte (the CLI prints `jq: error: {Display}` on exit
//! code 3, and differential tests observe some of these messages), so
//! converting a `bail!` site to a typed kind never changes user-visible
//! output. Token names render through `Token`'s `Debug` form, captured as
//! a pre-formatted string at construction.

use std::fmt;

/// Source position of a parse error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLoc {
    /// 1-based source line.
    pub line: usize,
    /// 0-based char offset into the program source. For errors raised
    /// after tokenization this is the start offset of the offending token.
    pub offset: usize,
}

/// A lex/parse-time error with a typed kind and optional source location.
///
/// Location is `None` for errors with no single source position: the
/// deferred unbound-variable / unknown-function reachability errors
/// (#765 / #807, raised after the whole program is parsed), module
/// resolution failures, and the AST-shape checks that run on already
/// parsed expressions (`mutate(...)` body form, unary-op name lookup).
#[derive(Debug, Clone)]
pub struct ParseError {
    pub loc: Option<SourceLoc>,
    pub kind: ParseErrorKind,
}

impl ParseError {
    pub fn new(loc: Option<SourceLoc>, kind: ParseErrorKind) -> Self {
        ParseError { loc, kind }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Message wording is the kind's alone — location is structured
        // data for programmatic use. The historical messages that embed a
        // position keep it inside the kind (`UnexpectedChar::pos`,
        // `UnexpectedTokenAt::pos`) so the rendered text is unchanged.
        self.kind.fmt(f)
    }
}

impl std::error::Error for ParseError {}

/// What went wrong. One variant per historical message shape; the
/// `Display` arm for each variant reproduces the legacy wording exactly.
#[derive(Debug, Clone)]
pub enum ParseErrorKind {
    /// Lexer hit a character it cannot start a token with. `pos` is the
    /// char offset embedded in the legacy message.
    UnexpectedChar { ch: char, pos: usize },
    /// `@` not followed by a format name.
    ExpectedFormatName,
    /// `$` not followed by a variable name.
    ExpectedVariableName,
    /// Number literal failed `f64` parsing.
    InvalidNumber { lexeme: String, reason: String },
    /// String ended in the middle of a `\` escape.
    UnterminatedStringEscape,
    /// `\u` escape with fewer than 4 hex digits before the string ended.
    IncompleteUnicodeEscape,
    /// `\uXXXX` escape whose digits do not parse as hex. `hex` is the
    /// rejected digit run when the lexer still had 4 chars to show.
    InvalidUnicodeEscape { hex: Option<String> },
    /// High surrogate not followed by a valid low-surrogate escape.
    InvalidSurrogatePair,
    /// String literal never closed.
    UnterminatedString,
    /// Top-level parse loop found a token it cannot start a term with.
    /// `pos` is the token index embedded in the legacy message.
    UnexpectedTokenAt { got: String, pos: usize },
    /// Postfix/operator position found an unconsumable token.
    UnexpectedToken { got: String },
    /// Unexpected token inside `\(...)` string interpolation.
    UnexpectedInterpToken { got: String },
    /// `$name` referenced but never bound (#765 defers this to a
    /// reachability check after the program is parsed).
    UndefinedVariable { name: String },
    /// `name/nargs` is neither a builtin nor a user definition (#807
    /// defers this like `UndefinedVariable`).
    UndefinedFunction { name: String, nargs: usize },
    /// `expect(tok)` mismatch. Both sides pre-formatted via `Debug`.
    ExpectedToken { expected: String, got: String },
    /// "expected {what}, got {got}" family — `what` names the production
    /// ("function name", "string after import", "object key", ...).
    Expected { what: &'static str, got: String },
    /// jq-style bison message: "syntax error, unexpected {what}\[,
    /// expecting {expecting}\]".
    SyntaxUnexpected {
        what: &'static str,
        expecting: Option<&'static str>,
    },
    /// Destructuring object pattern key is a non-string literal.
    ObjectKeyType { ty: &'static str, val: String },
    /// Object-construction key const-folded to a non-string (#726).
    ObjectKeyNonString { desc: String },
    /// `import`/`include` target not found on the search path.
    ModuleNotFound { name: String, data: bool },
    /// Module file exists but could not be read.
    ModuleLoad { path: String, err: String, data: bool },
    /// Data module (`import ... as $data`) is not valid JSON.
    DataModuleParse { path: String, err: String },
    /// `mutate(...)` body is not a top-level path-update operator (jqx).
    MutateBodyForm,
    /// Internal unary-op table lookup miss.
    UnknownUnaryOp { name: String },
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ParseErrorKind::*;
        match self {
            UnexpectedChar { ch, pos } => {
                write!(f, "unexpected character '{}' at position {}", ch, pos)
            }
            ExpectedFormatName => write!(f, "expected format name after @"),
            ExpectedVariableName => write!(f, "expected variable name after $"),
            InvalidNumber { lexeme, reason } => {
                write!(f, "invalid number '{}': {}", lexeme, reason)
            }
            UnterminatedStringEscape => write!(f, "unterminated string escape"),
            IncompleteUnicodeEscape => write!(f, "incomplete unicode escape"),
            InvalidUnicodeEscape { hex: Some(hex) } => {
                write!(f, "invalid unicode escape: \\u{}", hex)
            }
            InvalidUnicodeEscape { hex: None } => write!(f, "invalid unicode escape"),
            InvalidSurrogatePair => {
                write!(f, "Invalid \\uXXXX\\uXXXX surrogate pair escape")
            }
            UnterminatedString => write!(f, "unterminated string"),
            UnexpectedTokenAt { got, pos } => {
                write!(f, "unexpected token {} at position {}", got, pos)
            }
            UnexpectedToken { got } => write!(f, "unexpected token {}", got),
            UnexpectedInterpToken { got } => {
                write!(f, "unexpected token in string interpolation: {}", got)
            }
            UndefinedVariable { name } => write!(f, "${} is not defined", name),
            UndefinedFunction { name, nargs } => {
                write!(f, "{}/{} is not defined", name, nargs)
            }
            ExpectedToken { expected, got } => {
                write!(f, "expected {}, got {}", expected, got)
            }
            Expected { what, got } => write!(f, "expected {}, got {}", what, got),
            SyntaxUnexpected { what, expecting } => {
                write!(f, "syntax error, unexpected {}", what)?;
                if let Some(expecting) = expecting {
                    write!(f, ", expecting {}", expecting)?;
                }
                Ok(())
            }
            ObjectKeyType { ty, val } => {
                write!(f, "Cannot use {} ({}) as object key", ty, val)
            }
            ObjectKeyNonString { desc } => {
                write!(f, "Cannot use {} as object key", desc)
            }
            ModuleNotFound { name, data: true } => {
                write!(f, "Cannot find data module '{}'", name)
            }
            ModuleNotFound { name, data: false } => {
                write!(f, "Cannot find module '{}'", name)
            }
            ModuleLoad { path, err, data: true } => {
                write!(f, "Cannot load data module '{}': {}", path, err)
            }
            ModuleLoad { path, err, data: false } => {
                write!(f, "Cannot load module '{}': {}", path, err)
            }
            DataModuleParse { path, err } => {
                write!(f, "Cannot parse data module '{}': {}", path, err)
            }
            MutateBodyForm => write!(
                f,
                "mutate(...) body must be a top-level path-update operator \
                 (=, |=, +=, -=, *=, /=, %=, //=); wrap composite forms by \
                 distributing mutate inward across each leaf"
            ),
            UnknownUnaryOp { name } => write!(f, "unknown unary operation: {}", name),
        }
    }
}
