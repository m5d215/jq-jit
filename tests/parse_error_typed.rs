//! #1037: lex/parse errors are typed `ParseError { loc, kind }` values
//! reachable by downcast through the `anyhow` chain. Pin the downcast
//! surface (kind + source location) and the legacy message wording the
//! `Display` impl must preserve — the CLI prints `jq: error: {Display}`
//! on exit 3, so any wording drift here is user-visible.

use jq_jit::parse_error::{ParseError, ParseErrorKind};
use jq_jit::parser::Parser;

fn parse_err(program: &str) -> ParseError {
    let e = match Parser::parse(program) {
        Ok(_) => panic!("program must fail to parse: {:?}", program),
        Err(e) => e,
    };
    e.downcast_ref::<ParseError>()
        .unwrap_or_else(|| panic!("not a typed ParseError: {:#}", e))
        .clone()
}

#[test]
fn lexer_error_carries_char_offset_and_line() {
    let err = parse_err("1 ! 2");
    assert!(
        matches!(err.kind, ParseErrorKind::UnexpectedChar { ch: '!', pos: 2 }),
        "kind: {:?}",
        err.kind
    );
    let loc = err.loc.expect("lexer errors carry a location");
    assert_eq!((loc.line, loc.offset), (1, 2));
    assert_eq!(err.to_string(), "unexpected character '!' at position 2");
}

#[test]
fn parser_error_locates_the_offending_token() {
    // `]` is the 5th char (0-based offset 5) on source line 2.
    let err = parse_err(".a |\n]");
    assert!(
        matches!(err.kind, ParseErrorKind::UnexpectedToken { .. }),
        "kind: {:?}",
        err.kind
    );
    let loc = err.loc.expect("parser errors carry the token location");
    assert_eq!((loc.line, loc.offset), (2, 5));
    assert_eq!(err.to_string(), "unexpected token RBracket");
}

#[test]
fn deferred_reachability_errors_have_no_location() {
    // #765 / #807 defer unbound-name errors to a whole-program
    // reachability pass, so no single source position applies.
    let err = parse_err("foo");
    match &err.kind {
        ParseErrorKind::UndefinedFunction { name, nargs } => {
            assert_eq!((name.as_str(), *nargs), ("foo", 0));
        }
        k => panic!("kind: {:?}", k),
    }
    assert!(err.loc.is_none());
    assert_eq!(err.to_string(), "foo/0 is not defined");

    let err = parse_err("$undef_var");
    assert!(matches!(err.kind, ParseErrorKind::UndefinedVariable { .. }));
    assert!(err.loc.is_none());
    assert_eq!(err.to_string(), "$undef_var is not defined");
}

#[test]
fn bison_style_syntax_errors_keep_jq_wording() {
    let err = parse_err(".[:]");
    assert!(
        matches!(
            err.kind,
            ParseErrorKind::SyntaxUnexpected { what: "']'", expecting: None }
        ),
        "kind: {:?}",
        err.kind
    );
    assert_eq!(err.to_string(), "syntax error, unexpected ']'");

    let err = parse_err(". as [$__loc__] | .");
    assert_eq!(
        err.to_string(),
        "syntax error, unexpected $__loc__, expecting BINDING or '[' or '{'"
    );
}

#[test]
fn expect_mismatch_reports_both_tokens() {
    // `{"a" 1}`: the `"a"` entry parses as shorthand, so the object parser
    // next expects `,` or `}` and `expect(RBrace)` trips on the number.
    let err = parse_err(r#"{"a" 1}"#);
    match &err.kind {
        ParseErrorKind::ExpectedToken { expected, got } => {
            assert_eq!(expected, "RBrace");
            assert_eq!(got, "Num(1.0, None)");
        }
        k => panic!("kind: {:?}", k),
    }
    assert_eq!(err.to_string(), "expected RBrace, got Num(1.0, None)");
}

#[test]
fn string_escape_errors_are_typed() {
    let err = parse_err(r#""ab\ud834 ""#);
    assert!(matches!(err.kind, ParseErrorKind::InvalidSurrogatePair));
    assert_eq!(
        err.to_string(),
        "Invalid \\uXXXX\\uXXXX surrogate pair escape"
    );

    let err = parse_err(r#""abc"#);
    assert!(matches!(err.kind, ParseErrorKind::UnterminatedString));
    assert!(err.loc.is_some());
}
