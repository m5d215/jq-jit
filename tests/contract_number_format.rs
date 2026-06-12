//! #1028: number-to-text formatting has a single core (`write_jq_number` /
//! `num_repr_text` in src/value.rs) and every public entry point is a thin
//! adapter over it. These contracts pin (a) byte-identical output across all
//! entry points for the same number, and (b) the exact rendering rules the
//! historical per-sink copies implemented, so a future "fix one formatter"
//! patch that reintroduces drift (the #616 class) fails here.

use std::rc::Rc;

use jq_jit::value::{
    format_jq_number, push_jq_number_bytes, push_jq_number_str, push_num_tojson_str,
    push_value_num_repr_bytes, push_value_num_repr_str, write_value_compact_ext, Value,
};

/// All bare-f64 entry points, rendered and asserted identical.
fn render_all(n: f64) -> String {
    let from_format = format_jq_number(n);
    let mut from_str = String::new();
    push_jq_number_str(&mut from_str, n);
    let mut from_bytes = Vec::new();
    push_jq_number_bytes(&mut from_bytes, n);
    let from_bytes = String::from_utf8(from_bytes).expect("formatter output is ASCII");
    assert_eq!(from_format, from_str, "format vs push_str for {n:?}");
    assert_eq!(from_format, from_bytes, "format vs push_bytes for {n:?}");
    from_format
}

#[test]
fn bare_f64_corpus_pins() {
    // (input, expected) — expected strings are jq 1.8.1's rendering of a
    // computed (repr-less) f64, as established by #110/#143/#426/#721.
    let cases: &[(f64, &str)] = &[
        // zeros and integers (itoa fast path; i64-boundary neighborhood)
        (0.0, "0"),
        (-0.0, "-0"), // #110
        (1.0, "1"),
        (-1.0, "-1"),
        (42.0, "42"),
        (1e15, "1000000000000000"),
        (9999999999999998.0, "9999999999999998"), // largest exacts below 1e16
        (-9999999999999998.0, "-9999999999999998"),
        (9007199254740992.0, "9007199254740992"), // 2^53
        (-9007199254740992.0, "-9007199254740992"),
        // specials
        (f64::NAN, "null"),
        (f64::INFINITY, "1.7976931348623157e+308"),
        (f64::NEG_INFINITY, "-1.7976931348623157e+308"),
        // small side: scientific lowercase with 2-digit-padded exponent (#426)
        (1e-4, "0.0001"), // boundary stays fixed-point
        (9.999e-5, "9.999e-05"),
        (1e-7, "1e-07"),
        (1.5e-10, "1.5e-10"),
        (5e-324, "5e-324"), // smallest subnormal
        (-5e-324, "-5e-324"),
        // large side: sigdigits rule (#721)
        (1e16, "1e+16"),
        (-1e16, "-1e+16"),
        (1e22, "1e+22"),
        (1.2345678e22, "12345678000000000000000"), // high precision stays fixed
        (f64::MAX, "1.7976931348623157e+308"),
        // common decimals (ryu branch)
        (1.5, "1.5"),
        (-2.25, "-2.25"),
        (0.1, "0.1"),
        (3.141592653589793, "3.141592653589793"),
        (123.456, "123.456"),
        // shortest-digit tie: 919895839836842.25 is exactly halfway between
        // the two shortest 16-digit candidates. jq's dtoa (and ryu) round
        // the tie to "…2"; Rust `Display` picks "…3", which is why the old
        // Display-based String formatter drifted from the byte formatter
        // (and from jq) — the unification pins the jq side.
        (919895839836842.25, "919895839836842.2"),
    ];
    for &(n, expected) in cases {
        assert_eq!(render_all(n), expected, "for input {n:?}");
    }
}

/// Sweep a deterministic corpus and assert every bare-f64 entry point stays
/// byte-identical. The common-decimal branch historically rendered through
/// Rust `Display` in the String formatter and through ryu in the byte
/// formatter; both emit shortest round-trip digits but break exact-halfway
/// ties differently (`Display` rounds up, ryu — like jq's dtoa — rounds to
/// even), which was a live tostring/@csv-vs-output divergence. The unified
/// core uses ryu everywhere; output must also round-trip back to the input.
#[test]
fn all_entry_points_agree_across_sweep() {
    let mut state: u64 = 0x1028_5EED_CAFE_F00D;
    let mut lcg = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    // Uniform magnitudes across the fixed-point window [1e-4, 1e16).
    for _ in 0..200_000 {
        let mantissa = (lcg() >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        let exp = (lcg() % 20) as i32 - 4; // 10^-4 ..= 10^15
        let n = (mantissa + 0.000_1) * 10f64.powi(exp);
        let out = render_all(n);
        assert_eq!(out.parse::<f64>().ok(), Some(n), "round-trip for {n:?}");
    }
    // Random bit patterns: every entry point must still agree (NaN payloads,
    // subnormals, huge exponents).
    for _ in 0..200_000 {
        render_all(f64::from_bits(lcg()));
    }
}

/// Repr-preserving trio: @csv/@tsv writers (`push_value_num_repr_*`) and the
/// tojson/tostring writer (`push_num_tojson_str`) share one repr-selection
/// core. They must agree byte-for-byte except for NaN (empty CSV cell vs
/// JSON `null`, #771).
fn render_repr_all(n: f64, repr: Option<&str>) -> (String, String) {
    let repr: Option<Rc<str>> = repr.map(Rc::from);
    let repr = repr.as_ref();
    let mut csv_str = String::new();
    push_value_num_repr_str(&mut csv_str, n, repr);
    let mut csv_bytes = Vec::new();
    push_value_num_repr_bytes(&mut csv_bytes, n, repr);
    let csv_bytes = String::from_utf8(csv_bytes).expect("formatter output is ASCII");
    assert_eq!(csv_str, csv_bytes, "repr str vs bytes for {n:?} {repr:?}");
    let mut tojson = String::new();
    push_num_tojson_str(&mut tojson, n, repr);
    if !n.is_nan() {
        assert_eq!(csv_str, tojson, "csv vs tojson for {n:?} {repr:?}");
    }
    (csv_str, tojson)
}

#[test]
fn repr_preserving_corpus_pins() {
    // (n, repr, expected) — jq's literal-preserving canonical form (#475,
    // #457, #616): exact reprs keep their shape in uppercase-E decnum style;
    // non-roundtripping reprs fall back to the computed-f64 form.
    let cases: &[(f64, Option<&str>, &str)] = &[
        (1.0, Some("1.0"), "1.0"),
        (1.0, Some("1.00"), "1.00"),
        (0.0, Some("0e10"), "0E+10"),
        (1e10, Some("1e10"), "1E+10"),
        (1e10, Some("1.0e10"), "1.0E+10"),
        (1e-5, Some("1e-5"), "0.00001"),
        (5e-324, Some("5e-324"), "5E-324"), // #616: smallest subnormal
        (f64::MAX, Some("1.7976931348623157e308"), "1.7976931348623157E+308"),
        // 16 significant digits: not exactly representable, repr dropped.
        (13911860366432393u64 as f64, Some("13911860366432393"), "13911860366432392"),
        (1.5, None, "1.5"),
        (-0.0, None, "-0"),
    ];
    for &(n, repr, expected) in cases {
        let (csv, _) = render_repr_all(n, repr);
        assert_eq!(csv, expected, "for input {n:?} repr {repr:?}");
    }
    // NaN: empty CSV cell (#771) vs JSON null.
    let (csv, tojson) = render_repr_all(f64::NAN, None);
    assert_eq!(csv, "");
    assert_eq!(tojson, "null");
    let (csv, tojson) = render_repr_all(f64::NAN, Some("nan"));
    assert_eq!(csv, "");
    assert_eq!(tojson, "null");
}

/// #1077: `num_repr_text` short-circuits plain short decimal lexemes with a
/// single scan instead of the three full checks (validity / exactness /
/// normalization). Pin the boundary shapes on both sides of the gate so the
/// fast path can never drift from the semantics it replaces — every
/// expected value below was cross-checked against jq 1.8.1 (the 16-digit
/// integer is the known no-decnum fallback, #236/#415).
#[test]
fn short_canonical_plain_fast_path_boundary_pins() {
    let cases: &[(&str, &str)] = &[
        // Inside the fast path: emitted verbatim.
        ("123", "123"),
        ("-7", "-7"),
        ("1.5", "1.5"),
        ("1.50", "1.50"),
        ("0.5", "0.5"),
        ("-0", "-0"),
        ("999999999999999", "999999999999999"), // 15 digits, max accepted
        ("0.000001", "0.000001"),               // 5-zero fraction stays plain
        ("123.4567890123", "123.4567890123"),
        // Rejected by the fast path (conservative): the full checks decide.
        ("1.0000005", "1.0000005"), // 6-zero run but te in canonical window
        ("0.000000", "0.000000"),   // pure-zero 6-digit fraction stays plain
        ("0.0000001", "1E-7"),      // #611 normalization
        ("-0.0000001", "-1E-7"),
        ("1e3", "1E+3"),            // exponent → canonical uppercase-E form
        ("9999999999999999", "1e+16"), // 16 digits → repr dropped, f64 form
    ];
    for &(lexeme, expected) in cases {
        let n: f64 = lexeme.parse().expect("test lexeme parses");
        let repr: Rc<str> = Rc::from(lexeme);
        let mut out = String::new();
        push_num_tojson_str(&mut out, n, Some(&repr));
        assert_eq!(out, expected, "for lexeme {lexeme:?}");
    }
}

/// The `--sort-keys` writer used to carry its own number formatter that
/// normalised `-0` to `0`, splitting `-S` output from every other path
/// (jq 1.8.1 prints `-0`). Unified in #1028 — pin the fix.
#[test]
fn sort_keys_writer_preserves_negative_zero() {
    let mut out = Vec::new();
    write_value_compact_ext(&mut out, &Value::number(-0.0), true).unwrap();
    assert_eq!(out, b"-0");
    let mut out = Vec::new();
    write_value_compact_ext(
        &mut out,
        &Value::object_from_pairs(vec![("a", Value::number(-0.0))]),
        true,
    )
    .unwrap();
    assert_eq!(out, br#"{"a":-0}"#);
}
