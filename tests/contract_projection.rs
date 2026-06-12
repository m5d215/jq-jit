//! #1055: IR-driven projection pushdown. `Filter::needed_input_fields`
//! derives the static set of top-level fields a filter reads; the host then
//! parses only those fields per record (`json_stream_project`), skipping
//! Value materialization for the rest. These contracts pin the two
//! load-bearing properties:
//!
//!  1. extraction soundness — a filter executed against the projected
//!     record produces byte-for-byte the same outputs (and the same
//!     errors) as against the fully parsed record;
//!  2. extraction coverage — shapes the walk is supposed to handle keep
//!     projecting, and shapes that observe the whole record (identity,
//!     iteration over `.`, `keys`, `input`, bare `error`, ...) keep
//!     refusing, since projection would change their observable output.

use jq_jit::interpreter::Filter;
use jq_jit::value::{json_stream, json_stream_project, Value};

/// Filters that must project, with the exact expected field set.
const PROJECTING: &[(&str, &[&str])] = &[
    (".x", &["x"]),
    ("{a: {b: .x}}", &["x"]),
    (".x, .y", &["x", "y"]),
    ("[.x, .y, .name]", &["name", "x", "y"]),
    ("[.x, .y] | min", &["x", "y"]),
    ("select(.x > 1) | {a: .x}", &["x"]),
    ("select(.x > 1) | select(.y < 9) | [.x, .y]", &["x", "y"]),
    (".x | tostring", &["x"]),
    (".x | keys", &["x"]),
    (".x | ..", &["x"]),
    (".x[]", &["x"]),
    (".x[] | .k", &["x"]),
    (".a.b.c", &["a"]),
    (".tags[0]", &["tags"]),
    (".x[1:3]", &["x"]),
    ("if .a then .b else .c end", &["a", "b", "c"]),
    (".x // 0", &["x"]),
    (".x?", &["x"]),
    ("\"\\(.x)-\\(.y)\"", &["x", "y"]),
    (".x | @base64", &["x"]),
    ("limit(2; .x[])", &["x"]),
    ("first(.x[])", &["x"]),
    ("reduce .x[] as $v (0; . + $v)", &["x"]),
    ("foreach .x[] as $v (0; . + $v)", &["x"]),
    ("[.x[] | select(. > 2)]", &["x"]),
    (".x as $a | .y + $a", &["x", "y"]),
    ("{(.k): .v}", &["k", "v"]),
    ("try .x catch .", &["x"]),
    ("error(.msg) // 1", &["msg"]),
    (".x | match(\"a\")", &["x"]),
    (".name | test(\"z\")", &["name"]),
    (".x | tonumber + 1", &["x"]),
    ("all(.x[]; . > 0)", &["x"]),
    (".x | while(. < 30; . * 2)", &["x"]),
];

/// Filters that must NOT project: their output (or stream consumption)
/// observes parts of the record outside any static field set.
const NON_PROJECTING: &[&str] = &[
    ".",
    ".[]",
    "keys",
    "to_entries",
    "has(\"x\")",
    "select(.x > 1)",       // outputs the record itself
    "..",                   // yields the record first
    "error",                // throws the record as the error payload
    "input",
    ".x | input",            // pulls a (projected) record from the stream
    ".x | input_line_number", // parser bookkeeping, not derived data
    "while(.x < 3; .x + 1)", // iterates record-derived snapshots of `.`
    ". as $r | .x | $r",     // smuggles the record out through a variable
    "recurse(.children[]?)", // recurse(f) yields the record before f's outputs
    "recurse(.children[]?; . != null) | .name",
    "[.x, debug(\"m\")]",    // debug prints its arg but passes the record through
    "[.x, stderr]",
];

/// Inputs exercising the projecting parser's edges: extra fields before /
/// after / between the needed ones, missing fields (null backfill),
/// duplicate keys (last-wins), escaped keys, nested composites in both
/// kept and skipped fields, empty objects, and non-object records (which
/// bypass projection entirely and must still flow through).
const INPUTS: &[&str] = &[
    r#"{"x":1,"y":2,"name":"n","k":"kk","v":7,"a":{"b":{"c":3}},"tags":[9,8],"msg":"m"}"#,
    r#"{"skip0":[1,{"q":2}],"x":[3,4,5],"skip1":"zz","y":[6],"name":"item","a":{"b":{"c":null}},"tags":[],"k":"K","v":[1],"msg":"e","skipN":{"deep":{"er":1}}}"#,
    r#"{}"#,
    r#"{"unrelated":1,"other":"s"}"#,
    r#"{"x":2,"x":3,"y":1,"y":{"z":9}}"#,
    r#"{"x":5,"y":6,"name":"esc"}"#,
    r#"{"x":"a\/bA\t","y":1e3,"name":"s"}"#,
    "[1,2,3]",
    "\"scalar\"",
    "null",
    "42",
    r#"{"x":{"big":[1,2,{"n":3}]},"y":true,"name":null}"#,
];

fn run_filter(f: &mut Filter, inputs: &[Value]) -> Vec<Result<Value, String>> {
    let mut out = Vec::new();
    for inp in inputs {
        let mut vals = Vec::new();
        match f.execute_cb(inp, &mut |v| {
            vals.push(v.clone());
            Ok(true)
        }) {
            Ok(_) => out.extend(vals.into_iter().map(Ok)),
            Err(e) => {
                out.extend(vals.into_iter().map(Ok));
                out.push(Err(e.to_string()));
            }
        }
    }
    out
}

#[test]
fn projected_execution_matches_full_parse() {
    // One multi-record stream per input string plus one combined stream,
    // so the streaming entry point sees record boundaries too.
    let combined = INPUTS.join("\n");
    let mut streams: Vec<&str> = INPUTS.to_vec();
    streams.push(&combined);

    for &(filter, expected_fields) in PROJECTING {
        let mut f = Filter::with_options(filter, &[], false).expect(filter);
        let fields = f
            .needed_input_fields()
            .unwrap_or_else(|| panic!("{filter:?} must project"));
        assert_eq!(
            fields,
            expected_fields.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "{filter:?} projected field set"
        );
        let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();

        for stream in &streams {
            let mut full = Vec::new();
            json_stream(stream, |v| {
                full.push(v);
                Ok(())
            })
            .expect("full parse");
            let mut projected = Vec::new();
            json_stream_project(stream, &field_refs, |v| {
                projected.push(v);
                Ok(())
            })
            .expect("projected parse");

            let out_full = run_filter(&mut f, &full);
            let out_proj = run_filter(&mut f, &projected);
            assert_eq!(
                out_full, out_proj,
                "{filter:?} diverges between full and projected parse on {stream:?}"
            );
        }
    }
}

#[test]
fn record_observers_refuse_projection() {
    for filter in NON_PROJECTING {
        let f = Filter::with_options(filter, &[], false).expect(filter);
        assert_eq!(
            f.needed_input_fields(),
            None,
            "{filter:?} must not project — it observes the whole record"
        );
    }
}

/// The projecting parser tracks found keys in a u64 bitset; the extraction
/// must refuse field sets that would overflow it.
#[test]
fn field_set_capped_at_bitset_width() {
    let wide: Vec<String> = (0..65).map(|i| format!(".f{i:02}")).collect();
    let filter65 = format!("[{}]", wide.join(", "));
    let f = Filter::with_options(&filter65, &[], false).expect("parse");
    assert_eq!(f.needed_input_fields(), None, "65 fields must refuse");

    let filter64 = format!("[{}]", wide[..64].join(", "));
    let f = Filter::with_options(&filter64, &[], false).expect("parse");
    assert_eq!(
        f.needed_input_fields().map(|v| v.len()),
        Some(64),
        "64 fields must still project"
    );
}
