//! jq-jit: A JIT-compiled jq implementation.
//!
//! Usage: jq-jit [OPTIONS] <FILTER> [FILE...]

use std::io::{self, BufRead, Read, Write};
use std::process;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use jq_jit::value::{Value, json_to_value, json_stream, json_stream_offsets, json_stream_raw, json_stream_project, json_object_get_num, json_object_get_two_nums, json_object_get_field_raw, json_object_get_fields_raw_buf, json_object_get_nested_field_raw, parse_json_num, json_value_length, json_object_keys_to_buf_reuse, json_object_extract_keys_only, json_object_keys_unsorted_to_buf, json_object_has_key, json_object_has_all_keys, json_object_has_any_key, json_type_byte, json_object_del_field, json_object_merge_literal, json_object_sort_keys, json_object_filter_by_value_type, json_each_value_raw, json_each_value_cb, json_to_entries_raw, json_with_entries_select_value_cmp, json_object_set_field_raw, json_object_update_field_num, is_json_compact, push_json_compact_raw, push_tojson_raw, push_json_pretty_raw, push_json_pretty_raw_at, value_to_json_precise, value_to_json_pretty_ext, push_compact_line, push_pretty_line, push_jq_number_bytes, write_value_compact_ext, write_value_compact_line, write_value_pretty_line, pool_value};
use jq_jit::interpreter::Filter;

fn json_escape_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(bytes.len());
    for &b in bytes {
        match b {
            b'"' => buf.extend_from_slice(b"\\\""),
            b'\\' => buf.extend_from_slice(b"\\\\"),
            b'\n' => buf.extend_from_slice(b"\\n"),
            b'\r' => buf.extend_from_slice(b"\\r"),
            b'\t' => buf.extend_from_slice(b"\\t"),
            c if c < 0x20 => { use std::io::Write; let _ = write!(buf, "\\u{:04x}", c); }
            _ => buf.push(b),
        }
    }
    buf
}

/// Unescape a JSON string's inner bytes (between quotes).
/// Handles \\", \\\\, \\n, \\r, \\t, \\/, \\uXXXX.
fn json_unescape_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'"' => { buf.push(b'"'); i += 2; }
                b'\\' => { buf.push(b'\\'); i += 2; }
                b'/' => { buf.push(b'/'); i += 2; }
                b'n' => { buf.push(b'\n'); i += 2; }
                b'r' => { buf.push(b'\r'); i += 2; }
                b't' => { buf.push(b'\t'); i += 2; }
                b'b' => { buf.push(0x08); i += 2; }
                b'f' => { buf.push(0x0c); i += 2; }
                b'u' if i + 5 < bytes.len() => {
                    if let Ok(s) = std::str::from_utf8(&bytes[i+2..i+6]) {
                        if let Ok(cp) = u32::from_str_radix(s, 16) {
                            if let Some(c) = char::from_u32(cp) {
                                let mut tmp = [0u8; 4];
                                buf.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
                            }
                        }
                    }
                    i += 6;
                }
                _ => { buf.push(bytes[i]); i += 1; }
            }
        } else {
            buf.push(bytes[i]);
            i += 1;
        }
    }
    buf
}

/// Base64 encode bytes directly into output buffer.
fn base64_encode_to(input: &[u8], out: &mut Vec<u8>) {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut i = 0;
    while i + 3 <= input.len() {
        let b0 = input[i] as u32;
        let b1 = input[i + 1] as u32;
        let b2 = input[i + 2] as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((n >> 18) & 63) as usize]);
        out.push(TABLE[((n >> 12) & 63) as usize]);
        out.push(TABLE[((n >> 6) & 63) as usize]);
        out.push(TABLE[(n & 63) as usize]);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(TABLE[((n >> 18) & 63) as usize]);
        out.push(TABLE[((n >> 12) & 63) as usize]);
        out.push(TABLE[((n >> 6) & 63) as usize]);
        out.push(b'=');
    } else if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(TABLE[((n >> 18) & 63) as usize]);
        out.push(TABLE[((n >> 12) & 63) as usize]);
        out.push(b'=');
        out.push(b'=');
    }
}

/// URI-encode bytes directly into output buffer (RFC 3986).
fn uri_encode_to(input: &[u8], out: &mut Vec<u8>) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for &b in input {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b),
            _ => {
                out.push(b'%');
                out.push(HEX[(b >> 4) as usize]);
                out.push(HEX[(b & 0x0f) as usize]);
            }
        }
    }
}

/// HTML-escape bytes directly into output buffer.
fn html_encode_to(input: &[u8], out: &mut Vec<u8>) {
    for &b in input {
        match b {
            b'<' => out.extend_from_slice(b"&lt;"),
            b'>' => out.extend_from_slice(b"&gt;"),
            b'&' => out.extend_from_slice(b"&amp;"),
            b'\'' => out.extend_from_slice(b"&#39;"),
            _ => out.push(b),
        }
    }
}

/// Get field names referenced by a RemapExpr.
fn remap_expr_fields(rexpr: &jq_jit::interpreter::RemapExpr) -> Vec<&str> {
    use jq_jit::interpreter::RemapExpr;
    match rexpr {
        RemapExpr::Field(f) | RemapExpr::FieldToString(f) => vec![f.as_str()],
        RemapExpr::FieldOpConst(f, _, _) | RemapExpr::FieldCmpConst(f, _, _) | RemapExpr::FieldOpConstToString(f, _, _) => vec![f.as_str()],
        RemapExpr::FieldOpField(f1, _, f2) | RemapExpr::FieldCmpField(f1, _, f2) => vec![f1.as_str(), f2.as_str()],
        RemapExpr::ConstOpField(_, _, f) => vec![f.as_str()],
        RemapExpr::Arith(_, fields) => fields.iter().map(|f| f.as_str()).collect(),
        RemapExpr::FieldMinMax(f1, f2, _) => vec![f1.as_str(), f2.as_str()],
        RemapExpr::LiteralJson(_) => vec![],
        RemapExpr::FieldLength(f) => vec![f.as_str()],
        RemapExpr::StringInterp(parts) => {
            let mut fields = Vec::new();
            for part in parts {
                if let jq_jit::interpreter::InterpPart::Field(f) = part {
                    if !fields.contains(&f.as_str()) {
                        fields.push(f.as_str());
                    }
                }
            }
            fields
        }
        RemapExpr::FieldSplitJoin(f, _, _) => vec![f.as_str()],
        RemapExpr::FieldStringCase(f, _) => vec![f.as_str()],
        RemapExpr::FieldSplitLength(f, _) => vec![f.as_str()],
        RemapExpr::FieldStrBuiltin(f, _, _) => vec![f.as_str()],
        RemapExpr::FieldSplitIndex(f, _, _) => vec![f.as_str()],
        RemapExpr::FieldOpFieldToString(f1, _, f2) => vec![f1.as_str(), f2.as_str()],
        RemapExpr::ArithToString(_, fields) | RemapExpr::ArithUnary(_, _, fields) | RemapExpr::ArithCmp(_, _, _, fields) => fields.iter().map(|f| f.as_str()).collect(),
        RemapExpr::FieldSlice(f, _, _) => vec![f.as_str()],
        RemapExpr::FieldArray(ref exprs) => {
            let mut fields: Vec<&str> = Vec::new();
            for rexpr in exprs {
                for name in remap_expr_fields(rexpr) {
                    if !fields.contains(&name) { fields.push(name); }
                }
            }
            fields
        }
        RemapExpr::BoolExpr(ref l, _, ref r) => {
            let mut fields = remap_expr_fields(l);
            for name in remap_expr_fields(r) {
                if !fields.contains(&name) { fields.push(name); }
            }
            fields
        }
        RemapExpr::FieldType(f) | RemapExpr::FieldNegate(f) => vec![f.as_str()],
        RemapExpr::CondChain(branches, else_out) => {
            let mut fields: Vec<&str> = Vec::new();
            for b in branches {
                let f = b.cond_field.as_str();
                if !fields.contains(&f) { fields.push(f); }
                if let jq_jit::interpreter::CondRhs::Field(f2) = &b.cond_rhs {
                    let f2s = f2.as_str();
                    if !fields.contains(&f2s) { fields.push(f2s); }
                }
                collect_branch_output_fields(&b.output, &mut fields);
            }
            collect_branch_output_fields(else_out, &mut fields);
            fields
        }
    }
}

fn collect_branch_output_fields<'a>(out: &'a jq_jit::interpreter::BranchOutput, fields: &mut Vec<&'a str>) {
    use jq_jit::interpreter::BranchOutput;
    match out {
        BranchOutput::Field(f) => {
            if !fields.contains(&f.as_str()) { fields.push(f.as_str()); }
        }
        BranchOutput::Remap(entries) => {
            for (_, rexpr) in entries {
                for name in remap_expr_fields(rexpr) {
                    if !fields.contains(&name) { fields.push(name); }
                }
            }
        }
        BranchOutput::Computed(rexpr) => {
            for name in remap_expr_fields(rexpr) {
                if !fields.contains(&name) { fields.push(name); }
            }
        }
        _ => {}
    }
}

/// Build JSON object key prefixes: ["{\"key1\":", ",\"key2\":", ...]
fn build_obj_key_prefixes<'a>(keys: impl Iterator<Item = &'a str>) -> Vec<Vec<u8>> {
    let mut prefixes = Vec::new();
    for (i, key) in keys.enumerate() {
        let mut prefix = Vec::new();
        if i == 0 { prefix.push(b'{'); } else { prefix.push(b','); }
        prefix.push(b'"');
        prefix.extend_from_slice(key.as_bytes());
        prefix.extend_from_slice(b"\":");
        prefixes.push(prefix);
    }
    prefixes
}

fn build_obj_key_prefixes_pretty<'a>(keys: impl Iterator<Item = &'a str>) -> Vec<Vec<u8>> {
    let mut prefixes = Vec::new();
    for (i, key) in keys.enumerate() {
        let mut prefix = Vec::new();
        if i == 0 { prefix.extend_from_slice(b"{\n  \""); } else { prefix.extend_from_slice(b",\n  \""); }
        prefix.extend_from_slice(key.as_bytes());
        prefix.extend_from_slice(b"\": ");
        prefixes.push(prefix);
    }
    prefixes
}

/// Pre-resolved remap expression — uses indices into the ranges array instead of
/// HashMap lookups for field names. Resolved once before the hot loop.
#[derive(Clone)]
enum ResolvedRemap {
    Field(usize),
    FieldOpConst(usize, jq_jit::ir::BinOp, f64),
    FieldOpField(usize, jq_jit::ir::BinOp, usize),
    ConstOpField(f64, jq_jit::ir::BinOp, usize),
    FieldCmpConst(usize, jq_jit::ir::BinOp, f64),
    FieldCmpField(usize, jq_jit::ir::BinOp, usize),
    FieldToString(usize),
    FieldOpConstToString(usize, jq_jit::ir::BinOp, f64),
    Arith(jq_jit::interpreter::ArithExpr),
    FieldMinMax(usize, usize, bool),
    LiteralJson(Vec<u8>),
    FieldLength(usize),
    StringInterp(Vec<ResolvedInterpPart>),
    FieldSplitJoin(usize, Vec<u8>, Vec<u8>), // field_idx, split_sep bytes, join_rep bytes
    FieldStringCase(usize, bool), // field_idx, is_upper
    FieldSplitLength(usize, Vec<u8>), // field_idx, separator bytes
    CondChain(Vec<ResolvedCondBranch>, Box<ResolvedBranchOutput>),
    FieldStrBuiltin(usize, jq_jit::interpreter::StrBuiltin, Vec<u8>), // field_idx, op, arg_bytes
    FieldSplitIndex(usize, Vec<u8>, i32), // field_idx, sep_bytes, index
    FieldOpFieldToString(usize, jq_jit::ir::BinOp, usize), // idx1, op, idx2
    ArithToString(jq_jit::interpreter::ArithExpr),
    ArithUnary(jq_jit::interpreter::MathUnary, jq_jit::interpreter::ArithExpr),
    FieldSlice(usize, Option<i64>, Option<i64>),
    FieldArray(Vec<ResolvedRemap>),
    BoolExpr(Box<ResolvedRemap>, jq_jit::ir::BinOp, Box<ResolvedRemap>),
    FieldType(usize),
    FieldNegate(usize),
    ArithCmp(jq_jit::interpreter::ArithExpr, jq_jit::ir::BinOp, f64),
}

/// Pre-resolved conditional branch for remap values.
#[derive(Clone)]
struct ResolvedCondBranch {
    cond_field_idx: usize,
    cond_arith_ops: Vec<(jq_jit::ir::BinOp, f64)>,
    cond_op: jq_jit::ir::BinOp,
    cond_rhs: ResolvedCondRhs,
    output: ResolvedBranchOutput,
}

#[derive(Clone)]
enum ResolvedCondRhs {
    Const(f64),
    Field(usize),
    Str(Vec<u8>), // raw bytes of the JSON string including quotes
    Null,
    Bool(bool),
    Startswith(Vec<u8>), // raw pattern bytes (no quotes)
    Endswith(Vec<u8>),
    Contains(Vec<u8>),
}

#[derive(Clone)]
enum ResolvedBranchOutput {
    Literal(Vec<u8>),
    Field(usize),
    Empty,
    Remap(Vec<ResolvedRemap>),
    Computed(Box<ResolvedRemap>),
}

/// Pre-resolved string interpolation part.
#[derive(Clone)]
enum ResolvedInterpPart {
    /// Pre-escaped JSON bytes for literal text
    Literal(Vec<u8>),
    /// Field index
    Field(usize),
}

/// Pre-resolve RemapExpr → ResolvedRemap using a field→index map.
fn resolve_remap_exprs(
    exprs: &[(impl AsRef<str>, jq_jit::interpreter::RemapExpr)],
    field_idx: &std::collections::HashMap<String, usize>,
) -> Vec<ResolvedRemap> {
    exprs.iter().map(|(_, rexpr)| resolve_one_remap(rexpr, field_idx)).collect()
}

fn resolve_one_remap(
    rexpr: &jq_jit::interpreter::RemapExpr,
    field_idx: &std::collections::HashMap<String, usize>,
) -> ResolvedRemap {
    use jq_jit::interpreter::RemapExpr;
    match rexpr {
        RemapExpr::Field(f) => ResolvedRemap::Field(field_idx[f.as_str()]),
        RemapExpr::FieldOpConst(f, op, n) => ResolvedRemap::FieldOpConst(field_idx[f.as_str()], *op, *n),
        RemapExpr::FieldOpField(f1, op, f2) => ResolvedRemap::FieldOpField(field_idx[f1.as_str()], *op, field_idx[f2.as_str()]),
        RemapExpr::ConstOpField(n, op, f) => ResolvedRemap::ConstOpField(*n, *op, field_idx[f.as_str()]),
        RemapExpr::FieldCmpConst(f, op, n) => ResolvedRemap::FieldCmpConst(field_idx[f.as_str()], *op, *n),
        RemapExpr::FieldCmpField(f1, op, f2) => ResolvedRemap::FieldCmpField(field_idx[f1.as_str()], *op, field_idx[f2.as_str()]),
        RemapExpr::FieldToString(f) => ResolvedRemap::FieldToString(field_idx[f.as_str()]),
        RemapExpr::FieldOpConstToString(f, op, n) => ResolvedRemap::FieldOpConstToString(field_idx[f.as_str()], *op, *n),
        RemapExpr::Arith(arith, fields) => ResolvedRemap::Arith(resolve_arith_expr(arith, fields, field_idx)),
        RemapExpr::FieldMinMax(f1, f2, is_max) => ResolvedRemap::FieldMinMax(field_idx[f1.as_str()], field_idx[f2.as_str()], *is_max),
        RemapExpr::LiteralJson(ref bytes) => ResolvedRemap::LiteralJson(bytes.clone()),
        RemapExpr::FieldLength(f) => ResolvedRemap::FieldLength(field_idx[f.as_str()]),
        RemapExpr::FieldSplitJoin(f, sep, rep) => {
            ResolvedRemap::FieldSplitJoin(field_idx[f.as_str()], sep.as_bytes().to_vec(), rep.as_bytes().to_vec())
        }
        RemapExpr::FieldStringCase(f, is_upper) => {
            ResolvedRemap::FieldStringCase(field_idx[f.as_str()], *is_upper)
        }
        RemapExpr::FieldSplitLength(f, sep) => {
            ResolvedRemap::FieldSplitLength(field_idx[f.as_str()], sep.as_bytes().to_vec())
        }
        RemapExpr::CondChain(branches, else_out) => {
            let resolved_branches = branches.iter().map(|b| {
                ResolvedCondBranch {
                    cond_field_idx: field_idx[b.cond_field.as_str()],
                    cond_arith_ops: b.cond_arith_ops.clone(),
                    cond_op: b.cond_op,
                    cond_rhs: match &b.cond_rhs {
                        jq_jit::interpreter::CondRhs::Const(n) => ResolvedCondRhs::Const(*n),
                        jq_jit::interpreter::CondRhs::Field(f) => ResolvedCondRhs::Field(field_idx[f.as_str()]),
                        jq_jit::interpreter::CondRhs::Str(s) => {
                            let mut buf = Vec::with_capacity(s.len() + 2);
                            buf.push(b'"');
                            buf.extend_from_slice(s.as_bytes());
                            buf.push(b'"');
                            ResolvedCondRhs::Str(buf)
                        }
                        jq_jit::interpreter::CondRhs::Null => ResolvedCondRhs::Null,
                        jq_jit::interpreter::CondRhs::Bool(b) => ResolvedCondRhs::Bool(*b),
                        jq_jit::interpreter::CondRhs::Startswith(s) => ResolvedCondRhs::Startswith(s.as_bytes().to_vec()),
                        jq_jit::interpreter::CondRhs::Endswith(s) => ResolvedCondRhs::Endswith(s.as_bytes().to_vec()),
                        jq_jit::interpreter::CondRhs::Contains(s) => ResolvedCondRhs::Contains(s.as_bytes().to_vec()),
                    },
                    output: resolve_branch_output(&b.output, field_idx),
                }
            }).collect();
            let resolved_else = resolve_branch_output(else_out, field_idx);
            ResolvedRemap::CondChain(resolved_branches, Box::new(resolved_else))
        }
        RemapExpr::FieldStrBuiltin(f, op, arg) => {
            ResolvedRemap::FieldStrBuiltin(field_idx[f.as_str()], *op, arg.as_bytes().to_vec())
        }
        RemapExpr::FieldSplitIndex(f, sep, idx) => {
            ResolvedRemap::FieldSplitIndex(field_idx[f.as_str()], sep.as_bytes().to_vec(), *idx)
        }
        RemapExpr::FieldOpFieldToString(f1, op, f2) => {
            ResolvedRemap::FieldOpFieldToString(field_idx[f1.as_str()], *op, field_idx[f2.as_str()])
        }
        RemapExpr::ArithToString(arith, fields) => {
            ResolvedRemap::ArithToString(resolve_arith_expr(arith, fields, field_idx))
        }
        RemapExpr::ArithUnary(math_op, arith, fields) => {
            ResolvedRemap::ArithUnary(*math_op, resolve_arith_expr(arith, fields, field_idx))
        }
        RemapExpr::FieldSlice(f, from, to) => {
            ResolvedRemap::FieldSlice(field_idx[f.as_str()], *from, *to)
        }
        RemapExpr::FieldArray(ref exprs) => {
            ResolvedRemap::FieldArray(exprs.iter().map(|e| resolve_one_remap(e, field_idx)).collect())
        }
        RemapExpr::BoolExpr(ref l, op, ref r) => {
            ResolvedRemap::BoolExpr(
                Box::new(resolve_one_remap(l, field_idx)),
                *op,
                Box::new(resolve_one_remap(r, field_idx)),
            )
        }
        RemapExpr::FieldType(f) => {
            ResolvedRemap::FieldType(field_idx[f.as_str()])
        }
        RemapExpr::FieldNegate(f) => {
            ResolvedRemap::FieldNegate(field_idx[f.as_str()])
        }
        RemapExpr::ArithCmp(arith, cmp_op, rhs_const, fields) => {
            ResolvedRemap::ArithCmp(resolve_arith_expr(arith, fields, field_idx), *cmp_op, *rhs_const)
        }
        RemapExpr::StringInterp(parts) => {
            let resolved = parts.iter().map(|p| match p {
                jq_jit::interpreter::InterpPart::Literal(s) => {
                    let mut buf = Vec::new();
                    for &b in s.as_bytes() {
                        match b {
                            b'"' => buf.extend_from_slice(b"\\\""),
                            b'\\' => buf.extend_from_slice(b"\\\\"),
                            b'\n' => buf.extend_from_slice(b"\\n"),
                            b'\r' => buf.extend_from_slice(b"\\r"),
                            b'\t' => buf.extend_from_slice(b"\\t"),
                            c if c < 0x20 => { use std::io::Write; let _ = write!(buf, "\\u{:04x}", c); }
                            _ => buf.push(b),
                        }
                    }
                    ResolvedInterpPart::Literal(buf)
                }
                jq_jit::interpreter::InterpPart::Field(f) => {
                    ResolvedInterpPart::Field(field_idx[f.as_str()])
                }
            }).collect();
            ResolvedRemap::StringInterp(resolved)
        }
    }
}

/// Remap ArithExpr field indices from local (into `fields` vec) to global (into `field_idx` map).
fn resolve_arith_expr(
    expr: &jq_jit::interpreter::ArithExpr,
    local_fields: &[String],
    field_idx: &std::collections::HashMap<String, usize>,
) -> jq_jit::interpreter::ArithExpr {
    use jq_jit::interpreter::ArithExpr;
    match expr {
        ArithExpr::Field(local_idx) => ArithExpr::Field(field_idx[local_fields[*local_idx].as_str()]),
        ArithExpr::Const(n) => ArithExpr::Const(*n),
        ArithExpr::BinOp(op, lhs, rhs) => ArithExpr::BinOp(
            *op,
            Box::new(resolve_arith_expr(lhs, local_fields, field_idx)),
            Box::new(resolve_arith_expr(rhs, local_fields, field_idx)),
        ),
        ArithExpr::Unary(op, inner) => ArithExpr::Unary(
            *op,
            Box::new(resolve_arith_expr(inner, local_fields, field_idx)),
        ),
    }
}

/// Resolve a BranchOutput to a ResolvedBranchOutput.
fn resolve_branch_output(
    out: &jq_jit::interpreter::BranchOutput,
    field_idx: &std::collections::HashMap<String, usize>,
) -> ResolvedBranchOutput {
    use jq_jit::interpreter::BranchOutput;
    match out {
        BranchOutput::Literal(bytes) => ResolvedBranchOutput::Literal(bytes.clone()),
        BranchOutput::Field(f) => ResolvedBranchOutput::Field(field_idx[f.as_str()]),
        BranchOutput::Empty => ResolvedBranchOutput::Empty,
        BranchOutput::Remap(entries) => {
            let resolved = entries.iter().map(|(_, rexpr)| resolve_one_remap(rexpr, field_idx)).collect();
            ResolvedBranchOutput::Remap(resolved)
        }
        BranchOutput::Computed(rexpr) => {
            ResolvedBranchOutput::Computed(Box::new(resolve_one_remap(rexpr, field_idx)))
        }
    }
}

/// Resolve a slice of RemapExpr (without keys) for computed_array.
fn resolve_remap_exprs_array(
    exprs: &[jq_jit::interpreter::RemapExpr],
    field_idx: &std::collections::HashMap<String, usize>,
) -> Vec<ResolvedRemap> {
    exprs.iter().map(|rexpr| resolve_one_remap(rexpr, field_idx)).collect()
}

/// Emit tostring on a raw JSON value: numbers → "N", strings → as-is, booleans/null → "true"/"false"/"null".
#[inline]
fn emit_tostring_raw(buf: &mut Vec<u8>, val: &[u8]) {
    if val.is_empty() { buf.extend_from_slice(b"null"); return; }
    match val[0] {
        b'"' => buf.extend_from_slice(val), // string → as-is
        b't' => buf.extend_from_slice(b"\"true\""),
        b'f' => buf.extend_from_slice(b"\"false\""),
        b'n' => buf.extend_from_slice(b"\"null\""),
        _ => {
            // number → wrap in quotes
            buf.push(b'"');
            // Re-format through push_jq_number_bytes for jq compat
            if let Some(n) = parse_json_num(val) {
                push_jq_number_bytes(buf, n);
            } else {
                buf.extend_from_slice(val);
            }
            buf.push(b'"');
        }
    }
}

/// Emit a raw JSON field value into a string interpolation context (tostring, no outer quotes).
/// Strings: copy inner content (already JSON-escaped). Numbers/bool/null: copy as-is.
/// Arrays/objects: copy raw JSON but escape any " and \ for embedding in JSON string.
#[inline]
fn emit_interp_field_raw(buf: &mut Vec<u8>, val: &[u8]) {
    if val.is_empty() { return; }
    match val[0] {
        b'"' if val.len() >= 2 => {
            // String: inner content is already JSON-escaped
            buf.extend_from_slice(&val[1..val.len()-1]);
        }
        b't' => buf.extend_from_slice(b"true"),
        b'f' => buf.extend_from_slice(b"false"),
        b'n' => buf.extend_from_slice(b"null"),
        b'[' | b'{' => {
            // Array/object: tojson — need to escape " and \ for embedding
            for &b in val {
                match b {
                    b'"' => buf.extend_from_slice(b"\\\""),
                    b'\\' => buf.extend_from_slice(b"\\\\"),
                    _ => buf.push(b),
                }
            }
        }
        _ => {
            // Number: re-format for jq compat
            if let Some(n) = parse_json_num(val) {
                push_jq_number_bytes(buf, n);
            } else {
                buf.extend_from_slice(val);
            }
        }
    }
}

/// Emit a resolved branch output into the buffer.
#[inline]
fn emit_resolved_branch_output(
    buf: &mut Vec<u8>,
    out: &ResolvedBranchOutput,
    raw: &[u8],
    ranges: &[(usize, usize)],
) {
    match out {
        ResolvedBranchOutput::Literal(ref bytes) => buf.extend_from_slice(bytes),
        ResolvedBranchOutput::Field(idx) => {
            let (vs, ve) = ranges[*idx];
            buf.extend_from_slice(&raw[vs..ve]);
        }
        ResolvedBranchOutput::Empty => {}
        ResolvedBranchOutput::Remap(ref resolved_values) => {
            let _ = resolved_values;
            buf.extend_from_slice(b"null");
        }
        ResolvedBranchOutput::Computed(ref resolved) => {
            emit_resolved_value(buf, resolved, raw, ranges);
        }
    }
}

/// Emit `.field | split(sep) | join(rep)` on a raw JSON string value.
/// Replaces all occurrences of `sep` with `rep` inside the string content.
#[inline]
fn emit_split_join_raw(buf: &mut Vec<u8>, val: &[u8], sep: &[u8], rep: &[u8]) {
    if val.len() < 2 || val[0] != b'"' {
        buf.extend_from_slice(b"null");
        return;
    }
    let content = &val[1..val.len()-1];
    buf.push(b'"');
    if sep.is_empty() {
        // split("") produces individual chars — join(rep) puts rep between each char
        let mut first = true;
        let mut i = 0;
        while i < content.len() {
            if !first {
                // JSON-escape the replacement
                for &b in rep {
                    match b {
                        b'"' => buf.extend_from_slice(b"\\\""),
                        b'\\' => buf.extend_from_slice(b"\\\\"),
                        _ => buf.push(b),
                    }
                }
            }
            first = false;
            if content[i] == b'\\' && i + 1 < content.len() {
                // Copy escape sequence as-is (already valid JSON)
                buf.push(content[i]);
                buf.push(content[i+1]);
                if content[i+1] == b'u' && i + 5 < content.len() {
                    buf.extend_from_slice(&content[i+2..i+6]);
                    i += 6;
                } else {
                    i += 2;
                }
            } else {
                // UTF-8: copy full codepoint bytes
                let b0 = content[i];
                let clen = if b0 < 0x80 { 1 } else if b0 < 0xE0 { 2 } else if b0 < 0xF0 { 3 } else { 4 };
                let end = (i + clen).min(content.len());
                buf.extend_from_slice(&content[i..end]);
                i = end;
            }
        }
    } else {
        // Replace sep with rep in content bytes
        let mut pos = 0;
        while pos < content.len() {
            if pos + sep.len() <= content.len() && &content[pos..pos+sep.len()] == sep {
                // JSON-escape the replacement
                for &b in rep {
                    match b {
                        b'"' => buf.extend_from_slice(b"\\\""),
                        b'\\' => buf.extend_from_slice(b"\\\\"),
                        _ => buf.push(b),
                    }
                }
                pos += sep.len();
            } else {
                buf.push(content[pos]);
                pos += 1;
            }
        }
    }
    buf.push(b'"');
}

/// Fast substring search using memchr for the first byte.
#[inline]
fn bytes_contains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() { return true; }
    if needle.len() > haystack.len() { return false; }
    if needle.len() == 1 {
        return memchr::memchr(needle[0], haystack).is_some();
    }
    let first = needle[0];
    let rest = &needle[1..];
    let max_start = haystack.len() - needle.len();
    let mut pos = 0;
    loop {
        match memchr::memchr(first, &haystack[pos..=max_start]) {
            Some(offset) => {
                let start = pos + offset;
                if haystack[start + 1..start + needle.len()] == *rest {
                    return true;
                }
                pos = start + 1;
            }
            None => return false,
        }
    }
}

/// Emit a single computed remap value into the output buffer.
/// Shared by computed_remap, computed_array, select_cmp_cremap handlers.
#[inline]
fn emit_remap_value(
    buf: &mut Vec<u8>,
    rexpr: &jq_jit::interpreter::RemapExpr,
    raw: &[u8],
    ranges: &[(usize, usize)],
    field_idx: &std::collections::HashMap<String, usize>,
) {
    use jq_jit::interpreter::RemapExpr;
    use jq_jit::ir::BinOp;
    match rexpr {
        RemapExpr::Field(f) => {
            let idx = field_idx[f.as_str()];
            let (vs, ve) = ranges[idx];
            buf.extend_from_slice(&raw[vs..ve]);
        }
        RemapExpr::FieldOpConst(f, op, n) => {
            let idx = field_idx[f.as_str()];
            let (vs, ve) = ranges[idx];
            if let Some(a) = parse_json_num(&raw[vs..ve]) {
                let r = match op { BinOp::Add => a + n, BinOp::Sub => a - n, BinOp::Mul => a * n, BinOp::Div => a / n, BinOp::Mod => a % n, _ => unreachable!() };
                push_jq_number_bytes(buf, r);
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::FieldOpField(f1, op, f2) => {
            let idx1 = field_idx[f1.as_str()];
            let idx2 = field_idx[f2.as_str()];
            let (vs1, ve1) = ranges[idx1];
            let (vs2, ve2) = ranges[idx2];
            if let (Some(a), Some(b)) = (parse_json_num(&raw[vs1..ve1]), parse_json_num(&raw[vs2..ve2])) {
                let r = match op { BinOp::Add => a + b, BinOp::Sub => a - b, BinOp::Mul => a * b, BinOp::Div => a / b, BinOp::Mod => a % b, _ => unreachable!() };
                push_jq_number_bytes(buf, r);
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::ConstOpField(n, op, f) => {
            let idx = field_idx[f.as_str()];
            let (vs, ve) = ranges[idx];
            if let Some(b) = parse_json_num(&raw[vs..ve]) {
                let r = match op { BinOp::Add => n + b, BinOp::Sub => n - b, BinOp::Mul => n * b, BinOp::Div => n / b, BinOp::Mod => n % b, _ => unreachable!() };
                push_jq_number_bytes(buf, r);
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::FieldCmpConst(f, op, n) => {
            let idx = field_idx[f.as_str()];
            let (vs, ve) = ranges[idx];
            if let Some(a) = parse_json_num(&raw[vs..ve]) {
                let r = match op { BinOp::Gt => a > *n, BinOp::Lt => a < *n, BinOp::Ge => a >= *n, BinOp::Le => a <= *n, BinOp::Eq => a == *n, BinOp::Ne => a != *n, _ => unreachable!() };
                buf.extend_from_slice(if r { b"true" } else { b"false" });
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::FieldCmpField(f1, op, f2) => {
            let idx1 = field_idx[f1.as_str()];
            let idx2 = field_idx[f2.as_str()];
            let (vs1, ve1) = ranges[idx1];
            let (vs2, ve2) = ranges[idx2];
            if let (Some(a), Some(b)) = (parse_json_num(&raw[vs1..ve1]), parse_json_num(&raw[vs2..ve2])) {
                let r = match op { BinOp::Gt => a > b, BinOp::Lt => a < b, BinOp::Ge => a >= b, BinOp::Le => a <= b, BinOp::Eq => a == b, BinOp::Ne => a != b, _ => unreachable!() };
                buf.extend_from_slice(if r { b"true" } else { b"false" });
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::FieldToString(f) => {
            let idx = field_idx[f.as_str()];
            let (vs, ve) = ranges[idx];
            emit_tostring_raw(buf, &raw[vs..ve]);
        }
        RemapExpr::FieldOpConstToString(f, op, n) => {
            let idx = field_idx[f.as_str()];
            let (vs, ve) = ranges[idx];
            if let Some(a) = parse_json_num(&raw[vs..ve]) {
                let r = match op { BinOp::Add => a + n, BinOp::Sub => a - n, BinOp::Mul => a * n, BinOp::Div => a / n, BinOp::Mod => a % n, _ => unreachable!() };
                buf.push(b'"');
                push_jq_number_bytes(buf, r);
                buf.push(b'"');
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::Arith(arith, fields) => {
            // Resolve local ArithExpr field indices to global indices on the fly
            if let Some(r) = eval_arith_raw_unresolved(arith, fields, raw, ranges, field_idx) {
                push_jq_number_bytes(buf, r);
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::FieldMinMax(f1, f2, is_max) => {
            let idx1 = field_idx[f1.as_str()];
            let idx2 = field_idx[f2.as_str()];
            let (vs1, ve1) = ranges[idx1];
            let (vs2, ve2) = ranges[idx2];
            if let (Some(a), Some(b)) = (parse_json_num(&raw[vs1..ve1]), parse_json_num(&raw[vs2..ve2])) {
                let r = if *is_max { if a >= b { a } else { b } } else { if a <= b { a } else { b } };
                push_jq_number_bytes(buf, r);
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::LiteralJson(ref bytes) => {
            buf.extend_from_slice(bytes);
        }
        RemapExpr::FieldLength(f) => {
            let idx = field_idx[f.as_str()];
            let resolved = ResolvedRemap::FieldLength(idx);
            emit_resolved_value(buf, &resolved, raw, ranges);
        }
        RemapExpr::FieldSplitJoin(f, sep, rep) => {
            let idx = field_idx[f.as_str()];
            let (vs, ve) = ranges[idx];
            emit_split_join_raw(buf, &raw[vs..ve], sep.as_bytes(), rep.as_bytes());
        }
        RemapExpr::FieldStringCase(f, is_upper) => {
            let idx = field_idx[f.as_str()];
            let resolved = ResolvedRemap::FieldStringCase(idx, *is_upper);
            emit_resolved_value(buf, &resolved, raw, ranges);
        }
        RemapExpr::FieldSplitLength(f, sep) => {
            let idx = field_idx[f.as_str()];
            let resolved = ResolvedRemap::FieldSplitLength(idx, sep.as_bytes().to_vec());
            emit_resolved_value(buf, &resolved, raw, ranges);
        }
        RemapExpr::FieldStrBuiltin(f, op, arg) => {
            let idx = field_idx[f.as_str()];
            let resolved = ResolvedRemap::FieldStrBuiltin(idx, *op, arg.as_bytes().to_vec());
            emit_resolved_value(buf, &resolved, raw, ranges);
        }
        RemapExpr::FieldSplitIndex(f, sep, sidx) => {
            let idx = field_idx[f.as_str()];
            let resolved = ResolvedRemap::FieldSplitIndex(idx, sep.as_bytes().to_vec(), *sidx);
            emit_resolved_value(buf, &resolved, raw, ranges);
        }
        RemapExpr::FieldOpFieldToString(f1, op, f2) => {
            let idx1 = field_idx[f1.as_str()];
            let idx2 = field_idx[f2.as_str()];
            let (vs1, ve1) = ranges[idx1];
            let (vs2, ve2) = ranges[idx2];
            if let (Some(a), Some(b)) = (parse_json_num(&raw[vs1..ve1]), parse_json_num(&raw[vs2..ve2])) {
                let r = match op { BinOp::Add => a + b, BinOp::Sub => a - b, BinOp::Mul => a * b, BinOp::Div => a / b, BinOp::Mod => a % b, _ => unreachable!() };
                buf.push(b'"');
                push_jq_number_bytes(buf, r);
                buf.push(b'"');
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::ArithToString(arith, fields) => {
            if let Some(r) = eval_arith_raw_unresolved(arith, fields, raw, ranges, field_idx) {
                buf.push(b'"');
                push_jq_number_bytes(buf, r);
                buf.push(b'"');
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::ArithUnary(math_op, arith, fields) => {
            if let Some(r) = eval_arith_raw_unresolved(arith, fields, raw, ranges, field_idx) {
                let result = apply_math_unary(*math_op, r);
                push_jq_number_bytes(buf, result);
            } else { buf.extend_from_slice(b"null"); }
        }
        RemapExpr::FieldSlice(f, from, to) => {
            let idx = field_idx[f.as_str()];
            let resolved = ResolvedRemap::FieldSlice(idx, *from, *to);
            emit_resolved_value(buf, &resolved, raw, ranges);
        }
        RemapExpr::FieldArray(ref exprs) => {
            buf.push(b'[');
            for (i, rexpr) in exprs.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                emit_remap_value(buf, rexpr, raw, ranges, field_idx);
            }
            buf.push(b']');
        }
        RemapExpr::BoolExpr(_, _, _) | RemapExpr::FieldType(_) | RemapExpr::FieldNegate(_) | RemapExpr::ArithCmp(_, _, _, _) => {
            let resolved = resolve_one_remap(rexpr, field_idx);
            emit_resolved_value(buf, &resolved, raw, ranges);
        }
        RemapExpr::CondChain(_, _) => {
            // Resolve and emit (slow path, used for non-pre-resolved paths)
            let resolved = resolve_one_remap(rexpr, field_idx);
            emit_resolved_value(buf, &resolved, raw, ranges);
        }
        RemapExpr::StringInterp(parts) => {
            buf.push(b'"');
            for part in parts {
                match part {
                    jq_jit::interpreter::InterpPart::Literal(s) => {
                        for &b in s.as_bytes() {
                            match b {
                                b'"' => buf.extend_from_slice(b"\\\""),
                                b'\\' => buf.extend_from_slice(b"\\\\"),
                                b'\n' => buf.extend_from_slice(b"\\n"),
                                b'\r' => buf.extend_from_slice(b"\\r"),
                                b'\t' => buf.extend_from_slice(b"\\t"),
                                c if c < 0x20 => { use std::io::Write; let _ = write!(buf, "\\u{:04x}", c); }
                                _ => buf.push(b),
                            }
                        }
                    }
                    jq_jit::interpreter::InterpPart::Field(f) => {
                        let idx = field_idx[f.as_str()];
                        let (vs, ve) = ranges[idx];
                        emit_interp_field_raw(buf, &raw[vs..ve]);
                    }
                }
            }
            buf.push(b'"');
        }
    }
}

/// Evaluate an ArithExpr using unresolved field names (for emit_remap_value).
#[inline]
fn eval_arith_raw_unresolved(
    expr: &jq_jit::interpreter::ArithExpr,
    local_fields: &[String],
    raw: &[u8],
    ranges: &[(usize, usize)],
    field_idx: &std::collections::HashMap<String, usize>,
) -> Option<f64> {
    use jq_jit::interpreter::ArithExpr;
    use jq_jit::ir::BinOp;
    match expr {
        ArithExpr::Field(local_idx) => {
            let global_idx = field_idx[local_fields[*local_idx].as_str()];
            let (vs, ve) = ranges[global_idx];
            parse_json_num(&raw[vs..ve])
        }
        ArithExpr::Const(n) => Some(*n),
        ArithExpr::BinOp(op, lhs, rhs) => {
            let l = eval_arith_raw_unresolved(lhs, local_fields, raw, ranges, field_idx)?;
            let r = eval_arith_raw_unresolved(rhs, local_fields, raw, ranges, field_idx)?;
            Some(match op {
                BinOp::Add => l + r,
                BinOp::Sub => l - r,
                BinOp::Mul => l * r,
                BinOp::Div => l / r,
                BinOp::Mod => l % r,
                _ => return None,
            })
        }
        ArithExpr::Unary(op, inner) => {
            let v = eval_arith_raw_unresolved(inner, local_fields, raw, ranges, field_idx)?;
            Some(apply_math_unary(*op, v))
        }
    }
}

/// Emit a pre-resolved remap value — no HashMap lookups, just direct index access.
#[inline]
fn emit_resolved_value(
    buf: &mut Vec<u8>,
    resolved: &ResolvedRemap,
    raw: &[u8],
    ranges: &[(usize, usize)],
) {
    use jq_jit::ir::BinOp;
    match *resolved {
        ResolvedRemap::Field(idx) => {
            let (vs, ve) = ranges[idx];
            buf.extend_from_slice(&raw[vs..ve]);
        }
        ResolvedRemap::FieldOpConst(idx, ref op, n) => {
            let (vs, ve) = ranges[idx];
            if let Some(a) = parse_json_num(&raw[vs..ve]) {
                let r = match op { BinOp::Add => a + n, BinOp::Sub => a - n, BinOp::Mul => a * n, BinOp::Div => a / n, BinOp::Mod => a % n, _ => unreachable!() };
                push_jq_number_bytes(buf, r);
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::FieldOpField(idx1, ref op, idx2) => {
            let (vs1, ve1) = ranges[idx1];
            let (vs2, ve2) = ranges[idx2];
            if let (Some(a), Some(b)) = (parse_json_num(&raw[vs1..ve1]), parse_json_num(&raw[vs2..ve2])) {
                let r = match op { BinOp::Add => a + b, BinOp::Sub => a - b, BinOp::Mul => a * b, BinOp::Div => a / b, BinOp::Mod => a % b, _ => unreachable!() };
                push_jq_number_bytes(buf, r);
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::ConstOpField(n, ref op, idx) => {
            let (vs, ve) = ranges[idx];
            if let Some(b) = parse_json_num(&raw[vs..ve]) {
                let r = match op { BinOp::Add => n + b, BinOp::Sub => n - b, BinOp::Mul => n * b, BinOp::Div => n / b, BinOp::Mod => n % b, _ => unreachable!() };
                push_jq_number_bytes(buf, r);
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::FieldCmpConst(idx, ref op, n) => {
            let (vs, ve) = ranges[idx];
            if let Some(a) = parse_json_num(&raw[vs..ve]) {
                let r = match op { BinOp::Gt => a > n, BinOp::Lt => a < n, BinOp::Ge => a >= n, BinOp::Le => a <= n, BinOp::Eq => a == n, BinOp::Ne => a != n, _ => unreachable!() };
                buf.extend_from_slice(if r { b"true" } else { b"false" });
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::FieldCmpField(idx1, ref op, idx2) => {
            let (vs1, ve1) = ranges[idx1];
            let (vs2, ve2) = ranges[idx2];
            if let (Some(a), Some(b)) = (parse_json_num(&raw[vs1..ve1]), parse_json_num(&raw[vs2..ve2])) {
                let r = match op { BinOp::Gt => a > b, BinOp::Lt => a < b, BinOp::Ge => a >= b, BinOp::Le => a <= b, BinOp::Eq => a == b, BinOp::Ne => a != b, _ => unreachable!() };
                buf.extend_from_slice(if r { b"true" } else { b"false" });
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::FieldToString(idx) => {
            let (vs, ve) = ranges[idx];
            emit_tostring_raw(buf, &raw[vs..ve]);
        }
        ResolvedRemap::FieldOpConstToString(idx, ref op, n) => {
            let (vs, ve) = ranges[idx];
            if let Some(a) = parse_json_num(&raw[vs..ve]) {
                let r = match op { BinOp::Add => a + n, BinOp::Sub => a - n, BinOp::Mul => a * n, BinOp::Div => a / n, BinOp::Mod => a % n, _ => unreachable!() };
                buf.push(b'"');
                push_jq_number_bytes(buf, r);
                buf.push(b'"');
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::Arith(ref arith) => {
            if let Some(r) = eval_arith_raw(arith, raw, ranges) {
                push_jq_number_bytes(buf, r);
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::FieldMinMax(idx1, idx2, is_max) => {
            let (vs1, ve1) = ranges[idx1];
            let (vs2, ve2) = ranges[idx2];
            if let (Some(a), Some(b)) = (parse_json_num(&raw[vs1..ve1]), parse_json_num(&raw[vs2..ve2])) {
                let r = if is_max { if a >= b { a } else { b } } else { if a <= b { a } else { b } };
                push_jq_number_bytes(buf, r);
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::LiteralJson(ref bytes) => {
            buf.extend_from_slice(bytes);
        }
        ResolvedRemap::StringInterp(ref parts) => {
            buf.push(b'"');
            for part in parts {
                match part {
                    ResolvedInterpPart::Literal(ref bytes) => {
                        buf.extend_from_slice(bytes);
                    }
                    ResolvedInterpPart::Field(idx) => {
                        let (vs, ve) = ranges[*idx];
                        emit_interp_field_raw(buf, &raw[vs..ve]);
                    }
                }
            }
            buf.push(b'"');
        }
        ResolvedRemap::FieldSplitJoin(idx, ref sep, ref rep) => {
            let (vs, ve) = ranges[idx];
            emit_split_join_raw(buf, &raw[vs..ve], sep, rep);
        }
        ResolvedRemap::FieldStringCase(idx, is_upper) => {
            let (vs, ve) = ranges[idx];
            let val = &raw[vs..ve];
            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"' {
                buf.push(b'"');
                for &b in &val[1..val.len()-1] {
                    if is_upper {
                        buf.push(b.to_ascii_uppercase());
                    } else {
                        buf.push(b.to_ascii_lowercase());
                    }
                }
                buf.push(b'"');
            } else {
                buf.extend_from_slice(b"null");
            }
        }
        ResolvedRemap::FieldSplitLength(idx, ref sep) => {
            let (vs, ve) = ranges[idx];
            let val = &raw[vs..ve];
            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                && !sep.is_empty()
            {
                let inner = &val[1..val.len()-1];
                let count = if sep.len() == 1 {
                    memchr::memchr_iter(sep[0], inner).count() + 1
                } else {
                    inner.windows(sep.len()).filter(|w| *w == &sep[..]).count() + 1
                };
                let mut ibuf = itoa::Buffer::new();
                buf.extend_from_slice(ibuf.format(count).as_bytes());
            } else if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"' && sep.is_empty() {
                // split("") → count codepoints
                let inner = &val[1..val.len()-1];
                if !inner.contains(&b'\\') {
                    let cp_count = inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
                    let mut ibuf = itoa::Buffer::new();
                    buf.extend_from_slice(ibuf.format(cp_count).as_bytes());
                } else {
                    buf.extend_from_slice(b"null");
                }
            } else {
                buf.extend_from_slice(b"null");
            }
        }
        ResolvedRemap::CondChain(ref branches, ref else_out) => {
            let mut matched = false;
            for b in branches {
                let (vs, ve) = ranges[b.cond_field_idx];
                let field_bytes = &raw[vs..ve];
                let cond_result = match &b.cond_rhs {
                    ResolvedCondRhs::Null => {
                        let is_null = field_bytes.starts_with(b"null");
                        match b.cond_op {
                            jq_jit::ir::BinOp::Eq => is_null,
                            jq_jit::ir::BinOp::Ne => !is_null,
                            _ => false,
                        }
                    }
                    ResolvedCondRhs::Bool(expected) => {
                        let is_true = field_bytes.starts_with(b"true");
                        let is_false = field_bytes.starts_with(b"false");
                        let val_bool = if is_true { Some(true) } else if is_false { Some(false) } else { None };
                        if let Some(val) = val_bool {
                            match b.cond_op {
                                jq_jit::ir::BinOp::Eq => val == *expected,
                                jq_jit::ir::BinOp::Ne => val != *expected,
                                _ => false,
                            }
                        } else { false }
                    }
                    ResolvedCondRhs::Str(ref rhs_bytes) => {
                        // Compare raw JSON bytes (field value includes quotes)
                        match b.cond_op {
                            jq_jit::ir::BinOp::Eq => field_bytes == rhs_bytes.as_slice(),
                            jq_jit::ir::BinOp::Ne => field_bytes != rhs_bytes.as_slice(),
                            _ => {
                                // String ordering: compare inner content
                                if field_bytes.first() == Some(&b'"') && rhs_bytes.first() == Some(&b'"') {
                                    let cmp = field_bytes.cmp(rhs_bytes);
                                    match b.cond_op {
                                        jq_jit::ir::BinOp::Gt => cmp == std::cmp::Ordering::Greater,
                                        jq_jit::ir::BinOp::Lt => cmp == std::cmp::Ordering::Less,
                                        jq_jit::ir::BinOp::Ge => cmp != std::cmp::Ordering::Less,
                                        jq_jit::ir::BinOp::Le => cmp != std::cmp::Ordering::Greater,
                                        _ => false,
                                    }
                                } else { false }
                            }
                        }
                    }
                    ResolvedCondRhs::Startswith(ref pat) => {
                        field_bytes.len() >= 2 && field_bytes[0] == b'"'
                            && field_bytes[1..field_bytes.len()-1].starts_with(pat)
                    }
                    ResolvedCondRhs::Endswith(ref pat) => {
                        field_bytes.len() >= 2 && field_bytes[0] == b'"'
                            && field_bytes[1..field_bytes.len()-1].ends_with(pat)
                    }
                    ResolvedCondRhs::Contains(ref pat) => {
                        field_bytes.len() >= 2 && field_bytes[0] == b'"' && {
                            let inner = &field_bytes[1..field_bytes.len()-1];
                            inner.windows(pat.len()).any(|w| w == pat.as_slice())
                        }
                    }
                    ResolvedCondRhs::Const(_) | ResolvedCondRhs::Field(_) => {
                        // Numeric comparison (original path)
                        if let Some(mut val) = parse_json_num(field_bytes) {
                            for &(ref aop, n) in &b.cond_arith_ops {
                                val = match aop {
                                    jq_jit::ir::BinOp::Add => val + n,
                                    jq_jit::ir::BinOp::Sub => val - n,
                                    jq_jit::ir::BinOp::Mul => val * n,
                                    jq_jit::ir::BinOp::Div => val / n,
                                    jq_jit::ir::BinOp::Mod => val % n,
                                    _ => val,
                                };
                            }
                            let rhs_val = match &b.cond_rhs {
                                ResolvedCondRhs::Const(n) => Some(*n),
                                ResolvedCondRhs::Field(idx) => {
                                    let (rs, re) = ranges[*idx];
                                    parse_json_num(&raw[rs..re])
                                }
                                _ => unreachable!(),
                            };
                            if let Some(rhs) = rhs_val {
                                match b.cond_op {
                                    jq_jit::ir::BinOp::Gt => val > rhs,
                                    jq_jit::ir::BinOp::Lt => val < rhs,
                                    jq_jit::ir::BinOp::Ge => val >= rhs,
                                    jq_jit::ir::BinOp::Le => val <= rhs,
                                    jq_jit::ir::BinOp::Eq => val == rhs,
                                    jq_jit::ir::BinOp::Ne => val != rhs,
                                    _ => false,
                                }
                            } else { false }
                        } else { false }
                    }
                };
                if cond_result {
                    emit_resolved_branch_output(buf, &b.output, raw, ranges);
                    matched = true;
                    break;
                }
            }
            if !matched {
                emit_resolved_branch_output(buf, else_out, raw, ranges);
            }
        }
        ResolvedRemap::FieldLength(idx) => {
            let (vs, ve) = ranges[idx];
            let val = &raw[vs..ve];
            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                && !val[1..val.len()-1].contains(&b'\\')
            {
                // ASCII-only fast path: count UTF-8 start bytes
                let content = &val[1..val.len()-1];
                let count = content.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
                push_jq_number_bytes(buf, count as f64);
            } else if let Some(len) = json_value_length(val, 0) {
                push_jq_number_bytes(buf, len as f64);
            } else if val[0] == b'"' {
                // String with escapes: count by unescaping
                let content = &val[1..val.len()-1];
                let mut count = 0usize;
                let mut i = 0;
                while i < content.len() {
                    if content[i] == b'\\' {
                        i += 1;
                        if i < content.len() {
                            if content[i] == b'u' {
                                // \uXXXX — possibly a surrogate pair
                                i += 4;
                                if i + 2 <= content.len() && content[i] == b'\\' && i + 1 < content.len() && content[i+1] == b'u' {
                                    // Check for surrogate pair
                                    i += 6; // skip \uXXXX
                                }
                            } else { i += 1; }
                        }
                        count += 1;
                    } else {
                        if (content[i] & 0xC0) != 0x80 { count += 1; }
                        i += 1;
                    }
                }
                push_jq_number_bytes(buf, count as f64);
            } else if let Some(n) = parse_json_num(val) {
                push_jq_number_bytes(buf, n.abs());
            } else {
                buf.extend_from_slice(b"null");
            }
        }
        ResolvedRemap::FieldStrBuiltin(idx, ref op, ref arg) => {
            let (vs, ve) = ranges[idx];
            let val = &raw[vs..ve];
            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"' {
                let inner = &val[1..val.len()-1];
                match op {
                    jq_jit::interpreter::StrBuiltin::Ltrimstr => {
                        if inner.starts_with(&arg[..]) {
                            buf.push(b'"');
                            buf.extend_from_slice(&inner[arg.len()..]);
                            buf.push(b'"');
                        } else {
                            buf.extend_from_slice(val);
                        }
                    }
                    jq_jit::interpreter::StrBuiltin::Rtrimstr => {
                        if inner.ends_with(&arg[..]) {
                            buf.push(b'"');
                            buf.extend_from_slice(&inner[..inner.len()-arg.len()]);
                            buf.push(b'"');
                        } else {
                            buf.extend_from_slice(val);
                        }
                    }
                    jq_jit::interpreter::StrBuiltin::Startswith => {
                        buf.extend_from_slice(if inner.starts_with(&arg[..]) { b"true" } else { b"false" });
                    }
                    jq_jit::interpreter::StrBuiltin::Endswith => {
                        buf.extend_from_slice(if inner.ends_with(&arg[..]) { b"true" } else { b"false" });
                    }
                    jq_jit::interpreter::StrBuiltin::Index => {
                        if arg.len() == 1 {
                            if let Some(pos) = memchr::memchr(arg[0], inner) {
                                let mut ibuf = itoa::Buffer::new();
                                buf.extend_from_slice(ibuf.format(pos).as_bytes());
                            } else {
                                buf.extend_from_slice(b"null");
                            }
                        } else if let Some(pos) = inner.windows(arg.len()).position(|w| w == &arg[..]) {
                            let mut ibuf = itoa::Buffer::new();
                            buf.extend_from_slice(ibuf.format(pos).as_bytes());
                        } else {
                            buf.extend_from_slice(b"null");
                        }
                    }
                    jq_jit::interpreter::StrBuiltin::Contains => {
                        let found = if arg.len() == 1 {
                            memchr::memchr(arg[0], inner).is_some()
                        } else {
                            inner.windows(arg.len()).any(|w| w == &arg[..])
                        };
                        buf.extend_from_slice(if found { b"true" } else { b"false" });
                    }
                }
            } else {
                buf.extend_from_slice(b"null");
            }
        }
        ResolvedRemap::FieldSplitIndex(idx, ref sep, split_idx) => {
            let (vs, ve) = ranges[idx];
            let val = &raw[vs..ve];
            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"' && !sep.is_empty() {
                let inner = &val[1..val.len()-1];
                // Collect split segments
                let mut segments: Vec<&[u8]> = Vec::new();
                if sep.len() == 1 {
                    let sep_byte = sep[0];
                    let mut start = 0;
                    for pos in memchr::memchr_iter(sep_byte, inner) {
                        segments.push(&inner[start..pos]);
                        start = pos + 1;
                    }
                    segments.push(&inner[start..]);
                } else {
                    let mut start = 0;
                    let mut i = 0;
                    while i + sep.len() <= inner.len() {
                        if &inner[i..i+sep.len()] == &sep[..] {
                            segments.push(&inner[start..i]);
                            start = i + sep.len();
                            i = start;
                        } else {
                            i += 1;
                        }
                    }
                    segments.push(&inner[start..]);
                }
                // Resolve index (negative wraps)
                let len = segments.len() as i32;
                let actual_idx = if split_idx < 0 { split_idx + len } else { split_idx };
                if actual_idx >= 0 && actual_idx < len {
                    buf.push(b'"');
                    buf.extend_from_slice(segments[actual_idx as usize]);
                    buf.push(b'"');
                } else {
                    buf.extend_from_slice(b"null");
                }
            } else {
                buf.extend_from_slice(b"null");
            }
        }
        ResolvedRemap::FieldOpFieldToString(idx1, ref op, idx2) => {
            let (vs1, ve1) = ranges[idx1];
            let (vs2, ve2) = ranges[idx2];
            if let (Some(a), Some(b)) = (parse_json_num(&raw[vs1..ve1]), parse_json_num(&raw[vs2..ve2])) {
                let r = match op { BinOp::Add => a + b, BinOp::Sub => a - b, BinOp::Mul => a * b, BinOp::Div => a / b, BinOp::Mod => a % b, _ => unreachable!() };
                buf.push(b'"');
                push_jq_number_bytes(buf, r);
                buf.push(b'"');
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::ArithToString(ref arith) => {
            if let Some(r) = eval_arith_raw(arith, raw, ranges) {
                buf.push(b'"');
                push_jq_number_bytes(buf, r);
                buf.push(b'"');
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::ArithUnary(math_op, ref arith) => {
            if let Some(r) = eval_arith_raw(arith, raw, ranges) {
                let result = apply_math_unary(math_op, r);
                push_jq_number_bytes(buf, result);
            } else { buf.extend_from_slice(b"null"); }
        }
        ResolvedRemap::FieldType(idx) => {
            let (vs, ve) = ranges[idx];
            let val = &raw[vs..ve];
            let type_str = match val[0] {
                b'"' => b"\"string\"" as &[u8],
                b't' | b'f' => b"\"boolean\"",
                b'n' => b"\"null\"",
                b'[' => b"\"array\"",
                b'{' => b"\"object\"",
                _ => b"\"number\"", // digits or minus
            };
            buf.extend_from_slice(type_str);
        }
        ResolvedRemap::FieldNegate(idx) => {
            let (vs, ve) = ranges[idx];
            let val = &raw[vs..ve];
            if let Some(n) = parse_json_num(val) {
                push_jq_number_bytes(buf, -n);
            } else {
                buf.extend_from_slice(b"null");
            }
        }
        ResolvedRemap::ArithCmp(ref arith, cmp_op, rhs_const) => {
            if let Some(lhs_val) = eval_arith_raw(arith, raw, ranges) {
                let result = match cmp_op {
                    jq_jit::ir::BinOp::Gt => lhs_val > rhs_const,
                    jq_jit::ir::BinOp::Lt => lhs_val < rhs_const,
                    jq_jit::ir::BinOp::Ge => lhs_val >= rhs_const,
                    jq_jit::ir::BinOp::Le => lhs_val <= rhs_const,
                    jq_jit::ir::BinOp::Eq => lhs_val == rhs_const,
                    jq_jit::ir::BinOp::Ne => lhs_val != rhs_const,
                    _ => false,
                };
                buf.extend_from_slice(if result { b"true" } else { b"false" });
            } else {
                buf.extend_from_slice(b"false");
            }
        }
        ResolvedRemap::FieldArray(ref elements) => {
            buf.push(b'[');
            for (i, elem) in elements.iter().enumerate() {
                if i > 0 { buf.push(b','); }
                emit_resolved_value(buf, elem, raw, ranges);
            }
            buf.push(b']');
        }
        ResolvedRemap::BoolExpr(ref l, ref bool_op, ref r) => {
            if let (Some(lv), Some(rv)) = (eval_resolved_bool(l, raw, ranges), eval_resolved_bool(r, raw, ranges)) {
                let result = match bool_op {
                    jq_jit::ir::BinOp::And => lv && rv,
                    jq_jit::ir::BinOp::Or => lv || rv,
                    _ => unreachable!(),
                };
                buf.extend_from_slice(if result { b"true" } else { b"false" });
            } else {
                buf.extend_from_slice(b"null");
            }
        }
        ResolvedRemap::FieldSlice(idx, from, to) => {
            let (vs, ve) = ranges[idx];
            let val = &raw[vs..ve];
            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                && !val[1..val.len()-1].contains(&b'\\')
            {
                let inner = &val[1..val.len()-1];
                // Count UTF-8 codepoints to find byte offsets
                let cp_count = inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count() as i64;
                let resolve = |v: i64| -> usize {
                    let v = if v < 0 { (cp_count + v).max(0) } else { v.min(cp_count) } as usize;
                    // Find byte offset for codepoint index v
                    let mut cp = 0usize;
                    for (i, &b) in inner.iter().enumerate() {
                        if cp == v { return i; }
                        if (b & 0xC0) != 0x80 { cp += 1; }
                    }
                    inner.len()
                };
                let start = from.map(|f| resolve(f)).unwrap_or(0);
                let end = to.map(|t| resolve(t)).unwrap_or(inner.len());
                buf.push(b'"');
                if start <= end && start <= inner.len() {
                    buf.extend_from_slice(&inner[start..end.min(inner.len())]);
                }
                buf.push(b'"');
            } else {
                buf.extend_from_slice(b"null");
            }
        }
    }
}

/// Evaluate a ResolvedRemap as a boolean (for BoolExpr and/or).
#[inline]
fn eval_resolved_bool(resolved: &ResolvedRemap, raw: &[u8], ranges: &[(usize, usize)]) -> Option<bool> {
    use jq_jit::ir::BinOp;
    match resolved {
        ResolvedRemap::FieldCmpConst(idx, ref op, n) => {
            let (vs, ve) = ranges[*idx];
            let a = parse_json_num(&raw[vs..ve])?;
            Some(match op { BinOp::Gt => a > *n, BinOp::Lt => a < *n, BinOp::Ge => a >= *n, BinOp::Le => a <= *n, BinOp::Eq => a == *n, BinOp::Ne => a != *n, _ => return None })
        }
        ResolvedRemap::FieldCmpField(idx1, ref op, idx2) => {
            let (vs1, ve1) = ranges[*idx1];
            let (vs2, ve2) = ranges[*idx2];
            let (a, b) = (parse_json_num(&raw[vs1..ve1])?, parse_json_num(&raw[vs2..ve2])?);
            Some(match op { BinOp::Gt => a > b, BinOp::Lt => a < b, BinOp::Ge => a >= b, BinOp::Le => a <= b, BinOp::Eq => a == b, BinOp::Ne => a != b, _ => return None })
        }
        ResolvedRemap::BoolExpr(ref l, ref bool_op, ref r) => {
            let lv = eval_resolved_bool(l, raw, ranges)?;
            let rv = eval_resolved_bool(r, raw, ranges)?;
            Some(match bool_op { BinOp::And => lv && rv, BinOp::Or => lv || rv, _ => return None })
        }
        _ => {
            // Evaluate as JSON and check truthiness
            let mut tmp = Vec::new();
            emit_resolved_value(&mut tmp, resolved, raw, ranges);
            // JSON truthiness: null and false are falsy, everything else is truthy
            if tmp == b"null" || tmp == b"false" { Some(false) } else { Some(true) }
        }
    }
}

#[inline]
fn apply_math_unary(op: jq_jit::interpreter::MathUnary, v: f64) -> f64 {
    use jq_jit::interpreter::MathUnary;
    match op {
        MathUnary::Sqrt => v.sqrt(),
        MathUnary::Floor => v.floor(),
        MathUnary::Ceil => v.ceil(),
        MathUnary::Fabs => v.abs(),
        MathUnary::Round => v.round(),
    }
}

/// Evaluate an ArithExpr against raw byte ranges, parsing field values on demand.
#[inline]
fn eval_arith_raw(
    expr: &jq_jit::interpreter::ArithExpr,
    raw: &[u8],
    ranges: &[(usize, usize)],
) -> Option<f64> {
    use jq_jit::interpreter::ArithExpr;
    use jq_jit::ir::BinOp;
    match expr {
        ArithExpr::Field(idx) => {
            let (vs, ve) = ranges[*idx];
            parse_json_num(&raw[vs..ve])
        }
        ArithExpr::Const(n) => Some(*n),
        ArithExpr::BinOp(op, lhs, rhs) => {
            let l = eval_arith_raw(lhs, raw, ranges)?;
            let r = eval_arith_raw(rhs, raw, ranges)?;
            Some(match op {
                BinOp::Add => l + r,
                BinOp::Sub => l - r,
                BinOp::Mul => l * r,
                BinOp::Div => l / r,
                BinOp::Mod => l % r,
                _ => return None,
            })
        }
        ArithExpr::Unary(op, inner) => {
            let v = eval_arith_raw(inner, raw, ranges)?;
            Some(apply_math_unary(*op, v))
        }
    }
}

fn main() {
    // Run on a thread with a large stack to handle deep recursion.
    // macOS lazily pages the stack, so the physical memory usage is proportional to actual depth.
    let builder = std::thread::Builder::new().stack_size(2048 * 1024 * 1024);
    let handler = builder.spawn(real_main).unwrap();
    let result = handler.join();
    if result.is_err() {
        std::process::exit(134); // SIGABRT-like
    }
}

fn real_main() {
    let args: Vec<String> = std::env::args().collect();

    let mut filter_str = None;
    let mut files: Vec<String> = Vec::new();
    let mut compact = false;
    let mut raw_output = false;
    let mut raw_input = false;
    let mut null_input = false;
    let mut slurp = false;
    let mut join_output = false;
    let mut tab = false;
    let mut indent_n = 2usize;
    let mut sort_keys = false;
    let mut exit_status = false;
    let mut arg_vars: Vec<(String, Value)> = Vec::new();
    let mut argjson_vars: Vec<(String, Value)> = Vec::new();
    let mut lib_dirs: Vec<String> = Vec::new();

    // Expand args: split combined short flags like -ncr into ["-n", "-c", "-r"]
    let mut expanded_args: Vec<String> = Vec::new();
    for arg in &args[1..] {
        if arg.starts_with('-') && !arg.starts_with("--") && arg.len() > 2
            && arg[1..].chars().all(|c| c.is_ascii_alphabetic()) {
            for ch in arg[1..].chars() {
                expanded_args.push(format!("-{}", ch));
            }
        } else {
            expanded_args.push(arg.clone());
        }
    }

    let mut i = 0;
    while i < expanded_args.len() {
        let arg = &expanded_args[i];
        match arg.as_str() {
            "-c" | "--compact-output" => compact = true,
            "-r" | "--raw-output" => raw_output = true,
            "-R" | "--raw-input" => raw_input = true,
            "-n" | "--null-input" => null_input = true,
            "-s" | "--slurp" => slurp = true,
            "-j" | "--join-output" => { join_output = true; raw_output = true; }
            "-S" | "--sort-keys" => sort_keys = true,
            "-e" | "--exit-status" => exit_status = true,
            "--tab" => tab = true,
            "--indent" => {
                i += 1;
                if i < expanded_args.len() {
                    indent_n = expanded_args[i].parse().unwrap_or(2);
                }
            }
            "-f" | "--from-file" => {
                i += 1;
                if i < expanded_args.len() {
                    let path = std::path::Path::new(&expanded_args[i]);
                    let content = match std::fs::read_to_string(path) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("jq: Could not open file {}: {}", expanded_args[i], e);
                            process::exit(2);
                        }
                    };
                    filter_str = Some(content.trim_end().to_string());
                    // Add the filter file's directory to lib search path for import resolution
                    if let Some(parent) = path.canonicalize().ok().and_then(|p| p.parent().map(|d| d.to_string_lossy().into_owned())) {
                        if !lib_dirs.contains(&parent) {
                            lib_dirs.push(parent);
                        }
                    }
                }
            }
            "--arg" => {
                if i + 2 < expanded_args.len() {
                    let name = expanded_args[i + 1].clone();
                    let val = Value::from_str(&expanded_args[i + 2]);
                    arg_vars.push((name, val));
                    i += 2;
                }
            }
            "--argjson" => {
                if i + 2 < expanded_args.len() {
                    let name = expanded_args[i + 1].clone();
                    match json_to_value(&expanded_args[i + 2]) {
                        Ok(val) => argjson_vars.push((name, val)),
                        Err(e) => {
                            eprintln!("jq: Invalid JSON text passed to --argjson: {}", e);
                            process::exit(2);
                        }
                    }
                    i += 2;
                }
            }
            "-L" => {
                i += 1;
                if i < expanded_args.len() {
                    lib_dirs.push(expanded_args[i].clone());
                }
            }
            "--args" => break,
            "--version" => {
                println!("jq-jit-0.1.0");
                process::exit(0);
            }
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            s if s.starts_with('-') && filter_str.is_some() => {
                eprintln!("jq: Unknown option: {}", s);
                process::exit(2);
            }
            _ => {
                if filter_str.is_none() {
                    filter_str = Some(arg.clone());
                } else {
                    files.push(arg.clone());
                }
            }
        }
        i += 1;
    }

    let filter_str = match filter_str {
        Some(f) => f,
        None => {
            eprintln!("jq - commandline JSON processor");
            eprintln!("Usage: jq-jit [OPTIONS] <FILTER> [FILE...]");
            process::exit(2);
        }
    };

    // Prepend --arg / --argjson bindings to the filter expression
    let filter_str = {
        let mut prefix = String::new();
        for (name, val) in &arg_vars {
            prefix.push_str(&value_to_json_precise(val));
            prefix.push_str(" as $");
            prefix.push_str(name);
            prefix.push_str(" | ");
        }
        for (name, val) in &argjson_vars {
            prefix.push_str(&value_to_json_precise(val));
            prefix.push_str(" as $");
            prefix.push_str(name);
            prefix.push_str(" | ");
        }
        if prefix.is_empty() {
            filter_str
        } else {
            prefix.push_str(&filter_str);
            prefix
        }
    };

    // Create filter without JIT initially — JIT is compiled lazily when input is large enough.
    let mut filter = match Filter::with_options(&filter_str, &lib_dirs, false) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("jq: error: {}", e);
            process::exit(3);
        }
    };

    // projection_fields is set below after all pattern detections

    let stdout = io::stdout();
    let mut out = io::BufWriter::with_capacity(65536, stdout.lock());

    let mut any_output_false = false;

    let format_value = |v: &Value| -> String {
        if raw_output {
            if let Value::Str(s) = v {
                return s.to_string();
            }
        }
        if compact {
            value_to_json_precise(v)
        } else if tab {
            value_to_json_pretty_ext(v, 0, 1, true, sort_keys)
        } else {
            value_to_json_pretty_ext(v, 0, indent_n, false, sort_keys)
        }
    };

    let mut had_error = false;

    // Pre-read stdin so we can estimate input size for JIT decision.
    let stdin_data: Option<String> = if !null_input && files.is_empty() && !raw_input {
        let mut s = String::new();
        io::stdin().lock().read_to_string(&mut s).unwrap_or(0);
        Some(s)
    } else {
        None
    };

    // Lazy JIT: compile only when input is large enough to amortize compilation cost.
    // Exception: always JIT for loop constructs (reduce/foreach/while/until/recurse)
    // since their runtime dominates regardless of input size.
    // Must be done before process_input closure captures &filter.
    const JIT_THRESHOLD: usize = 4096;
    if filter.has_loop_constructs() || null_input {
        filter.compile_jit();
    } else if !null_input {
        if files.is_empty() {
            if raw_input && !slurp {
                filter.compile_jit();
            } else if let Some(ref data) = stdin_data {
                if data.len() >= JIT_THRESHOLD {
                    filter.compile_jit();
                }
            }
        } else {
            // File input: check first file size
            if let Ok(meta) = std::fs::metadata(&files[0]) {
                if meta.len() as usize >= JIT_THRESHOLD {
                    filter.compile_jit();
                }
            }
        }
    }

    // Use Vec-based buffering for compact output to avoid per-value write_all overhead
    let use_compact_buf = compact && !raw_output && !sort_keys && !join_output;
    let use_pretty_buf = !compact && !raw_output && !sort_keys && !join_output && !tab;
    // Helper macro: emit raw JSON bytes to buffer with trailing newline,
    // handling compact vs pretty output.
    macro_rules! emit_raw_ln {
        ($buf:expr, $raw:expr) => {
            if use_pretty_buf {
                push_json_pretty_raw($buf, $raw, 2, false);
                $buf.push(b'\n');
            } else if is_json_compact($raw) {
                $buf.extend_from_slice($raw);
                $buf.push(b'\n');
            } else {
                push_json_compact_raw($buf, $raw);
                $buf.push(b'\n');
            }
        };
    }
    let field_access = if (use_compact_buf || use_pretty_buf) && !exit_status {
        filter.detect_field_access()
    } else { None };
    let nested_field = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() {
        filter.detect_nested_field_access()
    } else { None };
    let field_remap = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && nested_field.is_none() {
        filter.detect_field_remap()
    } else { None };
    let computed_remap = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && nested_field.is_none() && field_remap.is_none() {
        filter.detect_computed_remap()
    } else { None };
    let standalone_array = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_remap.is_none() && computed_remap.is_none() {
        filter.detect_standalone_array()
    } else { None };
    let field_binop = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_remap.is_none() && computed_remap.is_none() {
        filter.detect_field_binop()
    } else { None };
    let field_unary_num = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_binop.is_none() {
        filter.detect_field_unary_num()
    } else { None };
    let field_binop_const_unary = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_binop.is_none() && field_unary_num.is_none() {
        filter.detect_field_binop_const_unary()
    } else { None };
    let field_arith_chain = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_binop.is_none() && field_binop_const_unary.is_none() {
        filter.detect_field_arith_chain()
    } else { None };
    let field_arith_tostring = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_arith_chain.is_none() {
        filter.detect_field_arith_chain_tostring()
    } else { None };
    let field_binop_tostring = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_binop.is_none() && field_arith_tostring.is_none() {
        filter.detect_field_binop_tostring()
    } else { None };
    let numeric_expr = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_binop.is_none() && field_binop_const_unary.is_none() && field_arith_chain.is_none() {
        filter.detect_numeric_expr()
    } else { None };
    let numeric_expr_unary = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && numeric_expr.is_none() && field_binop.is_none() && field_arith_chain.is_none() {
        filter.detect_numeric_expr_unary()
    } else { None };
    let field_field_cmp = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_binop.is_none() {
        filter.detect_field_field_cmp()
    } else { None };
    let field_const_cmp = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_binop.is_none() && field_field_cmp.is_none() {
        filter.detect_field_const_cmp()
    } else { None };
    let arith_chain_cmp = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_const_cmp.is_none() && field_field_cmp.is_none() {
        filter.detect_arith_chain_cmp()
    } else { None };
    let compound_field_cmp = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_const_cmp.is_none() && field_field_cmp.is_none() && arith_chain_cmp.is_none() {
        filter.detect_compound_field_cmp()
    } else { None };
    let field_str_builtin = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_unary_num.is_none() {
        filter.detect_field_str_builtin()
    } else { None };
    let field_test = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_str_builtin.is_none() {
        filter.detect_field_test()
    } else { None };
    let field_gsub = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_str_builtin.is_none() && field_test.is_none() {
        filter.detect_field_gsub()
    } else { None };
    let field_format = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_str_builtin.is_none() && field_test.is_none() && field_gsub.is_none() {
        filter.detect_field_format()
    } else { None };
    let field_ltrimstr_tonumber = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_str_builtin.is_none() && field_test.is_none() {
        filter.detect_field_ltrimstr_tonumber()
    } else { None };
    let field_str_concat = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_remap.is_none() && field_binop.is_none() {
        filter.detect_field_str_concat()
    } else { None };
    let select_cmp = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_remap.is_none() && field_binop.is_none() && field_str_concat.is_none() {
        filter.detect_select_field_cmp()
    } else { None };
    let select_arith_cmp = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && field_access.is_none() {
        filter.detect_select_arith_cmp()
    } else { None };
    let select_str = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && field_access.is_none() {
        filter.detect_select_field_str()
    } else { None };
    let select_str_test = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_str.is_none() && field_access.is_none() {
        filter.detect_select_field_str_test()
    } else { None };
    let select_regex_test = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_str.is_none() && select_str_test.is_none() && field_access.is_none() {
        filter.detect_select_field_regex_test()
    } else { None };
    let select_nested_cmp = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_str.is_none() && select_str_test.is_none() {
        filter.detect_select_nested_cmp()
    } else { None };
    let computed_array = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && computed_remap.is_none() {
        filter.detect_computed_array()
    } else { None };
    let array_field = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && computed_array.is_none() {
        filter.detect_array_field_access()
    } else { None };
    let multi_field = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && array_field.is_none() {
        filter.detect_multi_field_access()
    } else { None };
    let is_length = (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && multi_field.is_none() && select_cmp.is_none() && filter.is_length();
    let is_keys = (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && select_cmp.is_none() && !is_length && filter.is_keys();
    let is_keys_unsorted = (use_compact_buf || use_pretty_buf) && !exit_status && !is_keys && !is_length && filter.is_keys_unsorted();
    let has_field = if (use_compact_buf || use_pretty_buf) && !exit_status && !is_length && !is_keys {
        filter.detect_has_field()
    } else { None };
    let has_multi = if (use_compact_buf || use_pretty_buf) && !exit_status && !is_length && !is_keys && has_field.is_none() {
        filter.detect_has_multi_field()
    } else { None };
    let is_type = (use_compact_buf || use_pretty_buf) && !exit_status && !is_length && !is_keys && has_field.is_none() && has_multi.is_none() && filter.is_type();
    let del_field = if (use_compact_buf || use_pretty_buf) && !exit_status && !is_length && !is_keys && !is_type && has_field.is_none() {
        filter.detect_del_field()
    } else { None };
    let obj_merge_lit = if (use_compact_buf || use_pretty_buf) && !exit_status && del_field.is_none() {
        filter.detect_obj_merge_literal()
    } else { None };
    let obj_merge_computed = if (use_compact_buf || use_pretty_buf) && !exit_status && obj_merge_lit.is_none() && del_field.is_none() {
        filter.detect_obj_merge_computed()
    } else { None };
    let is_each = (use_compact_buf || use_pretty_buf) && !exit_status && !is_length && !is_keys && !is_type && has_field.is_none() && del_field.is_none() && field_access.is_none() && filter.is_each();
    let is_sort_keys = (use_compact_buf || use_pretty_buf) && !exit_status && !is_each && filter.is_sort_keys();
    let is_to_entries = (use_compact_buf || use_pretty_buf) && !exit_status && !is_each && !is_sort_keys && filter.is_to_entries();
    let remap_to_entries = if (use_compact_buf || use_pretty_buf) && !exit_status && !is_to_entries && field_remap.is_none() {
        filter.detect_remap_to_entries()
    } else { None };
    let with_entries_select = if (use_compact_buf || use_pretty_buf) && !exit_status && !is_to_entries && remap_to_entries.is_none() {
        filter.detect_with_entries_select_value_cmp()
    } else { None };
    let with_entries_type = if (use_compact_buf || use_pretty_buf) && !exit_status && !is_to_entries && remap_to_entries.is_none() && with_entries_select.is_none() {
        filter.detect_with_entries_select_value_type()
    } else { None };
    let is_tojson = (use_compact_buf || use_pretty_buf) && !exit_status && !is_each && !is_to_entries && remap_to_entries.is_none() && with_entries_select.is_none() && with_entries_type.is_none() && filter.is_tojson();
    let string_interp_fields = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_remap.is_none() && field_binop.is_none() && field_str_concat.is_none() {
        filter.detect_string_interp_fields()
    } else { None };
    let string_add_chain = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && string_interp_fields.is_none() && field_str_concat.is_none() {
        filter.detect_string_add_chain()
    } else { None };
    let array_join = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() {
        filter.detect_array_join()
    } else { None };
    let literal_output = if (use_compact_buf || use_pretty_buf) && !exit_status { filter.detect_literal_output() } else { None };
    let array_fields_format = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() {
        filter.detect_array_fields_format()
    } else { None };
    // For -r mode: raw CSV/TSV output bypasses JSON encoding entirely
    let raw_csv_fields = if raw_output && !exit_status && !slurp && !join_output && array_fields_format.is_none() {
        filter.detect_array_fields_format()
    } else { None };
    let field_str_reverse = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() {
        filter.detect_field_str_reverse()
    } else { None };
    let field_split_join = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_str_reverse.is_none() {
        filter.detect_field_split_join()
    } else { None };
    let field_split_first = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_split_join.is_none() {
        filter.detect_field_split_first()
    } else { None };
    let field_split_last = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_split_join.is_none() && field_split_first.is_none() {
        filter.detect_field_split_last()
    } else { None };
    let field_split_nth = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_split_join.is_none() && field_split_first.is_none() && field_split_last.is_none() {
        filter.detect_field_split_index()
    } else { None };
    let field_slice = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() {
        filter.detect_field_slice()
    } else { None };
    let dynamic_key_obj = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && computed_remap.is_none() {
        filter.detect_dynamic_key_obj()
    } else { None };
    let field_update_num = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && computed_remap.is_none() && dynamic_key_obj.is_none() {
        filter.detect_field_update_num()
    } else { None };
    let field_assign_const = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_update_num.is_none() {
        filter.detect_field_assign_const()
    } else { None };
    let field_split_length = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() {
        filter.detect_field_split_length()
    } else { None };
    let field_strop_length = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_split_length.is_none() {
        filter.detect_field_strop_length()
    } else { None };
    let field_length_cmp = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_strop_length.is_none() {
        filter.detect_field_length_cmp()
    } else { None };
    let select_length_cmp_field = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_length_cmp.is_none() {
        filter.detect_select_field_length_cmp_then_field()
    } else { None };
    let min_two_fields = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() {
        filter.detect_min_two_fields()
    } else { None };
    let minmax_two = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && min_two_fields.is_none() {
        filter.detect_minmax_two_fields()
    } else { None };
    let minmax_n = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && min_two_fields.is_none() && minmax_two.is_none() {
        filter.detect_minmax_n_fields()
    } else { None };
    let field_alt = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() {
        filter.detect_field_alternative()
    } else { None };
    let field_field_alt = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_alt.is_none() {
        filter.detect_field_field_alternative()
    } else { None };
    let cond_chain = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && field_access.is_none() {
        filter.detect_cond_chain()
    } else { None };
    let cmp_branch_lit = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && field_access.is_none() && cond_chain.is_none() {
        filter.detect_cmp_branch_literals()
    } else { None };
    let arith_cmp_branch_lit = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && field_access.is_none() && cond_chain.is_none() && cmp_branch_lit.is_none() {
        filter.detect_arith_cmp_branch_literals()
    } else { None };
    let field_field_cmp_branch = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && field_access.is_none() && cond_chain.is_none() && cmp_branch_lit.is_none() && arith_cmp_branch_lit.is_none() {
        filter.detect_field_field_cmp_branch()
    } else { None };
    let if_cmp_arrays = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && cond_chain.is_none() && cmp_branch_lit.is_none() && field_field_cmp_branch.is_none() {
        filter.detect_if_cmp_then_arrays()
    } else { None };
    let select_compound = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && field_access.is_none() && cmp_branch_lit.is_none() && arith_cmp_branch_lit.is_none() && cond_chain.is_none() {
        filter.detect_select_compound_cmp()
    } else { None };
    let select_compound_field = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_compound.is_none() && field_access.is_none() {
        filter.detect_select_compound_cmp_then_field()
    } else { None };
    let select_compound_remap = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_compound.is_none() && select_compound_field.is_none() && field_access.is_none() {
        filter.detect_select_compound_cmp_then_remap()
    } else { None };
    let select_has_multi = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && field_access.is_none() {
        filter.detect_select_has_multi()
    } else { None };
    let select_cmp_field = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && field_access.is_none() {
        filter.detect_select_cmp_then_field()
    } else { None };
    let select_arith_cmp_field = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_cmp_field.is_none() && field_access.is_none() {
        filter.detect_select_arith_cmp_then_field()
    } else { None };
    let select_cmp_field_unary = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_cmp_field.is_none() && select_arith_cmp_field.is_none() && field_access.is_none() {
        filter.detect_select_cmp_then_field_unary()
    } else { None };
    let select_cmp_remap = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_cmp_field.is_none() && field_access.is_none() {
        filter.detect_select_cmp_then_remap()
    } else { None };
    let select_cmp_cremap = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_cmp_field.is_none() && select_cmp_remap.is_none() && field_access.is_none() {
        filter.detect_select_cmp_then_computed_remap()
    } else { None };
    let select_cmp_array = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_cmp_field.is_none() && select_cmp_remap.is_none() && select_cmp_cremap.is_none() && field_access.is_none() {
        filter.detect_select_cmp_then_array()
    } else { None };
    let select_arith_cmp_array = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_cmp_field.is_none() && select_cmp_array.is_none() && field_access.is_none() {
        filter.detect_select_arith_cmp_then_array()
    } else { None };
    let select_cmp_value = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_cmp_field.is_none() && select_cmp_remap.is_none() && select_cmp_cremap.is_none() && select_cmp_array.is_none() && select_arith_cmp_array.is_none() && field_access.is_none() {
        filter.detect_select_cmp_then_value()
    } else { None };
    let select_ff_cmp_field = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_cmp_field.is_none() && field_access.is_none() {
        filter.detect_select_field_cmp_field_then_field()
    } else { None };
    let select_ff_cmp = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_ff_cmp_field.is_none() && field_access.is_none() {
        filter.detect_select_field_field_cmp()
    } else { None };
    let select_ff_cmp_cremap = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_ff_cmp.is_none() && select_ff_cmp_field.is_none() && field_access.is_none() {
        filter.detect_select_ff_cmp_then_computed_remap()
    } else { None };
    let select_ff_cmp_value = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_ff_cmp.is_none() && select_ff_cmp_field.is_none() && select_ff_cmp_cremap.is_none() && field_access.is_none() {
        filter.detect_select_ff_cmp_then_value()
    } else { None };
    let select_ff_cmp_array = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_ff_cmp.is_none() && select_ff_cmp_field.is_none() && select_ff_cmp_cremap.is_none() && select_ff_cmp_value.is_none() && field_access.is_none() {
        filter.detect_select_ff_cmp_then_array()
    } else { None };
    let select_compound_array = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_compound.is_none() && select_compound_field.is_none() && select_compound_remap.is_none() && field_access.is_none() {
        filter.detect_select_compound_cmp_then_array()
    } else { None };
    let select_str_field = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_cmp_field.is_none() && field_access.is_none() {
        filter.detect_select_str_then_field()
    } else { None };
    let field_string_chain = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && field_str_builtin.is_none() {
        filter.detect_field_string_chain()
    } else { None };
    let remap_tostring_join = if (use_compact_buf || use_pretty_buf) && !exit_status && field_access.is_none() && array_join.is_none() {
        filter.detect_remap_tostring_join()
    } else { None };
    let select_str_cremap = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_str_field.is_none() && field_access.is_none() {
        filter.detect_select_str_then_computed_remap()
    } else { None };
    let select_str_array = if (use_compact_buf || use_pretty_buf) && !exit_status && select_cmp.is_none() && select_str_field.is_none() && select_str_cremap.is_none() && field_access.is_none() {
        filter.detect_select_str_then_array()
    } else { None };
    // Field projection: if filter only accesses specific fields, skip parsing the rest.
    // Only activate when no raw byte fast path matched (those handle their own parsing).
    let has_raw_fast_path = field_access.is_some() || nested_field.is_some() || field_remap.is_some()
        || computed_remap.is_some() || standalone_array.is_some()
        || field_binop.is_some() || field_binop_tostring.is_some() || field_unary_num.is_some() || field_binop_const_unary.is_some() || field_arith_chain.is_some() || field_arith_tostring.is_some() || numeric_expr.is_some() || numeric_expr_unary.is_some()
        || field_field_cmp.is_some() || field_const_cmp.is_some() || arith_chain_cmp.is_some() || compound_field_cmp.is_some()
        || field_str_builtin.is_some() || field_test.is_some() || field_gsub.is_some() || field_format.is_some() || field_ltrimstr_tonumber.is_some()
        || field_str_concat.is_some() || field_alt.is_some() || field_field_alt.is_some()
        || select_cmp.is_some() || select_arith_cmp.is_some()
        || cond_chain.is_some() || cmp_branch_lit.is_some() || arith_cmp_branch_lit.is_some() || field_field_cmp_branch.is_some() || if_cmp_arrays.is_some() || select_compound.is_some() || select_compound_field.is_some() || select_compound_remap.is_some()
        || select_str.is_some()
        || select_str_test.is_some() || select_regex_test.is_some() || select_nested_cmp.is_some()
        || select_cmp_field.is_some() || select_arith_cmp_field.is_some() || select_cmp_field_unary.is_some() || select_cmp_remap.is_some() || select_cmp_cremap.is_some() || select_cmp_array.is_some() || select_arith_cmp_array.is_some() || select_cmp_value.is_some() || select_ff_cmp_field.is_some() || select_ff_cmp.is_some() || select_ff_cmp_cremap.is_some() || select_ff_cmp_value.is_some() || select_ff_cmp_array.is_some() || select_compound_array.is_some() || select_str_field.is_some() || select_str_cremap.is_some() || select_str_array.is_some()
        || computed_array.is_some() || array_field.is_some() || multi_field.is_some() || is_length || is_keys
        || is_keys_unsorted || has_field.is_some() || has_multi.is_some() || select_has_multi.is_some() || is_type || del_field.is_some() || obj_merge_lit.is_some() || obj_merge_computed.is_some()
        || is_each || is_sort_keys || is_to_entries || remap_to_entries.is_some() || with_entries_select.is_some() || with_entries_type.is_some() || is_tojson || string_interp_fields.is_some() || string_add_chain.is_some() || array_join.is_some()
        || literal_output.is_some() || array_fields_format.is_some() || raw_csv_fields.is_some()
        || field_str_reverse.is_some() || field_split_join.is_some() || field_split_first.is_some() || field_split_last.is_some() || field_split_nth.is_some() || field_split_length.is_some() || field_strop_length.is_some() || field_length_cmp.is_some() || select_length_cmp_field.is_some() || field_slice.is_some()
        || dynamic_key_obj.is_some() || field_update_num.is_some() || field_assign_const.is_some()
        || min_two_fields.is_some() || minmax_two.is_some() || minmax_n.is_some() || field_string_chain.is_some() || remap_tostring_join.is_some() || filter.is_empty();
    let projection_fields: Option<Vec<String>> = if !has_raw_fast_path && !slurp && !raw_input {
        filter.needed_input_fields()
    } else { None };
    let mut compact_buf: Vec<u8> = if use_compact_buf || use_pretty_buf || raw_csv_fields.is_some() { Vec::with_capacity(1 << 17) } else { Vec::new() };
    let process_input = |input: &Value, raw_bytes: Option<&[u8]>, out: &mut io::BufWriter<io::StdoutLock>, cbuf: &mut Vec<u8>, any_false: &mut bool, had_error: &mut bool| {
        let result = filter.execute_cb(input, &mut |result| {
            if let Value::Error(e) = result {
                eprintln!("jq: error: {}", e.as_str());
                *had_error = true;
                return Ok(true);
            }
            if exit_status && !result.is_true() {
                *any_false = true;
            }
            if use_compact_buf {
                // Raw passthrough: if result is the unmodified input and bytes are compact,
                // copy original bytes directly instead of re-serializing
                if let Some(raw) = raw_bytes {
                    if std::ptr::eq(result, input) && is_json_compact(raw) {
                        cbuf.extend_from_slice(raw);
                        cbuf.push(b'\n');
                        if cbuf.len() >= 1 << 17 {
                            let _ = out.write_all(cbuf);
                            cbuf.clear();
                        }
                        return Ok(true);
                    }
                }
                push_compact_line(cbuf, result);
                if cbuf.len() >= 1 << 17 {
                    let _ = out.write_all(cbuf);
                    cbuf.clear();
                }
            } else if use_pretty_buf {
                push_pretty_line(cbuf, result, indent_n, tab);
                if cbuf.len() >= 1 << 17 {
                    let _ = out.write_all(cbuf);
                    cbuf.clear();
                }
            } else if join_output {
                if compact && !raw_output {
                    let _ = write_value_compact_ext(out, result, sort_keys);
                } else {
                    let formatted = format_value(result);
                    let _ = write!(out, "{}", formatted);
                }
            } else if compact && !raw_output {
                let _ = write_value_compact_line(out, result, sort_keys);
            } else if !raw_output {
                let _ = write_value_pretty_line(out, result, indent_n, tab, sort_keys);
            } else {
                let formatted = format_value(result);
                let _ = writeln!(out, "{}", formatted);
            }
            Ok(true)
        });
        if let Err(e) = result {
            let msg = format!("{}", e);
            if let Some(jq_msg) = msg.strip_prefix("__jqerror__:") {
                eprintln!("jq: error: {}", jq_msg);
            } else {
                eprintln!("jq: error: {}", msg);
            }
            *had_error = true;
        }
    };

    if null_input {
        // Pre-read inputs for `input`/`inputs` builtins
        if filter.uses_inputs() {
            let mut inputs_values = Vec::new();
            if files.is_empty() {
                // Read from stdin
                let mut input_str = String::new();
                io::stdin().lock().read_to_string(&mut input_str).unwrap_or(0);
                if raw_input {
                    for line in input_str.lines() {
                        inputs_values.push(Value::from_str(line));
                    }
                } else if let Err(e) = json_stream(&input_str, |v| {
                    inputs_values.push(v);
                    Ok(())
                }) {
                    eprintln!("jq: error (at <stdin>:0): {}", e);
                    process::exit(2);
                }
            } else {
                // Read from files
                for file in &files {
                    let content = match std::fs::read_to_string(file) {
                        Ok(c) => c,
                        Err(e) => { eprintln!("jq: error: Could not open file {}: {}", file, e); process::exit(2); }
                    };
                    if raw_input {
                        for line in content.lines() {
                            inputs_values.push(Value::from_str(line));
                        }
                    } else if let Err(e) = json_stream(&content, |v| {
                        inputs_values.push(v);
                        Ok(())
                    }) {
                        eprintln!("jq: error (at {}:0): {}", file, e);
                        process::exit(2);
                    }
                }
            }
            jq_jit::eval::set_inputs_queue(inputs_values);
        }
        process_input(&Value::Null, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
        jq_jit::eval::clear_inputs_queue();
    } else if files.is_empty() {
        // Read from stdin
        let stdin = io::stdin();
        if raw_input {
            let mut lines: Vec<Value> = Vec::new();
            for line in stdin.lock().lines() {
                match line {
                    Ok(l) => {
                        if slurp {
                            lines.push(Value::from_str(&l));
                        } else {
                            process_input(&Value::from_str(&l), None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    }
                    Err(e) => {
                        eprintln!("jq: error reading input: {}", e);
                        process::exit(2);
                    }
                }
            }
            if slurp {
                let arr = Value::Arr(std::rc::Rc::new(lines));
                process_input(&arr, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
            }
        } else {
            // stdin_data was pre-read above for JIT size estimation
            let input_str = stdin_data.unwrap_or_default();

            if slurp {
                // Parse all JSON values and collect into array
                let mut values = Vec::new();
                if let Err(e) = json_stream(&input_str, |v| {
                    values.push(v);
                    Ok(())
                }) {
                    eprintln!("jq: error (at <stdin>:0): {}", e);
                    process::exit(2);
                }
                let arr = Value::Arr(std::rc::Rc::new(values));
                process_input(&arr, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
            } else {
                let input_bytes = input_str.as_bytes();
                let parse_result = if filter.is_empty() {
                    json_stream_raw(&input_str, |_, _| Ok(()))
                } else if let Some(ref lit) = literal_output {
                    json_stream_raw(&input_str, |_, _| {
                        compact_buf.extend_from_slice(lit);
                        compact_buf.push(b'\n');
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref aff_fields, ref aff_format)) = array_fields_format {
                    // [.f1,.f2,...] | @csv or @tsv — raw byte field extract + format
                    let aff_refs: Vec<&str> = aff_fields.iter().map(|s| s.as_str()).collect();
                    let is_csv = aff_format == "csv";
                    let mut ranges_buf = vec![(0usize, 0usize); aff_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &aff_refs, &mut ranges_buf) {
                            // Check all string fields for escape sequences — fall back if any
                            let mut has_escapes = false;
                            for (vs, ve) in &ranges_buf {
                                let val = &raw[*vs..*ve];
                                if val[0] == b'"' && val[1..val.len()-1].contains(&b'\\') {
                                    has_escapes = true;
                                    break;
                                }
                            }
                            if has_escapes {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            } else {
                                compact_buf.push(b'"');
                                for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                    if i > 0 {
                                        if is_csv { compact_buf.push(b','); }
                                        else { compact_buf.extend_from_slice(b"\\t"); }
                                    }
                                    let val = &raw[*vs..*ve];
                                    if val[0] == b'"' {
                                        // Simple string (no escapes): content is raw UTF-8
                                        let inner = &val[1..val.len()-1];
                                        if is_csv {
                                            // CSV: wrap in \"...\"
                                            compact_buf.extend_from_slice(b"\\\"");
                                            compact_buf.extend_from_slice(inner);
                                            compact_buf.extend_from_slice(b"\\\"");
                                        } else {
                                            // TSV: output raw string content
                                            compact_buf.extend_from_slice(inner);
                                        }
                                    } else if val == b"null" {
                                        // null → empty
                                    } else {
                                        // number, boolean — output as-is
                                        compact_buf.extend_from_slice(val);
                                    }
                                }
                                compact_buf.push(b'"');
                                compact_buf.push(b'\n');
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref rcf_fields, ref rcf_format)) = raw_csv_fields {
                    // [.f1,.f2,...] | @csv/@tsv with -r flag — direct raw CSV/TSV output
                    let rcf_refs: Vec<&str> = rcf_fields.iter().map(|s| s.as_str()).collect();
                    let is_csv = rcf_format == "csv";
                    let sep = if is_csv { b',' } else { b'\t' };
                    let mut ranges_buf = vec![(0usize, 0usize); rcf_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &rcf_refs, &mut ranges_buf) {
                            for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                if i > 0 { compact_buf.push(sep); }
                                let val = &raw[*vs..*ve];
                                if val[0] == b'"' {
                                    let inner = &val[1..val.len()-1];
                                    if is_csv {
                                        compact_buf.push(b'"');
                                        if inner.contains(&b'\\') {
                                            // Has JSON escapes — decode first
                                            let decoded = json_unescape_bytes(inner);
                                            // CSV-escape: double any quotes
                                            for &b in &decoded {
                                                if b == b'"' { compact_buf.push(b'"'); }
                                                compact_buf.push(b);
                                            }
                                        } else if inner.contains(&b'"') {
                                            for &b in inner.iter() {
                                                if b == b'"' { compact_buf.push(b'"'); }
                                                compact_buf.push(b);
                                            }
                                        } else {
                                            compact_buf.extend_from_slice(inner);
                                        }
                                        compact_buf.push(b'"');
                                    } else {
                                        // TSV: output raw
                                        if inner.contains(&b'\\') {
                                            compact_buf.extend_from_slice(&json_unescape_bytes(inner));
                                        } else {
                                            compact_buf.extend_from_slice(inner);
                                        }
                                    }
                                } else if val == b"null" {
                                    // null → empty
                                } else if val == b"true" {
                                    compact_buf.extend_from_slice(b"true");
                                } else if val == b"false" {
                                    compact_buf.extend_from_slice(b"false");
                                } else {
                                    // number
                                    compact_buf.extend_from_slice(val);
                                }
                            }
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref rev_field) = field_str_reverse {
                    // .field | split("") | reverse | join("") — string reversal
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, rev_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"' {
                                let inner = &val[1..val.len()-1];
                                // Fast path: ASCII-only with no escapes
                                if !inner.contains(&b'\\') && inner.iter().all(|&b| b < 0x80) {
                                    compact_buf.push(b'"');
                                    for &b in inner.iter().rev() {
                                        compact_buf.push(b);
                                    }
                                    compact_buf.extend_from_slice(b"\"\n");
                                } else {
                                    // Decode JSON string, reverse chars, re-encode
                                    let unescaped = json_unescape_bytes(inner);
                                    if let Ok(s) = std::str::from_utf8(&unescaped) {
                                        let reversed: String = s.chars().rev().collect();
                                        compact_buf.push(b'"');
                                        compact_buf.extend_from_slice(&json_escape_bytes(reversed.as_bytes()));
                                        compact_buf.extend_from_slice(b"\"\n");
                                    } else {
                                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                    }
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref fsj_field, ref fsj_split, ref fsj_join)) = field_split_join {
                    // .field | split("x") | join("y") — raw byte string replace
                    let split_bytes = fsj_split.as_bytes();
                    let single_split = if split_bytes.len() == 1 { Some(split_bytes[0]) } else { None };
                    // Pre-escape join string for JSON output
                    let escaped_join = json_escape_bytes(fsj_join.as_bytes());
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, fsj_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && memchr::memchr(b'\\', &val[1..val.len()-1]).is_none()
                                && !split_bytes.is_empty()
                            {
                                let inner = &val[1..val.len()-1];
                                compact_buf.push(b'"');
                                // Simple string replace: split by X, join by Y
                                let mut pos = 0;
                                let mut first = true;
                                loop {
                                    let rest = &inner[pos..];
                                    let found = if let Some(d) = single_split {
                                        memchr::memchr(d, rest)
                                    } else {
                                        rest.windows(split_bytes.len()).position(|w| w == split_bytes)
                                    };
                                    if let Some(idx) = found {
                                        if !first { compact_buf.extend_from_slice(&escaped_join); }
                                        first = false;
                                        compact_buf.extend_from_slice(&rest[..idx]);
                                        pos += idx + split_bytes.len();
                                    } else {
                                        if !first { compact_buf.extend_from_slice(&escaped_join); }
                                        compact_buf.extend_from_slice(rest);
                                        break;
                                    }
                                }
                                compact_buf.push(b'"');
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sf_field, ref sf_delim)) = field_split_first {
                    // .field | split("s") | .[0] — extract first split segment from raw bytes
                    let delim_bytes = sf_delim.as_bytes();
                    let single_delim = if delim_bytes.len() == 1 { Some(delim_bytes[0]) } else { None };
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sf_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && memchr::memchr(b'\\', &val[1..val.len()-1]).is_none()
                                && !delim_bytes.is_empty()
                            {
                                let inner = &val[1..val.len()-1];
                                compact_buf.push(b'"');
                                let split_pos = if let Some(d) = single_delim {
                                    memchr::memchr(d, inner)
                                } else {
                                    inner.windows(delim_bytes.len()).position(|w| w == delim_bytes)
                                };
                                if let Some(idx) = split_pos {
                                    compact_buf.extend_from_slice(&inner[..idx]);
                                } else {
                                    compact_buf.extend_from_slice(inner);
                                }
                                compact_buf.push(b'"');
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sl_field, ref sl_delim)) = field_split_last {
                    // .field | split("s") | last — extract last split segment from raw bytes
                    let delim_bytes = sl_delim.as_bytes();
                    let single_delim = if delim_bytes.len() == 1 { Some(delim_bytes[0]) } else { None };
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sl_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && memchr::memchr(b'\\', &val[1..val.len()-1]).is_none()
                                && !delim_bytes.is_empty()
                            {
                                let inner = &val[1..val.len()-1];
                                compact_buf.push(b'"');
                                let split_pos = if let Some(d) = single_delim {
                                    memchr::memrchr(d, inner)
                                } else {
                                    // Find last occurrence
                                    let mut last_pos = None;
                                    let mut pos = 0;
                                    while pos + delim_bytes.len() <= inner.len() {
                                        if &inner[pos..pos+delim_bytes.len()] == delim_bytes {
                                            last_pos = Some(pos);
                                        }
                                        pos += 1;
                                    }
                                    last_pos
                                };
                                if let Some(idx) = split_pos {
                                    compact_buf.extend_from_slice(&inner[idx + delim_bytes.len()..]);
                                } else {
                                    compact_buf.extend_from_slice(inner);
                                }
                                compact_buf.push(b'"');
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sn_field, ref sn_delim, sn_idx)) = field_split_nth {
                    // .field | split("s") | .[N] — extract Nth split segment from raw bytes
                    let delim_bytes = sn_delim.as_bytes();
                    let single_delim = if delim_bytes.len() == 1 { Some(delim_bytes[0]) } else { None };
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sn_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && memchr::memchr(b'\\', &val[1..val.len()-1]).is_none()
                                && !delim_bytes.is_empty()
                            {
                                let inner = &val[1..val.len()-1];
                                // Split and collect segments
                                let segments: Vec<&[u8]> = if let Some(d) = single_delim {
                                    inner.split(|&b| b == d).collect()
                                } else {
                                    let mut segs = Vec::new();
                                    let mut pos = 0;
                                    while pos <= inner.len() {
                                        if let Some(p) = inner[pos..].windows(delim_bytes.len()).position(|w| w == delim_bytes) {
                                            segs.push(&inner[pos..pos+p]);
                                            pos = pos + p + delim_bytes.len();
                                        } else {
                                            segs.push(&inner[pos..]);
                                            break;
                                        }
                                    }
                                    segs
                                };
                                let actual_idx = if sn_idx >= 0 {
                                    sn_idx as usize
                                } else {
                                    let abs = (-sn_idx) as usize;
                                    if abs <= segments.len() { segments.len() - abs } else { segments.len() }
                                };
                                if actual_idx < segments.len() {
                                    compact_buf.push(b'"');
                                    compact_buf.extend_from_slice(segments[actual_idx]);
                                    compact_buf.push(b'"');
                                    compact_buf.push(b'\n');
                                } else {
                                    compact_buf.extend_from_slice(b"null\n");
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref spl_field, ref spl_delim)) = field_split_length {
                    // .field | split("s") | length — count delimiter occurrences + 1
                    let delim_bytes = spl_delim.as_bytes();
                    let single_delim = if delim_bytes.len() == 1 { Some(delim_bytes[0]) } else { None };
                    let mut ibuf = itoa::Buffer::new();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, spl_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && memchr::memchr(b'\\', &val[1..val.len()-1]).is_none()
                                && !delim_bytes.is_empty()
                            {
                                let inner = &val[1..val.len()-1];
                                let count = if let Some(d) = single_delim {
                                    memchr::memchr_iter(d, inner).count() + 1
                                } else {
                                    inner.windows(delim_bytes.len()).filter(|w| *w == delim_bytes).count() + 1
                                };
                                compact_buf.extend_from_slice(ibuf.format(count).as_bytes());
                                compact_buf.push(b'\n');
                            } else if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"' && delim_bytes.is_empty() {
                                // split("") gives array of single chars — length = char count
                                let inner = &val[1..val.len()-1];
                                // Count UTF-8 codepoints (handles escapes approximately — fall back for escaped strings)
                                if !inner.contains(&b'\\') {
                                    let cp_count = inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
                                    compact_buf.extend_from_slice(ibuf.format(cp_count).as_bytes());
                                    compact_buf.push(b'\n');
                                } else {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sol_field, ref sol_op, ref sol_arg)) = field_strop_length {
                    // .field | str_op | length — compute length from raw bytes
                    let mut ibuf = itoa::Buffer::new();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sol_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && !val[1..val.len()-1].contains(&b'\\')
                            {
                                let inner = &val[1..val.len()-1];
                                let result = match sol_op.as_str() {
                                    "ltrimstr" => {
                                        let prefix = sol_arg.as_ref().unwrap().as_bytes();
                                        if inner.starts_with(prefix) {
                                            // Count UTF-8 codepoints in remaining bytes
                                            let remaining = &inner[prefix.len()..];
                                            remaining.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                        } else {
                                            inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                        }
                                    }
                                    "rtrimstr" => {
                                        let suffix = sol_arg.as_ref().unwrap().as_bytes();
                                        if inner.ends_with(suffix) {
                                            let remaining = &inner[..inner.len() - suffix.len()];
                                            remaining.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                        } else {
                                            inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                        }
                                    }
                                    "identity_length" => {
                                        // ascii_downcase/upcase don't change length
                                        inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                    }
                                    "explode" => {
                                        // explode | length = codepoint count
                                        inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                    }
                                    _ => {
                                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                        return Ok(());
                                    }
                                };
                                compact_buf.extend_from_slice(ibuf.format(result).as_bytes());
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref flc_field, flc_op, flc_n)) = field_length_cmp {
                    // .field | length cmp N — string length comparison from raw bytes
                    use jq_jit::ir::BinOp;
                    let threshold = flc_n as usize;
                    let threshold_f = flc_n;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, flc_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && !val[1..val.len()-1].contains(&b'\\')
                            {
                                let inner = &val[1..val.len()-1];
                                let cp_count = inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
                                let result = match flc_op {
                                    BinOp::Gt => cp_count as f64 > threshold_f,
                                    BinOp::Lt => (cp_count as f64) < threshold_f,
                                    BinOp::Ge => cp_count as f64 >= threshold_f,
                                    BinOp::Le => cp_count as f64 <= threshold_f,
                                    BinOp::Eq => cp_count == threshold,
                                    BinOp::Ne => cp_count != threshold,
                                    _ => unreachable!(),
                                };
                                if result {
                                    compact_buf.extend_from_slice(b"true\n");
                                } else {
                                    compact_buf.extend_from_slice(b"false\n");
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref slcf_cond, slcf_op, slcf_n, ref slcf_out)) = select_length_cmp_field {
                    // select(.field | length cmp N) | .output_field
                    use jq_jit::ir::BinOp;
                    let threshold = slcf_n as usize;
                    let threshold_f = slcf_n;
                    let fields: Vec<&str> = if slcf_cond == slcf_out {
                        vec![slcf_cond.as_str()]
                    } else {
                        vec![slcf_cond.as_str(), slcf_out.as_str()]
                    };
                    let mut ranges_buf = vec![(0usize, 0usize); fields.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &fields, &mut ranges_buf) {
                            let cond_range = ranges_buf[0];
                            let cond_val = &raw[cond_range.0..cond_range.1];
                            if cond_val.len() >= 2 && cond_val[0] == b'"' && cond_val[cond_val.len()-1] == b'"'
                                && !cond_val[1..cond_val.len()-1].contains(&b'\\')
                            {
                                let inner = &cond_val[1..cond_val.len()-1];
                                let cp_count = inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
                                let pass = match slcf_op {
                                    BinOp::Gt => cp_count as f64 > threshold_f,
                                    BinOp::Lt => (cp_count as f64) < threshold_f,
                                    BinOp::Ge => cp_count as f64 >= threshold_f,
                                    BinOp::Le => cp_count as f64 <= threshold_f,
                                    BinOp::Eq => cp_count == threshold,
                                    BinOp::Ne => cp_count != threshold,
                                    _ => false,
                                };
                                if pass {
                                    let out_range = if fields.len() == 1 { ranges_buf[0] } else { ranges_buf[1] };
                                    let out_val = &raw[out_range.0..out_range.1];
                                    if use_pretty_buf && (out_val[0] == b'{' || out_val[0] == b'[') {
                                        push_json_pretty_raw(&mut compact_buf, out_val, 2, false);
                                    } else {
                                        compact_buf.extend_from_slice(out_val);
                                    }
                                    compact_buf.push(b'\n');
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref m2_f1, ref m2_f2)) = min_two_fields {
                    // [.x, .y] | sort | .[0] — min of two numeric fields
                    let fields: Vec<&str> = vec![m2_f1.as_str(), m2_f2.as_str()];
                    let mut ranges_buf = vec![(0usize, 0usize); 2];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &fields, &mut ranges_buf) {
                            if let (Some(a), Some(b)) = (
                                parse_json_num(&raw[ranges_buf[0].0..ranges_buf[0].1]),
                                parse_json_num(&raw[ranges_buf[1].0..ranges_buf[1].1]),
                            ) {
                                let r = if a <= b { a } else { b };
                                push_jq_number_bytes(&mut compact_buf, r);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref mm_f1, ref mm_f2, mm_is_max)) = minmax_two {
                    // [.x, .y] | max or min — raw byte fast path
                    let fields: Vec<&str> = vec![mm_f1.as_str(), mm_f2.as_str()];
                    let mut ranges_buf = vec![(0usize, 0usize); 2];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &fields, &mut ranges_buf) {
                            if let (Some(a), Some(b)) = (
                                parse_json_num(&raw[ranges_buf[0].0..ranges_buf[0].1]),
                                parse_json_num(&raw[ranges_buf[1].0..ranges_buf[1].1]),
                            ) {
                                let r = if mm_is_max { if a >= b { a } else { b } } else { if a <= b { a } else { b } };
                                push_jq_number_bytes(&mut compact_buf, r);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref mm_fields, mm_is_max)) = minmax_n {
                    // [.f1, .f2, .f3, ...] | min/max — N-field raw byte fast path
                    let field_refs: Vec<&str> = mm_fields.iter().map(|s| s.as_str()).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            let mut result: Option<f64> = None;
                            let mut all_ok = true;
                            for &(vs, ve) in ranges_buf.iter() {
                                if let Some(v) = parse_json_num(&raw[vs..ve]) {
                                    result = Some(match result {
                                        None => v,
                                        Some(cur) => if mm_is_max { if v > cur { v } else { cur } } else { if v < cur { v } else { cur } },
                                    });
                                } else {
                                    all_ok = false;
                                    break;
                                }
                            }
                            if all_ok {
                                if let Some(r) = result {
                                    push_jq_number_bytes(&mut compact_buf, r);
                                    compact_buf.push(b'\n');
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sl_field, sl_from, sl_to)) = field_slice {
                    // .field[from:to] — raw byte string slice
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sl_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && val[1..val.len()-1].is_ascii()
                                && !val[1..val.len()-1].contains(&b'\\')
                            {
                                let inner = &val[1..val.len()-1];
                                let len = inner.len() as i64;
                                let f = match sl_from {
                                    Some(v) => if v < 0 { (len + v).max(0) as usize } else { (v as usize).min(inner.len()) },
                                    None => 0,
                                };
                                let t = match sl_to {
                                    Some(v) => if v < 0 { (len + v).max(0) as usize } else { (v as usize).min(inner.len()) },
                                    None => inner.len(),
                                };
                                compact_buf.push(b'"');
                                if t > f { compact_buf.extend_from_slice(&inner[f..t]); }
                                compact_buf.push(b'"');
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref dk_key, ref dk_val)) = dynamic_key_obj {
                    // {(.key_field): .val_field} — extract both fields, build single-key object
                    let fields: Vec<&str> = vec![dk_key.as_str(), dk_val.as_str()];
                    let mut ranges_buf = vec![(0usize, 0usize); fields.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &fields, &mut ranges_buf) {
                            let (ks, ke) = ranges_buf[0];
                            let (vs, ve) = ranges_buf[1];
                            let key_val = &raw[ks..ke];
                            // Key must be a string
                            if key_val.len() >= 2 && key_val[0] == b'"' {
                                if use_pretty_buf {
                                    compact_buf.extend_from_slice(b"{\n  ");
                                    compact_buf.extend_from_slice(key_val);
                                    compact_buf.extend_from_slice(b": ");
                                    let val = &raw[vs..ve];
                                    if val[0] == b'{' || val[0] == b'[' {
                                        push_json_pretty_raw_at(&mut compact_buf, val, 2, false, 1);
                                    } else {
                                        compact_buf.extend_from_slice(val);
                                    }
                                    compact_buf.extend_from_slice(b"\n}\n");
                                } else {
                                    compact_buf.push(b'{');
                                    compact_buf.extend_from_slice(key_val);
                                    compact_buf.push(b':');
                                    compact_buf.extend_from_slice(&raw[vs..ve]);
                                    compact_buf.extend_from_slice(b"}\n");
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref fu_field, ref fu_op, fu_n)) = field_update_num {
                    let mut tmp = Vec::with_capacity(256);
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if use_pretty_buf {
                            tmp.clear();
                            if json_object_update_field_num(raw, 0, fu_field, *fu_op, fu_n, &mut tmp) {
                                push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else if !json_object_update_field_num(raw, 0, fu_field, *fu_op, fu_n, &mut compact_buf) {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        } else {
                            compact_buf.push(b'\n');
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref fa_field, ref fa_val)) = field_assign_const {
                    let mut tmp = Vec::with_capacity(256);
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if use_pretty_buf {
                            tmp.clear();
                            if json_object_set_field_raw(raw, 0, fa_field, fa_val, &mut tmp) {
                                let len = tmp.len();
                                if len > 0 && tmp[len-1] == b'\n' { tmp.truncate(len-1); }
                                push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else if !json_object_set_field_raw(raw, 0, fa_field, fa_val, &mut compact_buf) {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if filter.is_identity() && use_compact_buf && !exit_status {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if is_json_compact(raw) {
                            compact_buf.extend_from_slice(raw);
                            compact_buf.push(b'\n');
                        } else {
                            push_json_compact_raw(&mut compact_buf, raw);
                            compact_buf.push(b'\n');
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if filter.is_identity() && use_pretty_buf && !exit_status {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        push_json_pretty_raw(&mut compact_buf, raw, indent_n, tab);
                        compact_buf.push(b'\n');
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref fa_field) = field_access {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if raw[0] == b'{' {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, fa_field) {
                                let val_bytes = &raw[vs..ve];
                                emit_raw_ln!(&mut compact_buf, val_bytes);
                            } else {
                                compact_buf.extend_from_slice(b"null\n");
                            }
                        } else {
                            compact_buf.extend_from_slice(b"null\n");
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref nf) = nested_field {
                    let nf_refs: Vec<&str> = nf.iter().map(|s| s.as_str()).collect();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_nested_field_raw(raw, 0, &nf_refs) {
                            let val_bytes = &raw[vs..ve];
                            emit_raw_ln!(&mut compact_buf, val_bytes);
                        } else {
                            compact_buf.extend_from_slice(b"null\n");
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref celems) = computed_array {
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    for rexpr in celems {
                        let names = remap_expr_fields(rexpr);
                        for name in names {
                            if !field_idx.contains_key(name) {
                                field_idx.insert(name.to_string(), all_fields.len());
                                all_fields.push(name.to_string());
                            }
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved = resolve_remap_exprs_array(celems, &field_idx);
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            if use_pretty_buf {
                                compact_buf.extend_from_slice(b"[\n");
                                for (i, res) in resolved.iter().enumerate() {
                                    if i > 0 { compact_buf.extend_from_slice(b",\n"); }
                                    compact_buf.extend_from_slice(b"  ");
                                    emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                }
                                compact_buf.extend_from_slice(b"\n]\n");
                            } else {
                                compact_buf.push(b'[');
                                for (i, res) in resolved.iter().enumerate() {
                                    if i > 0 { compact_buf.push(b','); }
                                    emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                }
                                compact_buf.extend_from_slice(b"]\n");
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref af) = array_field {
                    let af_refs: Vec<&str> = af.iter().map(|s| s.as_str()).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); af_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &af_refs, &mut ranges_buf) {
                            if use_pretty_buf {
                                compact_buf.extend_from_slice(b"[\n");
                                for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                    if i > 0 { compact_buf.extend_from_slice(b",\n"); }
                                    compact_buf.extend_from_slice(b"  ");
                                    let val = &raw[*vs..*ve];
                                    if val[0] == b'{' || val[0] == b'[' {
                                        push_json_pretty_raw_at(&mut compact_buf, val, 2, false, 1);
                                    } else {
                                        compact_buf.extend_from_slice(val);
                                    }
                                }
                                compact_buf.extend_from_slice(b"\n]\n");
                            } else {
                                compact_buf.push(b'[');
                                for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                    if i > 0 { compact_buf.push(b','); }
                                    compact_buf.extend_from_slice(&raw[*vs..*ve]);
                                }
                                compact_buf.extend_from_slice(b"]\n");
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref mf) = multi_field {
                    let mf_refs: Vec<&str> = mf.iter().map(|s| s.as_str()).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); mf_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &mf_refs, &mut ranges_buf) {
                            for (vs, ve) in &ranges_buf {
                                let val_bytes = &raw[*vs..*ve];
                                emit_raw_ln!(&mut compact_buf, val_bytes);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref remap) = field_remap {
                    let input_fields: Vec<&str> = remap.iter().map(|(_, f)| f.as_str()).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); input_fields.len()];
                    if use_pretty_buf {
                        // Pretty output: {
                        //   "key": value,
                        //   ...
                        // }
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            if json_object_get_fields_raw_buf(raw, 0, &input_fields, &mut ranges_buf) {
                                compact_buf.extend_from_slice(b"{\n");
                                for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                    if i > 0 { compact_buf.extend_from_slice(b",\n"); }
                                    compact_buf.extend_from_slice(b"  \"");
                                    compact_buf.extend_from_slice(remap[i].0.as_bytes());
                                    compact_buf.extend_from_slice(b"\": ");
                                    let val = &raw[*vs..*ve];
                                    if val[0] == b'{' || val[0] == b'[' {
                                        push_json_pretty_raw_at(&mut compact_buf, val, 2, false, 1);
                                    } else {
                                        compact_buf.extend_from_slice(val);
                                    }
                                }
                                compact_buf.extend_from_slice(b"\n}\n");
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                            Ok(())
                        })
                    } else {
                        let mut key_prefixes: Vec<Vec<u8>> = Vec::with_capacity(remap.len());
                        for (i, (out_key, _)) in remap.iter().enumerate() {
                            let mut prefix = Vec::new();
                            if i == 0 { prefix.push(b'{'); } else { prefix.push(b','); }
                            prefix.push(b'"');
                            prefix.extend_from_slice(out_key.as_bytes());
                            prefix.extend_from_slice(b"\":");
                            key_prefixes.push(prefix);
                        }
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            if json_object_get_fields_raw_buf(raw, 0, &input_fields, &mut ranges_buf) {
                                for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                    compact_buf.extend_from_slice(&key_prefixes[i]);
                                    compact_buf.extend_from_slice(&raw[*vs..*ve]);
                                }
                                compact_buf.extend_from_slice(b"}\n");
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                            Ok(())
                        })
                    }
                } else if let Some(ref cremap) = computed_remap {
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    for (_, rexpr) in cremap {
                        for name in remap_expr_fields(rexpr) {
                            if !field_idx.contains_key(name) {
                                field_idx.insert(name.to_string(), all_fields.len());
                                all_fields.push(name.to_string());
                            }
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved = resolve_remap_exprs(cremap, &field_idx);
                    let key_prefixes = if use_pretty_buf {
                        build_obj_key_prefixes_pretty(cremap.iter().map(|(k, _)| k.as_str()))
                    } else {
                        build_obj_key_prefixes(cremap.iter().map(|(k, _)| k.as_str()))
                    };
                    let obj_close: &[u8] = if use_pretty_buf { b"\n}\n" } else { b"}\n" };
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            for (i, res) in resolved.iter().enumerate() {
                                compact_buf.extend_from_slice(&key_prefixes[i]);
                                emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                            }
                            compact_buf.extend_from_slice(obj_close);
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref arr_elems) = standalone_array {
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    for rexpr in arr_elems {
                        for name in remap_expr_fields(rexpr) {
                            if !field_idx.contains_key(name) {
                                field_idx.insert(name.to_string(), all_fields.len());
                                all_fields.push(name.to_string());
                            }
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                        .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                        .collect();
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            compact_buf.push(b'[');
                            for (i, res) in resolved.iter().enumerate() {
                                if i > 0 { compact_buf.push(b','); }
                                emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                            }
                            compact_buf.extend_from_slice(b"]\n");
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref f1, ref op, ref f2)) = field_binop {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((a, b)) = json_object_get_two_nums(raw, 0, f1, f2) {
                            let result = match op {
                                BinOp::Add => a + b,
                                BinOp::Sub => a - b,
                                BinOp::Mul => a * b,
                                BinOp::Div => a / b,
                                BinOp::Mod => a % b,
                                _ => unreachable!(),
                            };
                            if result.is_finite() {
                                push_jq_number_bytes(&mut compact_buf, result);
                                compact_buf.push(b'\n');
                            } else {
                                // Division by zero or mod edge case: fall back to normal path for error handling
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref f1, ref op, ref f2)) = field_binop_tostring {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((a, b)) = json_object_get_two_nums(raw, 0, f1, f2) {
                            let result = match op {
                                BinOp::Add => a + b,
                                BinOp::Sub => a - b,
                                BinOp::Mul => a * b,
                                BinOp::Div => a / b,
                                BinOp::Mod => a % b,
                                _ => unreachable!(),
                            };
                            if result.is_finite() {
                                compact_buf.push(b'"');
                                let i = result as i64;
                                if i as f64 == result {
                                    compact_buf.extend_from_slice(itoa::Buffer::new().format(i).as_bytes());
                                } else {
                                    compact_buf.extend_from_slice(ryu::Buffer::new().format(result).as_bytes());
                                }
                                compact_buf.extend_from_slice(b"\"\n");
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref uop)) = field_unary_num {
                    use jq_jit::ir::UnaryOp;
                    let is_string_op = matches!(uop, UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase);
                    let is_length_op = matches!(uop, UnaryOp::Length | UnaryOp::Utf8ByteLength);
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if is_string_op {
                            // String ops: extract raw field bytes
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
                                let val = &raw[vs..ve];
                                // Only fast-path for quoted strings without backslash escapes
                                if val.len() >= 2 && val[0] == b'"' && !val[1..val.len()-1].contains(&b'\\') {
                                    compact_buf.push(b'"');
                                    for &byte in &val[1..val.len()-1] {
                                        compact_buf.push(match uop {
                                            UnaryOp::AsciiDowncase => if byte >= b'A' && byte <= b'Z' { byte + 32 } else { byte },
                                            UnaryOp::AsciiUpcase => if byte >= b'a' && byte <= b'z' { byte - 32 } else { byte },
                                            _ => unreachable!(),
                                        });
                                    }
                                    compact_buf.push(b'"');
                                    compact_buf.push(b'\n');
                                } else {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else if is_length_op {
                            // Length ops: works on any field type
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
                                let val = &raw[vs..ve];
                                match val[0] {
                                    b'"' => {
                                        // String: count characters or bytes
                                        let inner = &val[1..ve-vs-1];
                                        let has_escape = inner.contains(&b'\\');
                                        if !has_escape {
                                            let len = if matches!(uop, UnaryOp::Utf8ByteLength) {
                                                inner.len()
                                            } else {
                                                // length: count Unicode chars — ASCII fast path
                                                if inner.is_ascii() { inner.len() }
                                                else { unsafe { std::str::from_utf8_unchecked(inner) }.chars().count() }
                                            };
                                            push_jq_number_bytes(&mut compact_buf, len as f64);
                                            compact_buf.push(b'\n');
                                        } else {
                                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                        }
                                    }
                                    b'[' | b'{' => {
                                        // Array/object: count elements using raw byte scanning
                                        if let Some(len) = json_value_length(val, 0) {
                                            push_jq_number_bytes(&mut compact_buf, len as f64);
                                            compact_buf.push(b'\n');
                                        } else {
                                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                        }
                                    }
                                    b'n' => {
                                        // null: length is 0
                                        compact_buf.push(b'0');
                                        compact_buf.push(b'\n');
                                    }
                                    _ => {
                                        // Number: length is abs(number)
                                        if let Some(n) = json_object_get_num(raw, 0, field) {
                                            push_jq_number_bytes(&mut compact_buf, n.abs());
                                        } else {
                                            compact_buf.extend_from_slice(b"null");
                                        }
                                        compact_buf.push(b'\n');
                                    }
                                }
                            } else {
                                // Field not found: .field is null, null | length = 0
                                compact_buf.extend_from_slice(b"0\n");
                            }
                        } else if let Some(n) = json_object_get_num(raw, 0, field) {
                            if matches!(uop, UnaryOp::ToString) {
                                compact_buf.push(b'"');
                                push_jq_number_bytes(&mut compact_buf, n);
                                compact_buf.push(b'"');
                                compact_buf.push(b'\n');
                            } else {
                                let result = match uop {
                                    UnaryOp::Floor => n.floor(),
                                    UnaryOp::Ceil => n.ceil(),
                                    UnaryOp::Sqrt => n.sqrt(),
                                    UnaryOp::Fabs | UnaryOp::Abs => n.abs(),
                                    _ => unreachable!(),
                                };
                                push_jq_number_bytes(&mut compact_buf, result);
                                compact_buf.push(b'\n');
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref bop, cval, ref uop_opt, const_left)) = field_binop_const_unary {
                    use jq_jit::ir::BinOp;
                    use jq_jit::ir::UnaryOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(n) = json_object_get_num(raw, 0, field) {
                            let (a, b) = if const_left { (cval, n) } else { (n, cval) };
                            let mid = match bop {
                                BinOp::Add => a + b,
                                BinOp::Sub => a - b,
                                BinOp::Mul => a * b,
                                BinOp::Div => a / b,
                                BinOp::Mod => a % b,
                                _ => unreachable!(),
                            };
                            let result = if let Some(uop) = uop_opt {
                                match uop {
                                    UnaryOp::Floor => mid.floor(),
                                    UnaryOp::Ceil => mid.ceil(),
                                    UnaryOp::Sqrt => mid.sqrt(),
                                    UnaryOp::Fabs | UnaryOp::Abs => mid.abs(),
                                    _ => unreachable!(),
                                }
                            } else { mid };
                            push_jq_number_bytes(&mut compact_buf, result);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref ops)) = field_arith_chain {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(n) = json_object_get_num(raw, 0, field) {
                            let mut result = n;
                            for &(ref op, c) in ops.iter() {
                                result = match op {
                                    BinOp::Add => result + c,
                                    BinOp::Sub => result - c,
                                    BinOp::Mul => result * c,
                                    BinOp::Div => result / c,
                                    BinOp::Mod => result % c,
                                    _ => unreachable!(),
                                };
                            }
                            push_jq_number_bytes(&mut compact_buf, result);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref ops)) = field_arith_tostring {
                    // .field arith_chain | tostring — arithmetic then format as string
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(n) = json_object_get_num(raw, 0, field) {
                            let mut result = n;
                            for &(ref op, c) in ops.iter() {
                                result = match op {
                                    BinOp::Add => result + c,
                                    BinOp::Sub => result - c,
                                    BinOp::Mul => result * c,
                                    BinOp::Div => result / c,
                                    BinOp::Mod => result % c,
                                    _ => unreachable!(),
                                };
                            }
                            // Output as quoted string
                            compact_buf.push(b'"');
                            let i = result as i64;
                            if i as f64 == result {
                                compact_buf.extend_from_slice(itoa::Buffer::new().format(i).as_bytes());
                            } else {
                                compact_buf.extend_from_slice(ryu::Buffer::new().format(result).as_bytes());
                            }
                            compact_buf.extend_from_slice(b"\"\n");
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref nfields, ref arith)) = numeric_expr {
                    let nf_count = nfields.len();
                    // Hoist allocations outside the hot loop
                    let field_refs: Vec<&str> = nfields.iter().map(|s| s.as_str()).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); nf_count];
                    let mut vals_buf: Vec<f64> = vec![0.0; nf_count];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        let ok = if nf_count == 1 {
                            if let Some(v) = json_object_get_num(raw, 0, &nfields[0]) {
                                vals_buf[0] = v; true
                            } else { false }
                        } else if nf_count == 2 {
                            if let Some((a, b)) = json_object_get_two_nums(raw, 0, &nfields[0], &nfields[1]) {
                                vals_buf[0] = a; vals_buf[1] = b; true
                            } else { false }
                        } else {
                            if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                                let mut all_ok = true;
                                for (i, &(s, e)) in ranges_buf.iter().enumerate() {
                                    match fast_float::parse::<f64, _>(unsafe { std::str::from_utf8_unchecked(&raw[s..e]) }) {
                                        Ok(n) => vals_buf[i] = n,
                                        Err(_) => { all_ok = false; break; }
                                    }
                                }
                                all_ok
                            } else { false }
                        };
                        if ok {
                            let result = arith.eval(&vals_buf);
                            push_jq_number_bytes(&mut compact_buf, result);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref nfields, ref arith, math_op)) = numeric_expr_unary {
                    let nf_count = nfields.len();
                    let field_refs: Vec<&str> = nfields.iter().map(|s| s.as_str()).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); nf_count];
                    let mut vals_buf: Vec<f64> = vec![0.0; nf_count];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        let ok = if nf_count == 1 {
                            if let Some(v) = json_object_get_num(raw, 0, &nfields[0]) {
                                vals_buf[0] = v; true
                            } else { false }
                        } else if nf_count == 2 {
                            if let Some((a, b)) = json_object_get_two_nums(raw, 0, &nfields[0], &nfields[1]) {
                                vals_buf[0] = a; vals_buf[1] = b; true
                            } else { false }
                        } else {
                            if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                                let mut all_ok = true;
                                for (i, &(s, e)) in ranges_buf.iter().enumerate() {
                                    match fast_float::parse::<f64, _>(unsafe { std::str::from_utf8_unchecked(&raw[s..e]) }) {
                                        Ok(n) => vals_buf[i] = n,
                                        Err(_) => { all_ok = false; break; }
                                    }
                                }
                                all_ok
                            } else { false }
                        };
                        if ok {
                            let result = apply_math_unary(math_op, arith.eval(&vals_buf));
                            push_jq_number_bytes(&mut compact_buf, result);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref f1, ref cmp_op, ref f2)) = field_field_cmp {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((n1, n2)) = json_object_get_two_nums(raw, 0, f1, f2) {
                            let result = match cmp_op {
                                BinOp::Gt => n1 > n2, BinOp::Lt => n1 < n2,
                                BinOp::Ge => n1 >= n2, BinOp::Le => n1 <= n2,
                                BinOp::Eq => n1 == n2, BinOp::Ne => n1 != n2,
                                _ => unreachable!(),
                            };
                            compact_buf.extend_from_slice(if result { b"true\n" } else { b"false\n" });
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref cmp_op, cval)) = field_const_cmp {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(n) = json_object_get_num(raw, 0, field) {
                            let result = match cmp_op {
                                BinOp::Gt => n > cval, BinOp::Lt => n < cval,
                                BinOp::Ge => n >= cval, BinOp::Le => n <= cval,
                                BinOp::Eq => n == cval, BinOp::Ne => n != cval,
                                _ => unreachable!(),
                            };
                            compact_buf.extend_from_slice(if result { b"true\n" } else { b"false\n" });
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref arith_ops, ref cmp_op, threshold)) = arith_chain_cmp {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(mut n) = json_object_get_num(raw, 0, field) {
                            for (aop, val) in arith_ops {
                                n = match aop {
                                    BinOp::Add => n + val, BinOp::Sub => n - val,
                                    BinOp::Mul => n * val, BinOp::Div => n / val,
                                    BinOp::Mod => n % val, _ => n,
                                };
                            }
                            let result = match cmp_op {
                                BinOp::Gt => n > threshold, BinOp::Lt => n < threshold,
                                BinOp::Ge => n >= threshold, BinOp::Le => n <= threshold,
                                BinOp::Eq => n == threshold, BinOp::Ne => n != threshold,
                                _ => unreachable!(),
                            };
                            compact_buf.extend_from_slice(if result { b"true\n" } else { b"false\n" });
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref conjunct, ref cmps)) = compound_field_cmp {
                    use jq_jit::ir::BinOp;
                    let is_and = matches!(conjunct, BinOp::And);
                    // Collect unique field names for lookup
                    let mut field_names: Vec<String> = Vec::new();
                    let mut field_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                    for (f, _, _) in cmps {
                        if !field_idx.contains_key(f) {
                            field_idx.insert(f.clone(), field_names.len());
                            field_names.push(f.clone());
                        }
                    }
                    let cmp_spec: Vec<(usize, BinOp, f64)> = cmps.iter().map(|(f, op, n)| (field_idx[f], *op, *n)).collect();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        // Fast path for exactly 2 unique fields
                        let got = if field_names.len() == 2 {
                            json_object_get_two_nums(raw, 0, &field_names[0], &field_names[1])
                                .map(|(a, b)| vec![a, b])
                        } else {
                            let mut vals = vec![f64::NAN; field_names.len()];
                            let mut ok = true;
                            for (i, fname) in field_names.iter().enumerate() {
                                if let Some(n) = json_object_get_num(raw, 0, fname) {
                                    vals[i] = n;
                                } else { ok = false; break; }
                            }
                            if ok { Some(vals) } else { None }
                        };
                        if let Some(vals) = got {
                            let mut result = is_and;
                            for (idx, op, threshold) in &cmp_spec {
                                let v = vals[*idx];
                                let cmp_result = match op {
                                    BinOp::Gt => v > *threshold, BinOp::Lt => v < *threshold,
                                    BinOp::Ge => v >= *threshold, BinOp::Le => v <= *threshold,
                                    BinOp::Eq => v == *threshold, BinOp::Ne => v != *threshold,
                                    _ => unreachable!(),
                                };
                                if is_and {
                                    if !cmp_result { result = false; break; }
                                } else {
                                    if cmp_result { result = true; break; }
                                }
                            }
                            compact_buf.extend_from_slice(if result { b"true\n" } else { b"false\n" });
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sb_field, ref sb_name, ref sb_arg)) = field_str_builtin {
                    let arg_bytes = sb_arg.as_bytes();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sb_field) {
                            let val = &raw[vs..ve];
                            // Only fast-path quoted strings without backslash escapes
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && !val[1..val.len()-1].contains(&b'\\')
                            {
                                let content = &val[1..val.len()-1];
                                match sb_name.as_str() {
                                    "startswith" => {
                                        if content.len() >= arg_bytes.len() && &content[..arg_bytes.len()] == arg_bytes {
                                            compact_buf.extend_from_slice(b"true\n");
                                        } else {
                                            compact_buf.extend_from_slice(b"false\n");
                                        }
                                    }
                                    "endswith" => {
                                        if content.len() >= arg_bytes.len() && &content[content.len()-arg_bytes.len()..] == arg_bytes {
                                            compact_buf.extend_from_slice(b"true\n");
                                        } else {
                                            compact_buf.extend_from_slice(b"false\n");
                                        }
                                    }
                                    "ltrimstr" => {
                                        compact_buf.push(b'"');
                                        if content.len() >= arg_bytes.len() && &content[..arg_bytes.len()] == arg_bytes {
                                            compact_buf.extend_from_slice(&content[arg_bytes.len()..]);
                                        } else {
                                            compact_buf.extend_from_slice(content);
                                        }
                                        compact_buf.push(b'"');
                                        compact_buf.push(b'\n');
                                    }
                                    "rtrimstr" => {
                                        compact_buf.push(b'"');
                                        if content.len() >= arg_bytes.len() && &content[content.len()-arg_bytes.len()..] == arg_bytes {
                                            compact_buf.extend_from_slice(&content[..content.len()-arg_bytes.len()]);
                                        } else {
                                            compact_buf.extend_from_slice(content);
                                        }
                                        compact_buf.push(b'"');
                                        compact_buf.push(b'\n');
                                    }
                                    "split" => {
                                        // Raw byte split: split JSON string content by separator,
                                        // output JSON array directly without Value construction
                                        compact_buf.push(b'[');
                                        if arg_bytes.is_empty() {
                                            // split("") = each byte as separate string
                                            for (j, &byte) in content.iter().enumerate() {
                                                if j > 0 { compact_buf.push(b','); }
                                                compact_buf.push(b'"');
                                                compact_buf.push(byte);
                                                compact_buf.push(b'"');
                                            }
                                        } else {
                                            let mut pos = 0;
                                            let mut first = true;
                                            while pos <= content.len() {
                                                if !first { compact_buf.push(b','); }
                                                first = false;
                                                // Find next occurrence of separator
                                                let next = if pos + arg_bytes.len() <= content.len() {
                                                    content[pos..].windows(arg_bytes.len())
                                                        .position(|w| w == arg_bytes)
                                                        .map(|i| pos + i)
                                                } else { None };
                                                compact_buf.push(b'"');
                                                if let Some(found) = next {
                                                    compact_buf.extend_from_slice(&content[pos..found]);
                                                    compact_buf.push(b'"');
                                                    pos = found + arg_bytes.len();
                                                } else {
                                                    compact_buf.extend_from_slice(&content[pos..]);
                                                    compact_buf.push(b'"');
                                                    break;
                                                }
                                            }
                                        }
                                        compact_buf.extend_from_slice(b"]\n");
                                    }
                                    _ => {
                                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                    }
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref ft_field, ref ft_pattern, ref ft_flags)) = field_test {
                    // Build regex pattern with flags
                    let re_pattern = if let Some(flags) = ft_flags {
                        let mut prefix = String::from("(?");
                        for c in flags.chars() {
                            match c { 'i' | 'm' | 's' => prefix.push(c), _ => {} }
                        }
                        prefix.push(')');
                        prefix.push_str(ft_pattern);
                        prefix
                    } else {
                        ft_pattern.clone()
                    };
                    let re = regex::Regex::new(&re_pattern);
                    if let Ok(re) = re {
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, ft_field) {
                                let val = &raw[vs..ve];
                                if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                    && !val[1..val.len()-1].contains(&b'\\')
                                {
                                    let content = &val[1..val.len()-1];
                                    let content_str = unsafe { std::str::from_utf8_unchecked(content) };
                                    if re.is_match(content_str) {
                                        compact_buf.extend_from_slice(b"true\n");
                                    } else {
                                        compact_buf.extend_from_slice(b"false\n");
                                    }
                                } else {
                                    // Has escape sequences — fallback
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                            Ok(())
                        })
                    } else {
                        // Regex compilation failed — fallback to JIT
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            Ok(())
                        })
                    }
                } else if let Some((ref gs_field, gs_global, ref gs_pattern, ref gs_replacement, ref gs_flags)) = field_gsub {
                    // .field | gsub/sub("pattern"; "replacement") — raw byte regex replacement
                    let re_pattern = if let Some(flags) = gs_flags {
                        let mut prefix = String::from("(?");
                        for c in flags.chars() {
                            match c { 'i' | 'm' | 's' => prefix.push(c), _ => {} }
                        }
                        prefix.push(')');
                        prefix.push_str(gs_pattern);
                        prefix
                    } else {
                        gs_pattern.clone()
                    };
                    if let Ok(re) = regex::Regex::new(&re_pattern) {
                        let repl = gs_replacement.as_str();
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, gs_field) {
                                let val = &raw[vs..ve];
                                if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                    && !val[1..val.len()-1].contains(&b'\\')
                                {
                                    let content = unsafe { std::str::from_utf8_unchecked(&val[1..val.len()-1]) };
                                    let result = if gs_global {
                                        re.replace_all(content, repl)
                                    } else {
                                        re.replace(content, repl)
                                    };
                                    compact_buf.push(b'"');
                                    // Escape the result for JSON
                                    for &b in result.as_bytes() {
                                        match b {
                                            b'"' => compact_buf.extend_from_slice(b"\\\""),
                                            b'\\' => compact_buf.extend_from_slice(b"\\\\"),
                                            b'\n' => compact_buf.extend_from_slice(b"\\n"),
                                            b'\r' => compact_buf.extend_from_slice(b"\\r"),
                                            b'\t' => compact_buf.extend_from_slice(b"\\t"),
                                            c if c < 0x20 => { use std::io::Write; let _ = write!(compact_buf, "\\u{:04x}", c); }
                                            _ => compact_buf.push(b),
                                        }
                                    }
                                    compact_buf.extend_from_slice(b"\"\n");
                                } else {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                            Ok(())
                        })
                    } else {
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            Ok(())
                        })
                    }
                } else if let Some((ref ff_field, ref ff_format)) = field_format {
                    // .field | @base64 / @uri / @html — raw byte format
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, ff_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && !val[1..val.len()-1].contains(&b'\\')
                            {
                                let content = &val[1..val.len()-1];
                                match ff_format.as_str() {
                                    "text" => {
                                        // @text on string: identity — output the JSON string as-is
                                        compact_buf.extend_from_slice(val);
                                        compact_buf.push(b'\n');
                                    }
                                    "json" => {
                                        // @json on string: wrap the JSON string value in extra quotes + escape
                                        push_tojson_raw(&mut compact_buf, val);
                                        compact_buf.push(b'\n');
                                    }
                                    _ => {
                                        compact_buf.push(b'"');
                                        match ff_format.as_str() {
                                            "base64" => base64_encode_to(content, &mut compact_buf),
                                            "uri" => uri_encode_to(content, &mut compact_buf),
                                            "html" => html_encode_to(content, &mut compact_buf),
                                            _ => {}
                                        }
                                        compact_buf.extend_from_slice(b"\"\n");
                                    }
                                }
                            } else {
                                // Non-string field values (numbers, booleans, null)
                                match ff_format.as_str() {
                                    "json" | "text" => {
                                        // @json/@text on non-string: wrap raw bytes in quotes
                                        compact_buf.push(b'"');
                                        compact_buf.extend_from_slice(val);
                                        compact_buf.extend_from_slice(b"\"\n");
                                    }
                                    _ => {
                                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                    }
                                }
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref lt_field, ref lt_prefix)) = field_ltrimstr_tonumber {
                    let prefix_bytes = lt_prefix.as_bytes();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, lt_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && !val[1..val.len()-1].contains(&b'\\')
                            {
                                let content = &val[1..val.len()-1];
                                let num_str = if content.len() >= prefix_bytes.len() && &content[..prefix_bytes.len()] == prefix_bytes {
                                    &content[prefix_bytes.len()..]
                                } else {
                                    content
                                };
                                // Parse the remaining string as a number
                                if let Ok(n) = fast_float::parse::<f64, _>(num_str) {
                                    push_jq_number_bytes(&mut compact_buf, n);
                                    compact_buf.push(b'\n');
                                } else {
                                    // tonumber on non-numeric string → null in jq-jit
                                    compact_buf.extend_from_slice(b"null\n");
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sc_field, ref suffix)) = field_str_concat {
                    let suffix_needs_escape = suffix.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20);
                    let suffix_escaped: Vec<u8> = if suffix_needs_escape {
                        let mut buf = Vec::new();
                        for &b in suffix.as_bytes() {
                            match b {
                                b'"' => buf.extend_from_slice(b"\\\""),
                                b'\\' => buf.extend_from_slice(b"\\\\"),
                                b'\n' => buf.extend_from_slice(b"\\n"),
                                b'\r' => buf.extend_from_slice(b"\\r"),
                                b'\t' => buf.extend_from_slice(b"\\t"),
                                c if c < 0x20 => {}
                                _ => buf.push(b),
                            }
                        }
                        buf
                    } else {
                        suffix.as_bytes().to_vec()
                    };
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if raw[0] == b'{' {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sc_field) {
                                let val = &raw[vs..ve];
                                if val[0] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                    compact_buf.extend_from_slice(&val[..val.len()-1]);
                                    compact_buf.extend_from_slice(&suffix_escaped);
                                    compact_buf.extend_from_slice(b"\"\n");
                                } else {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            } else {
                                compact_buf.push(b'"');
                                compact_buf.extend_from_slice(&suffix_escaped);
                                compact_buf.extend_from_slice(b"\"\n");
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref join_parts, ref join_sep)) = array_join {
                    // Fused [.field, "lit", ...] | join("sep") → raw byte string concatenation
                    let field_names: Vec<&str> = join_parts.iter()
                        .filter(|(is_lit, _)| !*is_lit)
                        .map(|(_, name)| name.as_str())
                        .collect();
                    let sep_bytes = join_sep.as_bytes();
                    // Pre-escape separator and literal parts for JSON string output
                    let escaped_sep = json_escape_bytes(sep_bytes);
                    let escaped_lits: Vec<Option<Vec<u8>>> = join_parts.iter().map(|(is_lit, s)| {
                        if *is_lit { Some(json_escape_bytes(s.as_bytes())) } else { None }
                    }).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); field_names.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if raw[0] != b'{' {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            return Ok(());
                        }
                        if json_object_get_fields_raw_buf(raw, 0, &field_names, &mut ranges_buf) {
                            compact_buf.push(b'"');
                            let mut field_idx = 0;
                            for (i, (is_lit, _)) in join_parts.iter().enumerate() {
                                if i > 0 { compact_buf.extend_from_slice(&escaped_sep); }
                                if *is_lit {
                                    compact_buf.extend_from_slice(escaped_lits[i].as_ref().unwrap());
                                } else {
                                    let (vs, ve) = ranges_buf[field_idx];
                                    field_idx += 1;
                                    let val = &raw[vs..ve];
                                    if val[0] == b'"' && val.len() >= 2 {
                                        compact_buf.extend_from_slice(&val[1..val.len()-1]);
                                    } else {
                                        compact_buf.extend_from_slice(val);
                                    }
                                }
                            }
                            compact_buf.push(b'"');
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref remap_exprs, ref join_sep)) = remap_tostring_join {
                    // [remap_exprs] | map(tostring) | join("sep")
                    let mut all_fields: Vec<String> = Vec::new();
                    for rexpr in remap_exprs {
                        for f in remap_expr_fields(rexpr) {
                            if !all_fields.iter().any(|x| x == f) {
                                all_fields.push(f.to_string());
                            }
                        }
                    }
                    let field_strs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let mut field_idx = std::collections::HashMap::new();
                    for (i, f) in all_fields.iter().enumerate() { field_idx.insert(f.clone(), i); }
                    let resolved: Vec<ResolvedRemap> = remap_exprs.iter().map(|rexpr| resolve_one_remap(rexpr, &field_idx)).collect();
                    let sep_bytes = join_sep.as_bytes();
                    let escaped_sep = json_escape_bytes(sep_bytes);
                    let mut ranges = vec![(0usize, 0usize); all_fields.len()];
                    let mut num_buf = Vec::<u8>::new();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if raw[0] != b'{' {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            return Ok(());
                        }
                        if json_object_get_fields_raw_buf(raw, 0, &field_strs, &mut ranges) {
                            compact_buf.push(b'"');
                            for (i, res) in resolved.iter().enumerate() {
                                if i > 0 { compact_buf.extend_from_slice(&escaped_sep); }
                                // Emit each element as a string (tostring semantics)
                                num_buf.clear();
                                emit_resolved_value(&mut num_buf, res, raw, &ranges);
                                // If it's a JSON string, strip quotes; otherwise use as-is
                                if num_buf.len() >= 2 && num_buf[0] == b'"' && num_buf[num_buf.len()-1] == b'"' {
                                    compact_buf.extend_from_slice(&num_buf[1..num_buf.len()-1]);
                                } else {
                                    // Escape for embedding in JSON string
                                    compact_buf.extend_from_slice(&json_escape_bytes(&num_buf));
                                }
                            }
                            compact_buf.push(b'"');
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref interp_parts) = string_interp_fields {
                    // Collect field names needed for extraction
                    let field_names: Vec<&str> = interp_parts.iter()
                        .filter(|(is_lit, _)| !*is_lit)
                        .map(|(_, name)| name.as_str())
                        .collect();
                    // Pre-escape literal parts for JSON string output
                    let escaped_lits: Vec<Option<Vec<u8>>> = interp_parts.iter().map(|(is_lit, s)| {
                        if *is_lit {
                            let mut buf = Vec::new();
                            for &b in s.as_bytes() {
                                match b {
                                    b'"' => buf.extend_from_slice(b"\\\""),
                                    b'\\' => buf.extend_from_slice(b"\\\\"),
                                    b'\n' => buf.extend_from_slice(b"\\n"),
                                    b'\r' => buf.extend_from_slice(b"\\r"),
                                    b'\t' => buf.extend_from_slice(b"\\t"),
                                    c if c < 0x20 => { let _ = write!(buf, "\\u{:04x}", c); }
                                    _ => buf.push(b),
                                }
                            }
                            Some(buf)
                        } else {
                            None
                        }
                    }).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); field_names.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if raw[0] != b'{' {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            return Ok(());
                        }
                        // Extract all needed fields
                        if json_object_get_fields_raw_buf(raw, 0, &field_names, &mut ranges_buf) {
                            compact_buf.push(b'"');
                            let mut field_idx = 0;
                            for (i, (is_lit, _)) in interp_parts.iter().enumerate() {
                                if *is_lit {
                                    compact_buf.extend_from_slice(escaped_lits[i].as_ref().unwrap());
                                } else {
                                    let (vs, ve) = ranges_buf[field_idx];
                                    field_idx += 1;
                                    let val = &raw[vs..ve];
                                    if val[0] == b'"' && val.len() >= 2 {
                                        // String: copy inner content (already JSON-escaped)
                                        compact_buf.extend_from_slice(&val[1..val.len()-1]);
                                    } else {
                                        // Number/bool/null: copy as-is (jq tostring behavior)
                                        compact_buf.extend_from_slice(val);
                                    }
                                }
                            }
                            compact_buf.push(b'"');
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref sac_parts) = string_add_chain {
                    use jq_jit::interpreter::StringAddPart;
                    // Collect unique field names
                    let mut field_names: Vec<&str> = Vec::new();
                    let mut field_idx_map = std::collections::HashMap::new();
                    for part in sac_parts.iter() {
                        let name = match part {
                            StringAddPart::Field(f) | StringAddPart::FieldToString(f) => f.as_str(),
                            _ => continue,
                        };
                        if !field_idx_map.contains_key(name) {
                            field_idx_map.insert(name, field_names.len());
                            field_names.push(name);
                        }
                    }
                    // Pre-compute actions for hot loop: avoid HashMap lookups per object
                    // 0 = literal (strip_quotes=false), 1 = field (strip_quotes=true), 2 = field_tostring (strip_quotes=true)
                    // We encode as (field_idx_or_lit_idx, strip_quotes)
                    let mut actions: Vec<(usize, bool)> = Vec::with_capacity(sac_parts.len()); // (idx, is_field)
                    let mut lit_bufs: Vec<Vec<u8>> = Vec::new();
                    for part in sac_parts.iter() {
                        match part {
                            StringAddPart::Literal(s) => {
                                let mut buf = Vec::new();
                                for &b in s.as_bytes() {
                                    match b {
                                        b'"' => buf.extend_from_slice(b"\\\""),
                                        b'\\' => buf.extend_from_slice(b"\\\\"),
                                        b'\n' => buf.extend_from_slice(b"\\n"),
                                        b'\r' => buf.extend_from_slice(b"\\r"),
                                        b'\t' => buf.extend_from_slice(b"\\t"),
                                        c if c < 0x20 => { let _ = write!(buf, "\\u{:04x}", c); }
                                        _ => buf.push(b),
                                    }
                                }
                                let lit_idx = lit_bufs.len();
                                lit_bufs.push(buf);
                                actions.push((lit_idx, false)); // not a field
                            }
                            StringAddPart::Field(f) | StringAddPart::FieldToString(f) => {
                                actions.push((field_idx_map[f.as_str()], true)); // is a field
                            }
                        }
                    }
                    let mut ranges_buf = vec![(0usize, 0usize); field_names.len()];
                    let mut lit_idx_counter = 0usize;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_names, &mut ranges_buf) {
                            compact_buf.push(b'"');
                            lit_idx_counter = 0;
                            for &(idx, is_field) in &actions {
                                if is_field {
                                    let (vs, ve) = ranges_buf[idx];
                                    let val = &raw[vs..ve];
                                    if val[0] == b'"' && val.len() >= 2 {
                                        compact_buf.extend_from_slice(&val[1..val.len()-1]);
                                    } else {
                                        compact_buf.extend_from_slice(val);
                                    }
                                } else {
                                    compact_buf.extend_from_slice(&lit_bufs[idx]);
                                }
                            }
                            compact_buf.extend_from_slice(b"\"\n");
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref op, threshold)) = select_cmp {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(val) = json_object_get_num(raw, 0, field) {
                            let pass = match op {
                                BinOp::Gt => val > threshold,
                                BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold,
                                BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold,
                                BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                emit_raw_ln!(&mut compact_buf, raw);
                                if compact_buf.len() >= 1 << 17 {
                                    let _ = out.write_all(&compact_buf);
                                    compact_buf.clear();
                                }
                            }
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref arith_ops, ref cmp_op, threshold)) = select_arith_cmp {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(mut val) = json_object_get_num(raw, 0, field) {
                            for (aop, n) in arith_ops {
                                val = match aop {
                                    BinOp::Add => val + n, BinOp::Sub => val - n,
                                    BinOp::Mul => val * n, BinOp::Div => val / n,
                                    BinOp::Mod => val % n, _ => val,
                                };
                            }
                            let pass = match cmp_op {
                                BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                emit_raw_ln!(&mut compact_buf, raw);
                                if compact_buf.len() >= 1 << 17 {
                                    let _ = out.write_all(&compact_buf);
                                    compact_buf.clear();
                                }
                            }
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref op, ref val)) = select_str {
                    use jq_jit::ir::BinOp;
                    // Build expected JSON string: "value" (with quotes)
                    let mut expected = Vec::with_capacity(val.len() + 2);
                    expected.push(b'"');
                    expected.extend_from_slice(val.as_bytes());
                    expected.push(b'"');
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
                            let val_bytes = &raw[vs..ve];
                            let matches = val_bytes == expected.as_slice();
                            let pass = match op { BinOp::Eq => matches, BinOp::Ne => !matches, _ => false };
                            if pass {
                                emit_raw_ln!(&mut compact_buf, raw);
                                if compact_buf.len() >= 1 << 17 {
                                    let _ = out.write_all(&compact_buf);
                                    compact_buf.clear();
                                }
                            }
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref builtin, ref arg)) = select_str_test {
                    // select(.field | startswith/endswith/contains("str")) — raw byte test
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
                            let val = &raw[vs..ve];
                            // Only handle simple quoted strings (no backslash escapes)
                            if val.len() >= 2 && val[0] == b'"' && val[ve-vs-1] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                let inner = &val[1..ve-vs-1];
                                let pass = match builtin.as_str() {
                                    "startswith" => inner.starts_with(arg.as_bytes()),
                                    "endswith" => inner.ends_with(arg.as_bytes()),
                                    "contains" => bytes_contains(inner, arg.as_bytes()),
                                    _ => false,
                                };
                                if pass {
                                    emit_raw_ln!(&mut compact_buf, raw);
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref rt_field, ref rt_pattern, ref rt_flags)) = select_regex_test {
                    // select(.field | test("regex")) — raw byte regex test, pass through matching lines
                    let re_pattern = if let Some(flags) = rt_flags {
                        let mut prefix = String::from("(?");
                        for c in flags.chars() {
                            match c { 'i' | 'm' | 's' => prefix.push(c), _ => {} }
                        }
                        prefix.push(')');
                        prefix.push_str(rt_pattern);
                        prefix
                    } else {
                        rt_pattern.clone()
                    };
                    if let Ok(re) = regex::Regex::new(&re_pattern) {
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, rt_field) {
                                let val = &raw[vs..ve];
                                if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                    && !val[1..val.len()-1].contains(&b'\\')
                                {
                                    let content = unsafe { std::str::from_utf8_unchecked(&val[1..val.len()-1]) };
                                    if re.is_match(content) {
                                        emit_raw_ln!(&mut compact_buf, raw);
                                    }
                                }
                            }
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                            Ok(())
                        })
                    } else {
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            Ok(())
                        })
                    }
                } else if let Some((ref fields, ref op, threshold)) = select_nested_cmp {
                    use jq_jit::ir::BinOp;
                    let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_nested_field_raw(raw, 0, &field_refs) {
                            if let Some(val) = parse_json_num(&raw[vs..ve]) {
                                let pass = match op {
                                    BinOp::Gt => val > threshold,
                                    BinOp::Lt => val < threshold,
                                    BinOp::Ge => val >= threshold,
                                    BinOp::Le => val <= threshold,
                                    BinOp::Eq => val == threshold,
                                    BinOp::Ne => val != threshold,
                                    _ => false,
                                };
                                if pass {
                                    emit_raw_ln!(&mut compact_buf, raw);
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref alt_field, ref fallback_bytes)) = field_alt {
                    // .field // literal: extract field, output raw or fallback if null/false
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, alt_field) {
                            let val = &raw[vs..ve];
                            if val == b"null" || val == b"false" {
                                compact_buf.extend_from_slice(fallback_bytes);
                            } else {
                                compact_buf.extend_from_slice(val);
                            }
                        } else {
                            compact_buf.extend_from_slice(fallback_bytes);
                        }
                        compact_buf.push(b'\n');
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref prim_field, ref fallback_field)) = field_field_alt {
                    // .field1 // .field2: try primary, if null/false/missing use fallback
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        let use_primary = if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, prim_field) {
                            let pval = &raw[vs..ve];
                            if pval != b"null" && pval != b"false" {
                                compact_buf.extend_from_slice(pval);
                                true
                            } else { false }
                        } else { false };
                        if !use_primary {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, fallback_field) {
                                compact_buf.extend_from_slice(&raw[vs..ve]);
                            } else {
                                compact_buf.extend_from_slice(b"null");
                            }
                        }
                        compact_buf.push(b'\n');
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref branches, ref else_output)) = cond_chain {
                    use jq_jit::interpreter::{BranchOutput, CondRhs};
                    use jq_jit::ir::BinOp;

                    // Specialized path: single-branch with lazy output fetch (no remap outputs)
                    if branches.len() == 1 && !matches!(branches[0].output, BranchOutput::Remap(_) | BranchOutput::Computed(_)) && !matches!(else_output, BranchOutput::Remap(_) | BranchOutput::Computed(_)) {
                        let br = &branches[0];
                        let cond_field = br.cond_field.as_str();
                        let cond_arith = &br.cond_arith_ops;
                        let cond_op = &br.cond_op;
                        let then_out = &br.output;
                        let rhs_field: Option<&str> = if let CondRhs::Field(ref f) = br.cond_rhs { Some(f.as_str()) } else { None };
                        let rhs_const: Option<f64> = if let CondRhs::Const(n) = br.cond_rhs { Some(n) } else { None };
                        let is_non_numeric_cmp = matches!(br.cond_rhs, CondRhs::Null | CondRhs::Str(_) | CondRhs::Bool(_) | CondRhs::Startswith(_) | CondRhs::Endswith(_) | CondRhs::Contains(_));
                        let rhs_str_json: Option<Vec<u8>> = if let CondRhs::Str(ref s) = br.cond_rhs {
                            let mut buf = Vec::with_capacity(s.len() + 2);
                            buf.push(b'"');
                            buf.extend_from_slice(s.as_bytes());
                            buf.push(b'"');
                            Some(buf)
                        } else { None };
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            // Non-numeric comparison (null/str/bool): compare raw bytes
                            if is_non_numeric_cmp {
                                let pass = if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, cond_field) {
                                    let field_bytes = &raw[vs..ve];
                                    match &br.cond_rhs {
                                        CondRhs::Null => match cond_op {
                                            BinOp::Eq => field_bytes == b"null",
                                            BinOp::Ne => field_bytes != b"null",
                                            _ => false,
                                        },
                                        CondRhs::Bool(expected) => {
                                            let is_true = field_bytes == b"true";
                                            let is_false = field_bytes == b"false";
                                            let val = if is_true { Some(true) } else if is_false { Some(false) } else { None };
                                            val.map_or(false, |v| match cond_op {
                                                BinOp::Eq => v == *expected,
                                                BinOp::Ne => v != *expected,
                                                _ => false,
                                            })
                                        }
                                        CondRhs::Str(_) => {
                                            let rhs_json = rhs_str_json.as_ref().unwrap();
                                            match cond_op {
                                                BinOp::Eq => field_bytes == rhs_json.as_slice(),
                                                BinOp::Ne => field_bytes != rhs_json.as_slice(),
                                                _ => false,
                                            }
                                        }
                                        CondRhs::Startswith(ref s) => {
                                            // field_bytes is JSON string like "user_123" — strip quotes
                                            field_bytes.len() >= 2 && field_bytes[0] == b'"'
                                                && field_bytes[1..field_bytes.len()-1].starts_with(s.as_bytes())
                                        }
                                        CondRhs::Endswith(ref s) => {
                                            field_bytes.len() >= 2 && field_bytes[0] == b'"'
                                                && field_bytes[1..field_bytes.len()-1].ends_with(s.as_bytes())
                                        }
                                        CondRhs::Contains(ref s) => {
                                            field_bytes.len() >= 2 && field_bytes[0] == b'"' && {
                                                let inner = &field_bytes[1..field_bytes.len()-1];
                                                let sb = s.as_bytes();
                                                inner.windows(sb.len()).any(|w| w == sb)
                                            }
                                        }
                                        _ => false,
                                    }
                                } else {
                                    // Field missing (null)
                                    matches!(br.cond_rhs, CondRhs::Null) && matches!(cond_op, BinOp::Eq)
                                };
                                let out_br = if pass { then_out } else { else_output };
                                match out_br {
                                    BranchOutput::Literal(ref bytes) => {
                                        compact_buf.extend_from_slice(bytes);
                                        compact_buf.push(b'\n');
                                    }
                                    BranchOutput::Field(ref f) => {
                                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, f) {
                                            let val = &raw[vs..ve];
                                            if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                                push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                            } else {
                                                compact_buf.extend_from_slice(val);
                                            }
                                            compact_buf.push(b'\n');
                                        } else {
                                            compact_buf.extend_from_slice(b"null\n");
                                        }
                                    }
                                    BranchOutput::Empty => {}
                                    BranchOutput::Remap(_) | BranchOutput::Computed(_) => unreachable!(),
                                }
                                if compact_buf.len() >= 1 << 17 {
                                    let _ = out.write_all(&compact_buf);
                                    compact_buf.clear();
                                }
                                return Ok(());
                            }
                            // Get comparison values
                            let (lv, rv, got) = if let Some(rf) = rhs_field {
                                if let Some((l, r)) = json_object_get_two_nums(raw, 0, cond_field, rf) {
                                    (l, r, true)
                                } else { (0.0, 0.0, false) }
                            } else {
                                let thr = rhs_const.unwrap();
                                if let Some(l) = json_object_get_num(raw, 0, cond_field) {
                                    (l, thr, true)
                                } else { (0.0, 0.0, false) }
                            };
                            if got {
                                // Apply arithmetic chain to LHS (keep original for field output)
                                let orig_lv = lv;
                                let mut cmp_lv = lv;
                                for (aop, n) in cond_arith {
                                    cmp_lv = match aop {
                                        BinOp::Add => cmp_lv + n, BinOp::Sub => cmp_lv - n,
                                        BinOp::Mul => cmp_lv * n, BinOp::Div => cmp_lv / n,
                                        BinOp::Mod => cmp_lv % n, _ => cmp_lv,
                                    };
                                }
                                let pass = match cond_op {
                                    BinOp::Gt => cmp_lv > rv, BinOp::Lt => cmp_lv < rv,
                                    BinOp::Ge => cmp_lv >= rv, BinOp::Le => cmp_lv <= rv,
                                    BinOp::Eq => cmp_lv == rv, BinOp::Ne => cmp_lv != rv,
                                    _ => false,
                                };
                                let out_br = if pass { then_out } else { else_output };
                                match out_br {
                                    BranchOutput::Literal(ref bytes) => {
                                        compact_buf.extend_from_slice(bytes);
                                        compact_buf.push(b'\n');
                                    }
                                    BranchOutput::Field(ref f) => {
                                        let fs = f.as_str();
                                        if fs == cond_field {
                                            push_jq_number_bytes(&mut compact_buf, orig_lv);
                                            compact_buf.push(b'\n');
                                        } else if rhs_field.map_or(false, |rf| fs == rf) {
                                            push_jq_number_bytes(&mut compact_buf, rv);
                                            compact_buf.push(b'\n');
                                        } else {
                                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, f) {
                                                let val = &raw[vs..ve];
                                                if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                                    push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                                } else {
                                                    compact_buf.extend_from_slice(val);
                                                }
                                                compact_buf.push(b'\n');
                                            }
                                        }
                                    }
                                    BranchOutput::Empty => {}
                                    BranchOutput::Remap(_) | BranchOutput::Computed(_) => unreachable!(),
                                }
                            } else {
                                // Fields not numeric — take else branch
                                match else_output {
                                    BranchOutput::Literal(ref bytes) => {
                                        compact_buf.extend_from_slice(bytes);
                                        compact_buf.push(b'\n');
                                    }
                                    BranchOutput::Field(ref f) => {
                                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, f) {
                                            let val = &raw[vs..ve];
                                            if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                                push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                            } else {
                                                compact_buf.extend_from_slice(val);
                                            }
                                            compact_buf.push(b'\n');
                                        }
                                    }
                                    BranchOutput::Empty => {}
                                    BranchOutput::Remap(_) | BranchOutput::Computed(_) => unreachable!(),
                                }
                            }
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                            Ok(())
                        })
                    } else {
                    // General path: collect all unique fields needed for conditions and field outputs
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    let ensure_field = |f: &String, all: &mut Vec<String>, idx: &mut std::collections::HashMap<String, usize>| {
                        if !idx.contains_key(f) {
                            idx.insert(f.clone(), all.len());
                            all.push(f.clone());
                        }
                    };
                    let collect_output_fields = |out: &BranchOutput, all: &mut Vec<String>, idx: &mut std::collections::HashMap<String, usize>| {
                        match out {
                            BranchOutput::Field(ref f) => { ensure_field(f, all, idx); }
                            BranchOutput::Remap(ref entries) => {
                                for (_, rexpr) in entries {
                                    for name in remap_expr_fields(rexpr) {
                                        let s = name.to_string();
                                        ensure_field(&s, all, idx);
                                    }
                                }
                            }
                            BranchOutput::Computed(ref rexpr) => {
                                for name in remap_expr_fields(rexpr) {
                                    let s = name.to_string();
                                    ensure_field(&s, all, idx);
                                }
                            }
                            _ => {}
                        }
                    };
                    for br in branches {
                        ensure_field(&br.cond_field, &mut all_fields, &mut field_idx);
                        if let CondRhs::Field(ref f) = br.cond_rhs {
                            ensure_field(f, &mut all_fields, &mut field_idx);
                        }
                        collect_output_fields(&br.output, &mut all_fields, &mut field_idx);
                    }
                    collect_output_fields(else_output, &mut all_fields, &mut field_idx);
                    // Pre-resolve remap exprs for each branch
                    // Pre-resolve Computed outputs for each branch
                    let branch_computed: Vec<Option<ResolvedRemap>> = branches.iter().map(|br| {
                        if let BranchOutput::Computed(ref rexpr) = br.output {
                            Some(resolve_one_remap(rexpr, &field_idx))
                        } else { None }
                    }).collect();
                    let else_computed = if let BranchOutput::Computed(ref rexpr) = else_output {
                        Some(resolve_one_remap(rexpr, &field_idx))
                    } else { None };
                    let branch_resolved: Vec<Option<(Vec<Vec<u8>>, Vec<ResolvedRemap>, &[u8])>> = branches.iter().map(|br| {
                        if let BranchOutput::Remap(ref entries) = br.output {
                            let kp = if use_pretty_buf {
                                build_obj_key_prefixes_pretty(entries.iter().map(|(k, _)| k.as_str()))
                            } else {
                                build_obj_key_prefixes(entries.iter().map(|(k, _)| k.as_str()))
                            };
                            let res = resolve_remap_exprs(entries, &field_idx);
                            Some((kp, res, if use_pretty_buf { &b"\n}\n"[..] } else { &b"}\n"[..] }))
                        } else { None }
                    }).collect();
                    let else_resolved = if let BranchOutput::Remap(ref entries) = else_output {
                        let kp = if use_pretty_buf {
                            build_obj_key_prefixes_pretty(entries.iter().map(|(k, _)| k.as_str()))
                        } else {
                            build_obj_key_prefixes(entries.iter().map(|(k, _)| k.as_str()))
                        };
                        let res = resolve_remap_exprs(entries, &field_idx);
                        Some((kp, res, if use_pretty_buf { &b"\n}\n"[..] } else { &b"}\n"[..] }))
                    } else { None };
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            let mut output = None;
                            let mut output_idx = 0usize;
                            for (i, br) in branches.iter().enumerate() {
                                let idx = field_idx[&br.cond_field];
                                let (vs, ve) = ranges_buf[idx];
                                let field_bytes = &raw[vs..ve];
                                let pass = match &br.cond_rhs {
                                    CondRhs::Null => match br.cond_op {
                                        BinOp::Eq => field_bytes == b"null",
                                        BinOp::Ne => field_bytes != b"null",
                                        _ => false,
                                    },
                                    CondRhs::Bool(expected) => {
                                        let val = if field_bytes == b"true" { Some(true) } else if field_bytes == b"false" { Some(false) } else { None };
                                        val.map_or(false, |v| match br.cond_op {
                                            BinOp::Eq => v == *expected,
                                            BinOp::Ne => v != *expected,
                                            _ => false,
                                        })
                                    }
                                    CondRhs::Str(ref s) => {
                                        let rhs_json = {
                                            let mut buf = Vec::with_capacity(s.len() + 2);
                                            buf.push(b'"');
                                            buf.extend_from_slice(s.as_bytes());
                                            buf.push(b'"');
                                            buf
                                        };
                                        match br.cond_op {
                                            BinOp::Eq => field_bytes == rhs_json.as_slice(),
                                            BinOp::Ne => field_bytes != rhs_json.as_slice(),
                                            _ => false,
                                        }
                                    }
                                    CondRhs::Startswith(ref s) => {
                                        field_bytes.len() >= 2 && field_bytes[0] == b'"'
                                            && field_bytes[1..field_bytes.len()-1].starts_with(s.as_bytes())
                                    }
                                    CondRhs::Endswith(ref s) => {
                                        field_bytes.len() >= 2 && field_bytes[0] == b'"'
                                            && field_bytes[1..field_bytes.len()-1].ends_with(s.as_bytes())
                                    }
                                    CondRhs::Contains(ref s) => {
                                        field_bytes.len() >= 2 && field_bytes[0] == b'"' && {
                                            let inner = &field_bytes[1..field_bytes.len()-1];
                                            let sb = s.as_bytes();
                                            inner.windows(sb.len()).any(|w| w == sb)
                                        }
                                    }
                                    _ => {
                                        // Numeric comparison
                                        if let Some(mut val) = parse_json_num(field_bytes) {
                                            for (aop, n) in &br.cond_arith_ops {
                                                val = match aop {
                                                    BinOp::Add => val + n, BinOp::Sub => val - n,
                                                    BinOp::Mul => val * n, BinOp::Div => val / n,
                                                    BinOp::Mod => val % n, _ => val,
                                                };
                                            }
                                            let rhs_val = match &br.cond_rhs {
                                                CondRhs::Const(n) => *n,
                                                CondRhs::Field(ref f) => {
                                                    let ri = field_idx[f];
                                                    let (rs, re) = ranges_buf[ri];
                                                    match parse_json_num(&raw[rs..re]) { Some(v) => v, None => { continue; } }
                                                }
                                                _ => unreachable!(),
                                            };
                                            match br.cond_op {
                                                BinOp::Gt => val > rhs_val, BinOp::Lt => val < rhs_val,
                                                BinOp::Ge => val >= rhs_val, BinOp::Le => val <= rhs_val,
                                                BinOp::Eq => val == rhs_val, BinOp::Ne => val != rhs_val,
                                                _ => false,
                                            }
                                        } else { false }
                                    }
                                };
                                if pass { output = Some(&br.output); output_idx = i; break; }
                            }
                            let out_branch = output.unwrap_or(else_output);
                            match out_branch {
                                BranchOutput::Literal(ref bytes) => {
                                    compact_buf.extend_from_slice(bytes);
                                    compact_buf.push(b'\n');
                                }
                                BranchOutput::Field(ref f) => {
                                    let idx = field_idx[f];
                                    let (vs, ve) = ranges_buf[idx];
                                    let val = &raw[vs..ve];
                                    if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                        push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                    } else {
                                        compact_buf.extend_from_slice(val);
                                    }
                                    compact_buf.push(b'\n');
                                }
                                BranchOutput::Remap(_) => {
                                    let resolved_data = if output.is_some() {
                                        branch_resolved[output_idx].as_ref()
                                    } else {
                                        else_resolved.as_ref()
                                    };
                                    if let Some((ref kp, ref res, close)) = resolved_data {
                                        for (j, rv) in res.iter().enumerate() {
                                            compact_buf.extend_from_slice(&kp[j]);
                                            emit_resolved_value(&mut compact_buf, rv, raw, &ranges_buf);
                                        }
                                        compact_buf.extend_from_slice(close);
                                    }
                                }
                                BranchOutput::Computed(_) => {
                                    let resolved_remap = if output.is_some() {
                                        branch_computed[output_idx].as_ref()
                                    } else {
                                        else_computed.as_ref()
                                    };
                                    if let Some(rv) = resolved_remap {
                                        emit_resolved_value(&mut compact_buf, rv, raw, &ranges_buf);
                                        compact_buf.push(b'\n');
                                    }
                                }
                                BranchOutput::Empty => { /* produce no output */ }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                    } // end general path
                } else if let Some((ref field, ref op, threshold, ref t_bytes, ref f_bytes)) = cmp_branch_lit {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(val) = json_object_get_num(raw, 0, field) {
                            let pass = match op {
                                BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            compact_buf.extend_from_slice(if pass { t_bytes } else { f_bytes });
                            compact_buf.push(b'\n');
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref field, ref arith_ops, ref cmp_op, threshold, ref t_bytes, ref f_bytes)) = arith_cmp_branch_lit {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(mut val) = json_object_get_num(raw, 0, field) {
                            for (aop, n) in arith_ops {
                                val = match aop {
                                    BinOp::Add => val + n, BinOp::Sub => val - n,
                                    BinOp::Mul => val * n, BinOp::Div => val / n,
                                    BinOp::Mod => val % n, _ => val,
                                };
                            }
                            let pass = match cmp_op {
                                BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            compact_buf.extend_from_slice(if pass { t_bytes } else { f_bytes });
                            compact_buf.push(b'\n');
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref f1, ref cmp_op, ref f2, ref t_bytes, ref f_bytes)) = field_field_cmp_branch {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, f1, f2) {
                            let pass = match cmp_op {
                                BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                _ => false,
                            };
                            compact_buf.extend_from_slice(if pass { t_bytes } else { f_bytes });
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref iacond, ref then_arr, ref else_arr)) = if_cmp_arrays {
                    use jq_jit::ir::BinOp;
                    use jq_jit::interpreter::IfArrayCond;
                    // Collect all fields from condition + both branches
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    match iacond {
                        IfArrayCond::FieldConst(f, _, _) => {
                            field_idx.insert(f.clone(), 0);
                            all_fields.push(f.clone());
                        }
                        IfArrayCond::FieldField(f1, _, f2) => {
                            field_idx.insert(f1.clone(), 0);
                            all_fields.push(f1.clone());
                            if !field_idx.contains_key(f2) {
                                field_idx.insert(f2.clone(), 1);
                                all_fields.push(f2.clone());
                            }
                        }
                    }
                    for rexpr in then_arr.iter().chain(else_arr.iter()) {
                        for name in remap_expr_fields(rexpr) {
                            if !field_idx.contains_key(name) {
                                field_idx.insert(name.to_string(), all_fields.len());
                                all_fields.push(name.to_string());
                            }
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved_then: Vec<ResolvedRemap> = then_arr.iter()
                        .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                        .collect();
                    let resolved_else: Vec<ResolvedRemap> = else_arr.iter()
                        .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                        .collect();
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            let pass = match iacond {
                                IfArrayCond::FieldConst(f, op, threshold) => {
                                    let idx = field_idx[f];
                                    let r = &ranges_buf[idx];
                                    if r.0 < r.1 {
                                        if let Some(v) = parse_json_num(&raw[r.0..r.1]) {
                                            match op {
                                                BinOp::Gt => v > *threshold, BinOp::Lt => v < *threshold,
                                                BinOp::Ge => v >= *threshold, BinOp::Le => v <= *threshold,
                                                BinOp::Eq => v == *threshold, BinOp::Ne => v != *threshold,
                                                _ => false,
                                            }
                                        } else { false }
                                    } else { false }
                                }
                                IfArrayCond::FieldField(f1, op, f2) => {
                                    let i1 = field_idx[f1];
                                    let i2 = field_idx[f2];
                                    let r1 = &ranges_buf[i1];
                                    let r2 = &ranges_buf[i2];
                                    if r1.0 < r1.1 && r2.0 < r2.1 {
                                        if let (Some(v1), Some(v2)) = (
                                            parse_json_num(&raw[r1.0..r1.1]),
                                            parse_json_num(&raw[r2.0..r2.1]),
                                        ) {
                                            match op {
                                                BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                                BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                                BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                                _ => false,
                                            }
                                        } else { false }
                                    } else { false }
                                }
                            };
                            let resolved = if pass { &resolved_then } else { &resolved_else };
                            compact_buf.push(b'[');
                            for (i, res) in resolved.iter().enumerate() {
                                if i > 0 { compact_buf.push(b','); }
                                emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                            }
                            compact_buf.extend_from_slice(b"]\n");
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref conj, ref cmps)) = select_compound {
                    use jq_jit::ir::BinOp;
                    let is_and = matches!(conj, BinOp::And);
                    // Specialized path for exactly 2 comparisons on different fields: use get_two_nums
                    let two_field = if cmps.len() == 2 && cmps[0].0 != cmps[1].0 {
                        Some((cmps[0].0.as_str(), cmps[1].0.as_str()))
                    } else { None };
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        let pass = if let Some((f1, f2)) = two_field {
                            if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, f1, f2) {
                                let c1 = match &cmps[0].1 {
                                    BinOp::Gt => v1 > cmps[0].2, BinOp::Lt => v1 < cmps[0].2,
                                    BinOp::Ge => v1 >= cmps[0].2, BinOp::Le => v1 <= cmps[0].2,
                                    BinOp::Eq => v1 == cmps[0].2, BinOp::Ne => v1 != cmps[0].2,
                                    _ => false,
                                };
                                let c2 = match &cmps[1].1 {
                                    BinOp::Gt => v2 > cmps[1].2, BinOp::Lt => v2 < cmps[1].2,
                                    BinOp::Ge => v2 >= cmps[1].2, BinOp::Le => v2 <= cmps[1].2,
                                    BinOp::Eq => v2 == cmps[1].2, BinOp::Ne => v2 != cmps[1].2,
                                    _ => false,
                                };
                                if is_and { c1 && c2 } else { c1 || c2 }
                            } else { false }
                        } else if is_and {
                            cmps.iter().all(|(field, op, threshold)| {
                                json_object_get_num(raw, 0, field).map_or(false, |val| match op {
                                    BinOp::Gt => val > *threshold, BinOp::Lt => val < *threshold,
                                    BinOp::Ge => val >= *threshold, BinOp::Le => val <= *threshold,
                                    BinOp::Eq => val == *threshold, BinOp::Ne => val != *threshold,
                                    _ => false,
                                })
                            })
                        } else {
                            cmps.iter().any(|(field, op, threshold)| {
                                json_object_get_num(raw, 0, field).map_or(false, |val| match op {
                                    BinOp::Gt => val > *threshold, BinOp::Lt => val < *threshold,
                                    BinOp::Ge => val >= *threshold, BinOp::Le => val <= *threshold,
                                    BinOp::Eq => val == *threshold, BinOp::Ne => val != *threshold,
                                    _ => false,
                                })
                            })
                        };
                        if pass {
                            emit_raw_ln!(&mut compact_buf, raw);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref conj, ref cmps, ref out_field)) = select_compound_field {
                    use jq_jit::ir::BinOp;
                    let is_and = matches!(conj, BinOp::And);
                    // Specialized path for 2 comparisons on different fields: lazy fetch output
                    let two_field_lazy = if cmps.len() == 2 && cmps[0].0 != cmps[1].0
                        && cmps[0].0 != *out_field && cmps[1].0 != *out_field {
                        Some((cmps[0].0.as_str(), cmps[1].0.as_str(), out_field.as_str()))
                    } else { None };
                    // General path: Collect all fields
                    let mut all_fields: Vec<String> = cmps.iter().map(|(f, _, _)| f.clone()).collect();
                    if !all_fields.contains(out_field) { all_fields.push(out_field.clone()); }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let out_idx = all_fields.iter().position(|f| f == out_field).unwrap();
                    let cmp_indices: Vec<(usize, BinOp, f64)> = cmps.iter().map(|(f, op, thr)| {
                        (all_fields.iter().position(|af| af == f).unwrap(), *op, *thr)
                    }).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((f1, f2, of)) = two_field_lazy {
                            // Lazy fetch: check comparison fields first, output field only on pass
                            if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, f1, f2) {
                                let c1 = match &cmps[0].1 {
                                    BinOp::Gt => v1 > cmps[0].2, BinOp::Lt => v1 < cmps[0].2,
                                    BinOp::Ge => v1 >= cmps[0].2, BinOp::Le => v1 <= cmps[0].2,
                                    BinOp::Eq => v1 == cmps[0].2, BinOp::Ne => v1 != cmps[0].2,
                                    _ => false,
                                };
                                let c2 = match &cmps[1].1 {
                                    BinOp::Gt => v2 > cmps[1].2, BinOp::Lt => v2 < cmps[1].2,
                                    BinOp::Ge => v2 >= cmps[1].2, BinOp::Le => v2 <= cmps[1].2,
                                    BinOp::Eq => v2 == cmps[1].2, BinOp::Ne => v2 != cmps[1].2,
                                    _ => false,
                                };
                                let pass = if is_and { c1 && c2 } else { c1 || c2 };
                                if pass {
                                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, of) {
                                        let val = &raw[vs..ve];
                                        if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                            push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                        } else {
                                            compact_buf.extend_from_slice(val);
                                        }
                                        compact_buf.push(b'\n');
                                    }
                                }
                            }
                        } else {
                            if !json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                                return Ok(());
                            }
                            let check = |idx: usize, op: &BinOp, thr: &f64| -> bool {
                                let (vs, ve) = ranges_buf[idx];
                                parse_json_num(&raw[vs..ve]).map_or(false, |val| match op {
                                    BinOp::Gt => val > *thr, BinOp::Lt => val < *thr,
                                    BinOp::Ge => val >= *thr, BinOp::Le => val <= *thr,
                                    BinOp::Eq => val == *thr, BinOp::Ne => val != *thr,
                                    _ => false,
                                })
                            };
                            let pass = if is_and {
                                cmp_indices.iter().all(|(idx, op, thr)| check(*idx, op, thr))
                            } else {
                                cmp_indices.iter().any(|(idx, op, thr)| check(*idx, op, thr))
                            };
                            if pass {
                                let (vs, ve) = ranges_buf[out_idx];
                                let val = &raw[vs..ve];
                                if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                    push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                } else {
                                    compact_buf.extend_from_slice(val);
                                }
                                compact_buf.push(b'\n');
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref conj, ref cmps, ref remap)) = select_compound_remap {
                    use jq_jit::ir::BinOp;
                    let is_and = matches!(conj, BinOp::And);
                    // Lazy path for 2 comparisons on different fields
                    let two_cmp_lazy = if cmps.len() == 2 && cmps[0].0 != cmps[1].0 {
                        Some((cmps[0].0.as_str(), cmps[1].0.as_str()))
                    } else { None };
                    // Build remap field list (used for lazy fetch on pass)
                    let remap_fields: Vec<&str> = {
                        let mut rf = Vec::new();
                        for (_, f) in remap { if !rf.contains(&f.as_str()) { rf.push(f.as_str()); } }
                        rf
                    };
                    let remap_field_map: std::collections::HashMap<&str, usize> =
                        remap_fields.iter().enumerate().map(|(i, f)| (*f, i)).collect();
                    let remap_out_indices: Vec<(&str, usize)> = remap.iter().map(|(k, f)| {
                        (k.as_str(), remap_field_map[f.as_str()])
                    }).collect();
                    // General path fields
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    let ensure_field = |f: &String, all: &mut Vec<String>, idx: &mut std::collections::HashMap<String, usize>| {
                        if !idx.contains_key(f) { idx.insert(f.clone(), all.len()); all.push(f.clone()); }
                    };
                    for (f, _, _) in cmps { ensure_field(f, &mut all_fields, &mut field_idx); }
                    for (_, f) in remap { ensure_field(f, &mut all_fields, &mut field_idx); }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let cmp_indices: Vec<(usize, BinOp, f64)> = cmps.iter().map(|(f, op, thr)| {
                        (field_idx[f], *op, *thr)
                    }).collect();
                    let gen_remap_indices: Vec<(&str, usize)> = remap.iter().map(|(k, f)| {
                        (k.as_str(), field_idx[f])
                    }).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); std::cmp::max(field_refs.len(), remap_fields.len())];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((f1, f2)) = two_cmp_lazy {
                            // Lazy: check comparisons first, extract remap fields only on pass
                            if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, f1, f2) {
                                let c1 = match &cmps[0].1 {
                                    BinOp::Gt => v1 > cmps[0].2, BinOp::Lt => v1 < cmps[0].2,
                                    BinOp::Ge => v1 >= cmps[0].2, BinOp::Le => v1 <= cmps[0].2,
                                    BinOp::Eq => v1 == cmps[0].2, BinOp::Ne => v1 != cmps[0].2,
                                    _ => false,
                                };
                                let c2 = match &cmps[1].1 {
                                    BinOp::Gt => v2 > cmps[1].2, BinOp::Lt => v2 < cmps[1].2,
                                    BinOp::Ge => v2 >= cmps[1].2, BinOp::Le => v2 <= cmps[1].2,
                                    BinOp::Eq => v2 == cmps[1].2, BinOp::Ne => v2 != cmps[1].2,
                                    _ => false,
                                };
                                let pass = if is_and { c1 && c2 } else { c1 || c2 };
                                if pass {
                                    if json_object_get_fields_raw_buf(raw, 0, &remap_fields, &mut ranges_buf) {
                                        compact_buf.push(b'{');
                                        for (i, (key, fidx)) in remap_out_indices.iter().enumerate() {
                                            if i > 0 { compact_buf.push(b','); }
                                            compact_buf.push(b'"');
                                            compact_buf.extend_from_slice(key.as_bytes());
                                            compact_buf.extend_from_slice(b"\":");
                                            let (vs, ve) = ranges_buf[*fidx];
                                            compact_buf.extend_from_slice(&raw[vs..ve]);
                                        }
                                        compact_buf.extend_from_slice(b"}\n");
                                    }
                                }
                            }
                        } else {
                            if !json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                                return Ok(());
                            }
                            let check = |idx: usize, op: &BinOp, thr: &f64| -> bool {
                                let (vs, ve) = ranges_buf[idx];
                                parse_json_num(&raw[vs..ve]).map_or(false, |val| match op {
                                    BinOp::Gt => val > *thr, BinOp::Lt => val < *thr,
                                    BinOp::Ge => val >= *thr, BinOp::Le => val <= *thr,
                                    BinOp::Eq => val == *thr, BinOp::Ne => val != *thr,
                                    _ => false,
                                })
                            };
                            let pass = if is_and {
                                cmp_indices.iter().all(|(idx, op, thr)| check(*idx, op, thr))
                            } else {
                                cmp_indices.iter().any(|(idx, op, thr)| check(*idx, op, thr))
                            };
                            if pass {
                                compact_buf.push(b'{');
                                for (i, (key, fidx)) in gen_remap_indices.iter().enumerate() {
                                    if i > 0 { compact_buf.push(b','); }
                                    compact_buf.push(b'"');
                                    compact_buf.extend_from_slice(key.as_bytes());
                                    compact_buf.extend_from_slice(b"\":");
                                    let (vs, ve) = ranges_buf[*fidx];
                                    compact_buf.extend_from_slice(&raw[vs..ve]);
                                }
                                compact_buf.extend_from_slice(b"}\n");
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref shm_fields, shm_is_and)) = select_has_multi {
                    let field_refs: Vec<&str> = shm_fields.iter().map(|s| s.as_str()).collect();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        let pass = if field_refs.len() == 1 {
                            json_object_has_key(raw, 0, field_refs[0]).unwrap_or(false)
                        } else if shm_is_and {
                            json_object_has_all_keys(raw, 0, &field_refs).unwrap_or(false)
                        } else {
                            json_object_has_any_key(raw, 0, &field_refs).unwrap_or(false)
                        };
                        if pass {
                            emit_raw_ln!(&mut compact_buf, raw);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref op, threshold, ref out_field)) = select_cmp_field {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                            let pass = match op {
                                BinOp::Gt => val > threshold,
                                BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold,
                                BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold,
                                BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                    compact_buf.extend_from_slice(&raw[vs..ve]);
                                    compact_buf.push(b'\n');
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref arith_ops, ref cmp_op, threshold, ref out_field)) = select_arith_cmp_field {
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(mut val) = json_object_get_num(raw, 0, sel_field) {
                            for (aop, n) in arith_ops {
                                val = match aop {
                                    BinOp::Add => val + n, BinOp::Sub => val - n,
                                    BinOp::Mul => val * n, BinOp::Div => val / n,
                                    BinOp::Mod => val % n, _ => val,
                                };
                            }
                            let pass = match cmp_op {
                                BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                    compact_buf.extend_from_slice(&raw[vs..ve]);
                                    compact_buf.push(b'\n');
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref sel_op, threshold, ref out_field, ref uop)) = select_cmp_field_unary {
                    // select(.field cmp N) | .output_field | unary_op
                    use jq_jit::ir::{BinOp, UnaryOp};
                    let is_string_op = matches!(uop, UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase);
                    let is_length = matches!(uop, UnaryOp::Length | UnaryOp::Utf8ByteLength);
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                            let pass = match sel_op {
                                BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                if is_length {
                                    // .field | length — for strings, count chars
                                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                        let fval = &raw[vs..ve];
                                        if fval.len() >= 2 && fval[0] == b'"' && fval[fval.len()-1] == b'"' {
                                            let inner = &fval[1..fval.len()-1];
                                            let len = if matches!(uop, UnaryOp::Utf8ByteLength) {
                                                inner.iter().filter(|&&b| b != b'\\').count()
                                            } else if !inner.contains(&b'\\') && inner.iter().all(|&b| b < 0x80) {
                                                inner.len()
                                            } else {
                                                // Need to unescape and count chars
                                                let unescaped = json_unescape_bytes(inner);
                                                if let Ok(s) = std::str::from_utf8(&unescaped) { s.chars().count() } else { unescaped.len() }
                                            };
                                            compact_buf.extend_from_slice(itoa::Buffer::new().format(len).as_bytes());
                                            compact_buf.push(b'\n');
                                        } else if fval[0] == b'[' || fval[0] == b'{' {
                                            // Array/object length — fall back
                                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                        } else if let Some(n) = parse_json_num(fval) {
                                            // Number: fabs
                                            push_jq_number_bytes(&mut compact_buf, n.abs());
                                            compact_buf.push(b'\n');
                                        } else {
                                            compact_buf.extend_from_slice(b"null\n");
                                        }
                                    }
                                } else if is_string_op {
                                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                        let fval = &raw[vs..ve];
                                        if fval.len() >= 2 && fval[0] == b'"' && !fval[1..fval.len()-1].contains(&b'\\') {
                                            compact_buf.push(b'"');
                                            for &b in &fval[1..fval.len()-1] {
                                                compact_buf.push(match uop {
                                                    UnaryOp::AsciiDowncase => if b >= b'A' && b <= b'Z' { b + 32 } else { b },
                                                    UnaryOp::AsciiUpcase => if b >= b'a' && b <= b'z' { b - 32 } else { b },
                                                    _ => b,
                                                });
                                            }
                                            compact_buf.extend_from_slice(b"\"\n");
                                        } else {
                                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                        }
                                    }
                                } else if matches!(uop, UnaryOp::ToString) {
                                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                        let fval = &raw[vs..ve];
                                        if fval[0] == b'"' {
                                            compact_buf.extend_from_slice(fval);
                                            compact_buf.push(b'\n');
                                        } else if let Some(n) = parse_json_num(fval) {
                                            compact_buf.push(b'"');
                                            let i = n as i64;
                                            if i as f64 == n {
                                                compact_buf.extend_from_slice(itoa::Buffer::new().format(i).as_bytes());
                                            } else {
                                                compact_buf.extend_from_slice(ryu::Buffer::new().format(n).as_bytes());
                                            }
                                            compact_buf.extend_from_slice(b"\"\n");
                                        } else {
                                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                        }
                                    }
                                } else {
                                    // Numeric unary (floor/ceil/sqrt/fabs/abs)
                                    if let Some(n) = json_object_get_num(raw, 0, out_field) {
                                        let result = match uop {
                                            UnaryOp::Floor => n.floor(), UnaryOp::Ceil => n.ceil(),
                                            UnaryOp::Sqrt => n.sqrt(), UnaryOp::Fabs | UnaryOp::Abs => n.abs(),
                                            _ => n,
                                        };
                                        push_jq_number_bytes(&mut compact_buf, result);
                                        compact_buf.push(b'\n');
                                    } else {
                                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                    }
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref op, threshold, ref pairs)) = select_cmp_remap {
                    use jq_jit::ir::BinOp;
                    // Collect unique remap source fields (excluding sel_field which is checked via get_num)
                    let remap_fields: Vec<&str> = {
                        let mut v: Vec<&str> = Vec::new();
                        for (_, src) in pairs { if !v.contains(&src.as_str()) { v.push(src.as_str()); } }
                        v
                    };
                    // Pre-compute output indices: for each pair, the index into remap_fields
                    let pair_indices: Vec<usize> = pairs.iter().map(|(_, src)| {
                        remap_fields.iter().position(|&f| f == src.as_str()).unwrap()
                    }).collect();
                    let key_prefixes = if use_pretty_buf {
                        build_obj_key_prefixes_pretty(pairs.iter().map(|(k, _)| k.as_str()))
                    } else {
                        build_obj_key_prefixes(pairs.iter().map(|(k, _)| k.as_str()))
                    };
                    let obj_close: &[u8] = if use_pretty_buf { b"\n}\n" } else { b"}\n" };
                    let mut ranges_buf = vec![(0usize, 0usize); remap_fields.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                            let pass = match op {
                                BinOp::Gt => val > threshold,
                                BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold,
                                BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold,
                                BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                if json_object_get_fields_raw_buf(raw, 0, &remap_fields, &mut ranges_buf) {
                                    for (i, &idx) in pair_indices.iter().enumerate() {
                                        compact_buf.extend_from_slice(&key_prefixes[i]);
                                        let (vs, ve) = ranges_buf[idx];
                                        if use_pretty_buf {
                                            let val_bytes = &raw[vs..ve];
                                            if val_bytes[0] == b'{' || val_bytes[0] == b'[' {
                                                push_json_pretty_raw_at(&mut compact_buf, val_bytes, 2, false, 1);
                                            } else {
                                                compact_buf.extend_from_slice(val_bytes);
                                            }
                                        } else {
                                            compact_buf.extend_from_slice(&raw[vs..ve]);
                                        }
                                    }
                                    compact_buf.extend_from_slice(obj_close);
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref sel_op, threshold, ref cremap)) = select_cmp_cremap {
                    use jq_jit::ir::BinOp;
                    // Collect all unique fields needed (select field + remap fields)
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    // Select field first
                    field_idx.insert(sel_field.clone(), 0);
                    all_fields.push(sel_field.clone());
                    for (_, rexpr) in cremap {
                        for name in remap_expr_fields(rexpr) {
                            if !field_idx.contains_key(name) {
                                field_idx.insert(name.to_string(), all_fields.len());
                                all_fields.push(name.to_string());
                            }
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved = resolve_remap_exprs(cremap, &field_idx);
                    let key_prefixes = if use_pretty_buf {
                        build_obj_key_prefixes_pretty(cremap.iter().map(|(k, _)| k.as_str()))
                    } else {
                        build_obj_key_prefixes(cremap.iter().map(|(k, _)| k.as_str()))
                    };
                    let obj_close: &[u8] = if use_pretty_buf { b"\n}\n" } else { b"}\n" };
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        // Lazy fetch: check select condition first with fast single-field lookup
                        if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                            let pass = match sel_op {
                                BinOp::Gt => val > threshold,
                                BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold,
                                BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold,
                                BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                // Only now extract all remap fields
                                if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                                    for (i, res) in resolved.iter().enumerate() {
                                        compact_buf.extend_from_slice(&key_prefixes[i]);
                                        emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                    }
                                    compact_buf.extend_from_slice(obj_close);
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref sel_op, threshold, ref arr_elems)) = select_cmp_array {
                    use jq_jit::ir::BinOp;
                    // Collect all unique fields needed (select field + array element fields)
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    field_idx.insert(sel_field.clone(), 0);
                    all_fields.push(sel_field.clone());
                    for rexpr in arr_elems {
                        for name in remap_expr_fields(rexpr) {
                            if !field_idx.contains_key(name) {
                                field_idx.insert(name.to_string(), all_fields.len());
                                all_fields.push(name.to_string());
                            }
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                        .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                        .collect();
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                            let pass = match sel_op {
                                BinOp::Gt => val > threshold,
                                BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold,
                                BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold,
                                BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                                    compact_buf.push(b'[');
                                    for (i, res) in resolved.iter().enumerate() {
                                        if i > 0 { compact_buf.push(b','); }
                                        emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                    }
                                    compact_buf.extend_from_slice(b"]\n");
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref arith_ops, ref sel_op, threshold, ref arr_elems)) = select_arith_cmp_array {
                    use jq_jit::ir::BinOp;
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    field_idx.insert(sel_field.clone(), 0);
                    all_fields.push(sel_field.clone());
                    for rexpr in arr_elems {
                        for name in remap_expr_fields(rexpr) {
                            if !field_idx.contains_key(name) {
                                field_idx.insert(name.to_string(), all_fields.len());
                                all_fields.push(name.to_string());
                            }
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                        .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                        .collect();
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(mut val) = json_object_get_num(raw, 0, sel_field) {
                            for (aop, n) in arith_ops {
                                val = match aop {
                                    BinOp::Add => val + n, BinOp::Sub => val - n,
                                    BinOp::Mul => val * n, BinOp::Div => val / n,
                                    BinOp::Mod => if n.is_finite() && *n != 0.0 { val % n } else { val },
                                    _ => val,
                                };
                            }
                            let pass = match sel_op {
                                BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                                    compact_buf.push(b'[');
                                    for (i, res) in resolved.iter().enumerate() {
                                        if i > 0 { compact_buf.push(b','); }
                                        emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                    }
                                    compact_buf.extend_from_slice(b"]\n");
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref sel_op, threshold, ref out_rexpr)) = select_cmp_value {
                    use jq_jit::interpreter::RemapExpr;
                    use jq_jit::ir::BinOp;
                    // Detect fused select+compute opportunities:
                    // 1. FieldOpConst where select field == compute field → reuse val
                    // 2. FieldOpField where select field is one of the two → use get_two_nums
                    // 3. ConstOpField where select field == compute field → reuse val
                    let fused_mode: u8 = match out_rexpr {
                        RemapExpr::FieldOpConst(f, _, _) if f == sel_field => 1, // reuse val
                        RemapExpr::FieldOpField(f1, _, f2) if f1 == sel_field || f2 == sel_field => 2, // two_nums
                        RemapExpr::ConstOpField(_, _, f) if f == sel_field => 3, // reuse val
                        _ => 0, // general
                    };
                    // For mode 2: determine the "other" field (not sel_field)
                    let other_field: Option<&str> = if fused_mode == 2 {
                        match out_rexpr {
                            RemapExpr::FieldOpField(f1, _, f2) => {
                                if f1 == sel_field { Some(f2.as_str()) } else { Some(f1.as_str()) }
                            }
                            _ => None,
                        }
                    } else { None };
                    // Pre-compute fields for fused_mode 0 (general path)
                    let mut gen_all_fields: Vec<String> = Vec::new();
                    let mut gen_field_idx = std::collections::HashMap::new();
                    if fused_mode == 0 {
                        for name in remap_expr_fields(out_rexpr) {
                            if !gen_field_idx.contains_key(name) {
                                gen_field_idx.insert(name.to_string(), gen_all_fields.len());
                                gen_all_fields.push(name.to_string());
                            }
                        }
                    }
                    let gen_field_refs: Vec<&str> = gen_all_fields.iter().map(|s| s.as_str()).collect();
                    let mut ranges_buf = vec![(0usize, 0usize); gen_field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        match fused_mode {
                            1 => {
                                // Fused: select field == FieldOpConst field
                                if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                                    let pass = match sel_op {
                                        BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                        BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                        BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                        _ => false,
                                    };
                                    if pass {
                                        if let RemapExpr::FieldOpConst(_, op, n) = out_rexpr {
                                            let r = match op { BinOp::Add => val + n, BinOp::Sub => val - n, BinOp::Mul => val * n, BinOp::Div => val / n, BinOp::Mod => val % n, _ => unreachable!() };
                                            push_jq_number_bytes(&mut compact_buf, r);
                                        }
                                        compact_buf.push(b'\n');
                                    }
                                }
                            }
                            2 => {
                                // Fused: select field in FieldOpField — single-pass two_nums
                                let of = other_field.unwrap();
                                if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, sel_field, of) {
                                    let pass = match sel_op {
                                        BinOp::Gt => v1 > threshold, BinOp::Lt => v1 < threshold,
                                        BinOp::Ge => v1 >= threshold, BinOp::Le => v1 <= threshold,
                                        BinOp::Eq => v1 == threshold, BinOp::Ne => v1 != threshold,
                                        _ => false,
                                    };
                                    if pass {
                                        if let RemapExpr::FieldOpField(f1, op, _) = out_rexpr {
                                            // v1 is sel_field, v2 is other_field
                                            let (lhs_val, rhs_val) = if f1 == sel_field { (v1, v2) } else { (v2, v1) };
                                            let r = match op { BinOp::Add => lhs_val + rhs_val, BinOp::Sub => lhs_val - rhs_val, BinOp::Mul => lhs_val * rhs_val, BinOp::Div => lhs_val / rhs_val, BinOp::Mod => lhs_val % rhs_val, _ => unreachable!() };
                                            push_jq_number_bytes(&mut compact_buf, r);
                                        }
                                        compact_buf.push(b'\n');
                                    }
                                }
                            }
                            3 => {
                                // Fused: select field == ConstOpField field
                                if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                                    let pass = match sel_op {
                                        BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                        BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                        BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                        _ => false,
                                    };
                                    if pass {
                                        if let RemapExpr::ConstOpField(n, op, _) = out_rexpr {
                                            let r = match op { BinOp::Add => n + val, BinOp::Sub => n - val, BinOp::Mul => n * val, BinOp::Div => n / val, BinOp::Mod => n % val, _ => unreachable!() };
                                            push_jq_number_bytes(&mut compact_buf, r);
                                        }
                                        compact_buf.push(b'\n');
                                    }
                                }
                            }
                            _ => {
                                // General path: extract select field, then output fields
                                if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                                    let pass = match sel_op {
                                        BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                        BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                        BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                        _ => false,
                                    };
                                    if pass {
                                        if json_object_get_fields_raw_buf(raw, 0, &gen_field_refs, &mut ranges_buf) {
                                            emit_remap_value(&mut compact_buf, out_rexpr, raw, &ranges_buf, &gen_field_idx);
                                        } else {
                                            compact_buf.extend_from_slice(b"null");
                                        }
                                        compact_buf.push(b'\n');
                                    }
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sff_f1, sff_op, ref sff_f2, ref sff_out)) = select_ff_cmp_field {
                    // select(.f1 cmp .f2) | .output — field-field comparison select
                    use jq_jit::ir::BinOp;
                    // Optimization: if output field differs from comparison fields,
                    // first get comparison nums cheaply, only fetch output on pass
                    let out_is_separate = sff_out != sff_f1 && sff_out != sff_f2;
                    let mut all_fields: Vec<&str> = Vec::new();
                    let mut idx = std::collections::HashMap::new();
                    for f in [sff_f1.as_str(), sff_f2.as_str(), sff_out.as_str()] {
                        if !idx.contains_key(f) { idx.insert(f, all_fields.len()); all_fields.push(f); }
                    }
                    let mut ranges_buf = vec![(0usize, 0usize); all_fields.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if out_is_separate {
                            // Fast path: first get comparison numbers only
                            if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, sff_f1, sff_f2) {
                                let pass = match sff_op {
                                    BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                    BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                    BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                    _ => false,
                                };
                                if pass {
                                    // Only now fetch the output field
                                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sff_out) {
                                        let out_val = &raw[vs..ve];
                                        if use_pretty_buf && (out_val[0] == b'{' || out_val[0] == b'[') {
                                            push_json_pretty_raw(&mut compact_buf, out_val, 2, false);
                                        } else {
                                            compact_buf.extend_from_slice(out_val);
                                        }
                                        compact_buf.push(b'\n');
                                    }
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else if json_object_get_fields_raw_buf(raw, 0, &all_fields, &mut ranges_buf) {
                            let i1 = idx[sff_f1.as_str()];
                            let i2 = idx[sff_f2.as_str()];
                            if let (Some(v1), Some(v2)) = (
                                parse_json_num(&raw[ranges_buf[i1].0..ranges_buf[i1].1]),
                                parse_json_num(&raw[ranges_buf[i2].0..ranges_buf[i2].1]),
                            ) {
                                let pass = match sff_op {
                                    BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                    BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                    BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                    _ => false,
                                };
                                if pass {
                                    let oi = idx[sff_out.as_str()];
                                    let (vs, ve) = ranges_buf[oi];
                                    let out_val = &raw[vs..ve];
                                    if use_pretty_buf && (out_val[0] == b'{' || out_val[0] == b'[') {
                                        push_json_pretty_raw(&mut compact_buf, out_val, 2, false);
                                    } else {
                                        compact_buf.extend_from_slice(out_val);
                                    }
                                    compact_buf.push(b'\n');
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sff_f1, sff_op, ref sff_f2)) = select_ff_cmp {
                    // select(.f1 cmp .f2) — field-to-field comparison, output whole object
                    use jq_jit::ir::BinOp;
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, sff_f1, sff_f2) {
                            let pass = match sff_op {
                                BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                _ => false,
                            };
                            if pass {
                                emit_raw_ln!(&mut compact_buf, raw);
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sff_f1, ref sff_op, ref sff_f2, ref cremap)) = select_ff_cmp_cremap {
                    // select(.f1 cmp .f2) | {computed_remap} — field-field comparison + computed remap
                    use jq_jit::ir::BinOp;
                    // Collect all unique fields needed (select fields + remap fields)
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    for f in [sff_f1, sff_f2] {
                        if !field_idx.contains_key(f) {
                            field_idx.insert(f.clone(), all_fields.len());
                            all_fields.push(f.clone());
                        }
                    }
                    for (_, rexpr) in cremap {
                        for name in remap_expr_fields(rexpr) {
                            if !field_idx.contains_key(name) {
                                field_idx.insert(name.to_string(), all_fields.len());
                                all_fields.push(name.to_string());
                            }
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved = resolve_remap_exprs(cremap, &field_idx);
                    let f1_idx = field_idx[sff_f1];
                    let f2_idx = field_idx[sff_f2];
                    let key_prefixes = if use_pretty_buf {
                        build_obj_key_prefixes_pretty(cremap.iter().map(|(k, _)| k.as_str()))
                    } else {
                        build_obj_key_prefixes(cremap.iter().map(|(k, _)| k.as_str()))
                    };
                    let obj_close: &[u8] = if use_pretty_buf { b"\n}\n" } else { b"}\n" };
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            // Parse both comparison fields as numbers
                            let r1 = &ranges_buf[f1_idx];
                            let r2 = &ranges_buf[f2_idx];
                            if r1.0 < r1.1 && r2.0 < r2.1 {
                                if let (Some(v1), Some(v2)) = (
                                    parse_json_num(&raw[r1.0..r1.1]),
                                    parse_json_num(&raw[r2.0..r2.1]),
                                ) {
                                    let pass = match sff_op {
                                        BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                        BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                        BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                        _ => false,
                                    };
                                    if pass {
                                        for (i, res) in resolved.iter().enumerate() {
                                            compact_buf.extend_from_slice(&key_prefixes[i]);
                                            emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                        }
                                        compact_buf.extend_from_slice(obj_close);
                                    }
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sff_f1, ref sff_op, ref sff_f2, ref out_rexpr)) = select_ff_cmp_value {
                    // select(.f1 cmp .f2) | value — field-field select + computed value output
                    use jq_jit::ir::BinOp;
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    for f in [sff_f1, sff_f2] {
                        if !field_idx.contains_key(f) {
                            field_idx.insert(f.clone(), all_fields.len());
                            all_fields.push(f.clone());
                        }
                    }
                    for name in remap_expr_fields(out_rexpr) {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved = resolve_one_remap(out_rexpr, &field_idx);
                    let f1_idx = field_idx[sff_f1];
                    let f2_idx = field_idx[sff_f2];
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            let r1 = &ranges_buf[f1_idx];
                            let r2 = &ranges_buf[f2_idx];
                            if r1.0 < r1.1 && r2.0 < r2.1 {
                                if let (Some(v1), Some(v2)) = (
                                    parse_json_num(&raw[r1.0..r1.1]),
                                    parse_json_num(&raw[r2.0..r2.1]),
                                ) {
                                    let pass = match sff_op {
                                        BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                        BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                        BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                        _ => false,
                                    };
                                    if pass {
                                        emit_resolved_value(&mut compact_buf, &resolved, raw, &ranges_buf);
                                        compact_buf.push(b'\n');
                                    }
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sff_f1, ref sff_op, ref sff_f2, ref arr_elems)) = select_ff_cmp_array {
                    // select(.f1 cmp .f2) | [array]
                    use jq_jit::ir::BinOp;
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    for f in [sff_f1, sff_f2] {
                        if !field_idx.contains_key(f) {
                            field_idx.insert(f.clone(), all_fields.len());
                            all_fields.push(f.clone());
                        }
                    }
                    for rexpr in arr_elems {
                        for name in remap_expr_fields(rexpr) {
                            if !field_idx.contains_key(name) {
                                field_idx.insert(name.to_string(), all_fields.len());
                                all_fields.push(name.to_string());
                            }
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                        .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                        .collect();
                    let f1_idx = field_idx[sff_f1];
                    let f2_idx = field_idx[sff_f2];
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            let r1 = &ranges_buf[f1_idx];
                            let r2 = &ranges_buf[f2_idx];
                            if r1.0 < r1.1 && r2.0 < r2.1 {
                                if let (Some(v1), Some(v2)) = (
                                    parse_json_num(&raw[r1.0..r1.1]),
                                    parse_json_num(&raw[r2.0..r2.1]),
                                ) {
                                    let pass = match sff_op {
                                        BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                        BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                        BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                        _ => false,
                                    };
                                    if pass {
                                        compact_buf.push(b'[');
                                        for (i, res) in resolved.iter().enumerate() {
                                            if i > 0 { compact_buf.push(b','); }
                                            emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                        }
                                        compact_buf.extend_from_slice(b"]\n");
                                    }
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref conj, ref cmps, ref arr_elems)) = select_compound_array {
                    // select(.x > N and .y < M) | [array]
                    use jq_jit::ir::BinOp;
                    let mut all_fields: Vec<String> = Vec::new();
                    let mut field_idx = std::collections::HashMap::new();
                    for (f, _, _) in cmps {
                        if !field_idx.contains_key(f) {
                            field_idx.insert(f.clone(), all_fields.len());
                            all_fields.push(f.clone());
                        }
                    }
                    for rexpr in arr_elems {
                        for name in remap_expr_fields(rexpr) {
                            if !field_idx.contains_key(name) {
                                field_idx.insert(name.to_string(), all_fields.len());
                                all_fields.push(name.to_string());
                            }
                        }
                    }
                    let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                        .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                        .collect();
                    let resolved_cmps: Vec<(usize, BinOp, f64)> = cmps.iter()
                        .map(|(f, op, n)| (field_idx[f], *op, *n))
                        .collect();
                    let is_and = matches!(conj, BinOp::And);
                    let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            let mut pass = is_and;
                            for (idx, op, threshold) in &resolved_cmps {
                                let r = &ranges_buf[*idx];
                                let cmp_pass = if r.0 < r.1 {
                                    if let Some(v) = parse_json_num(&raw[r.0..r.1]) {
                                        match op {
                                            BinOp::Gt => v > *threshold, BinOp::Lt => v < *threshold,
                                            BinOp::Ge => v >= *threshold, BinOp::Le => v <= *threshold,
                                            BinOp::Eq => v == *threshold, BinOp::Ne => v != *threshold,
                                            _ => false,
                                        }
                                    } else { false }
                                } else { false };
                                if is_and {
                                    if !cmp_pass { pass = false; break; }
                                } else {
                                    if cmp_pass { pass = true; break; }
                                }
                            }
                            if pass {
                                compact_buf.push(b'[');
                                for (i, res) in resolved.iter().enumerate() {
                                    if i > 0 { compact_buf.push(b','); }
                                    emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                }
                                compact_buf.extend_from_slice(b"]\n");
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref test_type, ref test_arg, ref out_field)) = select_str_field {
                    let expected_eq = if test_type == "eq" || test_type == "ne" {
                        let mut e = Vec::with_capacity(test_arg.len() + 2);
                        e.push(b'"'); e.extend_from_slice(test_arg.as_bytes()); e.push(b'"');
                        Some(e)
                    } else { None };
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        let pass = if let Some(ref expected) = expected_eq {
                            // eq/ne test
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                                let val_bytes = &raw[vs..ve];
                                let m = val_bytes == expected.as_slice();
                                if test_type == "eq" { m } else { !m }
                            } else { false }
                        } else {
                            // startswith/endswith/contains
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                                let val = &raw[vs..ve];
                                if val.len() >= 2 && val[0] == b'"' && val[ve-vs-1] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                    let inner = &val[1..ve-vs-1];
                                    match test_type.as_str() {
                                        "startswith" => inner.starts_with(test_arg.as_bytes()),
                                        "endswith" => inner.ends_with(test_arg.as_bytes()),
                                        "contains" => bytes_contains(inner, test_arg.as_bytes()),
                                        _ => false,
                                    }
                                } else { false }
                            } else { false }
                        };
                        if pass {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                let val = &raw[vs..ve];
                                if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                    push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                } else {
                                    compact_buf.extend_from_slice(val);
                                }
                                compact_buf.push(b'\n');
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref chain_field, ref chain_ops, ref chain_terminal)) = field_string_chain {
                    use jq_jit::interpreter::{StringChainOp, StringChainTerminal};
                    let mut tmp_str = Vec::<u8>::new();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, chain_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && !val[1..val.len()-1].contains(&b'\\')
                            {
                                tmp_str.clear();
                                tmp_str.extend_from_slice(&val[1..val.len()-1]);
                                for op in chain_ops {
                                    match op {
                                        StringChainOp::AsciiDowncase => {
                                            for b in tmp_str.iter_mut() {
                                                if *b >= b'A' && *b <= b'Z' { *b += 32; }
                                            }
                                        }
                                        StringChainOp::AsciiUpcase => {
                                            for b in tmp_str.iter_mut() {
                                                if *b >= b'a' && *b <= b'z' { *b -= 32; }
                                            }
                                        }
                                        StringChainOp::Ltrimstr(ref prefix) => {
                                            let pb = prefix.as_bytes();
                                            if tmp_str.len() >= pb.len() && &tmp_str[..pb.len()] == pb {
                                                tmp_str.drain(..pb.len());
                                            }
                                        }
                                        StringChainOp::Rtrimstr(ref suffix) => {
                                            let sb = suffix.as_bytes();
                                            if tmp_str.len() >= sb.len() && &tmp_str[tmp_str.len()-sb.len()..] == sb {
                                                let new_len = tmp_str.len() - sb.len();
                                                tmp_str.truncate(new_len);
                                            }
                                        }
                                    }
                                }
                                // Apply terminal
                                match chain_terminal {
                                    StringChainTerminal::None => {
                                        compact_buf.push(b'"');
                                        compact_buf.extend_from_slice(&tmp_str);
                                        compact_buf.extend_from_slice(b"\"\n");
                                    }
                                    StringChainTerminal::Startswith(ref arg) => {
                                        let ab = arg.as_bytes();
                                        if tmp_str.len() >= ab.len() && &tmp_str[..ab.len()] == ab {
                                            compact_buf.extend_from_slice(b"true\n");
                                        } else {
                                            compact_buf.extend_from_slice(b"false\n");
                                        }
                                    }
                                    StringChainTerminal::Endswith(ref arg) => {
                                        let ab = arg.as_bytes();
                                        if tmp_str.len() >= ab.len() && &tmp_str[tmp_str.len()-ab.len()..] == ab {
                                            compact_buf.extend_from_slice(b"true\n");
                                        } else {
                                            compact_buf.extend_from_slice(b"false\n");
                                        }
                                    }
                                    StringChainTerminal::Contains(ref arg) => {
                                        let ab = arg.as_bytes();
                                        if bytes_contains(&tmp_str, ab) {
                                            compact_buf.extend_from_slice(b"true\n");
                                        } else {
                                            compact_buf.extend_from_slice(b"false\n");
                                        }
                                    }
                                    StringChainTerminal::Length => {
                                        let count = tmp_str.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
                                        push_jq_number_bytes(&mut compact_buf, count as f64);
                                        compact_buf.push(b'\n');
                                    }
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref test_type, ref test_arg, ref remap_pairs)) = select_str_cremap {
                    // select(.field | string_test) | {computed_remap}
                    let mut all_fields: Vec<String> = Vec::new();
                    all_fields.push(sel_field.clone());
                    for (_, rexpr) in remap_pairs {
                        for f in remap_expr_fields(rexpr) {
                            if !all_fields.iter().any(|x| x == f) {
                                all_fields.push(f.to_string());
                            }
                        }
                    }
                    let field_strs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let mut field_idx = std::collections::HashMap::new();
                    for (i, f) in all_fields.iter().enumerate() { field_idx.insert(f.clone(), i); }
                    let resolved = resolve_remap_exprs(remap_pairs, &field_idx);
                    let obj_keys: Vec<&str> = remap_pairs.iter().map(|(k, _)| k.as_str()).collect();
                    let key_prefixes = if use_pretty_buf {
                        build_obj_key_prefixes_pretty(obj_keys.iter().copied())
                    } else {
                        build_obj_key_prefixes(obj_keys.iter().copied())
                    };
                    let obj_close: &[u8] = if use_pretty_buf { b"\n}\n" } else { b"}\n" };
                    let expected_eq = if test_type == "eq" || test_type == "ne" {
                        let mut e = Vec::with_capacity(test_arg.len() + 2);
                        e.push(b'"'); e.extend_from_slice(test_arg.as_bytes()); e.push(b'"');
                        Some(e)
                    } else { None };
                    let mut ranges = vec![(0usize, 0usize); all_fields.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        // Check string condition
                        let pass = if let Some(ref expected) = expected_eq {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                                let val_bytes = &raw[vs..ve];
                                let m = val_bytes == expected.as_slice();
                                if test_type == "eq" { m } else { !m }
                            } else { false }
                        } else {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                                let val = &raw[vs..ve];
                                if val.len() >= 2 && val[0] == b'"' && val[ve-vs-1] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                    let inner = &val[1..ve-vs-1];
                                    match test_type.as_str() {
                                        "startswith" => inner.starts_with(test_arg.as_bytes()),
                                        "endswith" => inner.ends_with(test_arg.as_bytes()),
                                        "contains" => bytes_contains(inner, test_arg.as_bytes()),
                                        _ => false,
                                    }
                                } else { false }
                            } else { false }
                        };
                        if pass {
                            if json_object_get_fields_raw_buf(raw, 0, &field_strs, &mut ranges) {
                                for (i, (prefix, res)) in key_prefixes.iter().zip(resolved.iter()).enumerate() {
                                    compact_buf.extend_from_slice(prefix);
                                    emit_resolved_value(&mut compact_buf, res, raw, &ranges);
                                    let _ = i;
                                }
                                compact_buf.extend_from_slice(obj_close);
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref sel_field, ref test_type, ref test_arg, ref arr_elems)) = select_str_array {
                    // select(.field | string_test) | [array]
                    let mut all_fields: Vec<String> = Vec::new();
                    all_fields.push(sel_field.clone());
                    for rexpr in arr_elems {
                        for f in remap_expr_fields(rexpr) {
                            if !all_fields.iter().any(|x| x == f) {
                                all_fields.push(f.to_string());
                            }
                        }
                    }
                    let field_strs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                    let mut field_idx = std::collections::HashMap::new();
                    for (i, f) in all_fields.iter().enumerate() { field_idx.insert(f.clone(), i); }
                    let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                        .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                        .collect();
                    let expected_eq = if test_type == "eq" || test_type == "ne" {
                        let mut e = Vec::with_capacity(test_arg.len() + 2);
                        e.push(b'"'); e.extend_from_slice(test_arg.as_bytes()); e.push(b'"');
                        Some(e)
                    } else { None };
                    let mut ranges = vec![(0usize, 0usize); all_fields.len()];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        let pass = if let Some(ref expected) = expected_eq {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                                let val_bytes = &raw[vs..ve];
                                let m = val_bytes == expected.as_slice();
                                if test_type == "eq" { m } else { !m }
                            } else { false }
                        } else {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                                let val = &raw[vs..ve];
                                if val.len() >= 2 && val[0] == b'"' && val[ve-vs-1] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                    let inner = &val[1..ve-vs-1];
                                    match test_type.as_str() {
                                        "startswith" => inner.starts_with(test_arg.as_bytes()),
                                        "endswith" => inner.ends_with(test_arg.as_bytes()),
                                        "contains" => bytes_contains(inner, test_arg.as_bytes()),
                                        _ => false,
                                    }
                                } else { false }
                            } else { false }
                        };
                        if pass {
                            if json_object_get_fields_raw_buf(raw, 0, &field_strs, &mut ranges) {
                                compact_buf.push(b'[');
                                for (i, res) in resolved.iter().enumerate() {
                                    if i > 0 { compact_buf.push(b','); }
                                    emit_resolved_value(&mut compact_buf, res, raw, &ranges);
                                }
                                compact_buf.extend_from_slice(b"]\n");
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_length {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(len) = json_value_length(raw, 0) {
                            push_jq_number_bytes(&mut compact_buf, len as f64);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_keys {
                    let mut tmp = Vec::new();
                    let mut keys_buf: Vec<(usize, usize)> = Vec::new();
                    // Cache sorted output for identical key sets
                    let mut cached_output: Vec<u8> = Vec::new();
                    let mut cached_keys: Vec<Vec<u8>> = Vec::new(); // unsorted key bytes
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        // Try cached permutation: extract unsorted keys, check if same set
                        if !cached_keys.is_empty() {
                            if let Some(extracted) = json_object_extract_keys_only(raw, 0, &mut keys_buf) {
                                if extracted == cached_keys.len() {
                                    // Check if all keys match
                                    let mut same = true;
                                    for (i, (ks, ke)) in keys_buf.iter().enumerate() {
                                        if &raw[*ks..*ke] != cached_keys[i].as_slice() {
                                            same = false;
                                            break;
                                        }
                                    }
                                    if same {
                                        if use_pretty_buf {
                                            // cached_output has no trailing \n
                                            push_json_pretty_raw(&mut compact_buf, &cached_output, 2, false);
                                            compact_buf.push(b'\n');
                                        } else {
                                            compact_buf.extend_from_slice(&cached_output);
                                            compact_buf.push(b'\n');
                                        }
                                        if compact_buf.len() >= 1 << 17 {
                                            let _ = out.write_all(&compact_buf);
                                            compact_buf.clear();
                                        }
                                        return Ok(());
                                    }
                                }
                            }
                        }
                        // Full path: extract, sort, output
                        if use_pretty_buf {
                            tmp.clear();
                            if json_object_keys_to_buf_reuse(raw, 0, &mut tmp, &mut keys_buf) {
                                let len = tmp.len();
                                if len > 0 && tmp[len-1] == b'\n' { tmp.truncate(len-1); }
                                // Cache for future objects
                                if cached_keys.is_empty() {
                                    cached_output = tmp.clone();
                                    // We need to re-extract unsorted keys for caching
                                    // keys_buf was sorted in-place, so re-extract
                                    let mut unsorted: Vec<(usize, usize)> = Vec::new();
                                    if let Some(_) = json_object_extract_keys_only(raw, 0, &mut unsorted) {
                                        cached_keys = unsorted.iter().map(|(s, e)| raw[*s..*e].to_vec()).collect();
                                    }
                                }
                                push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let before = compact_buf.len();
                            if json_object_keys_to_buf_reuse(raw, 0, &mut compact_buf, &mut keys_buf) {
                                // Cache the output (without trailing \n)
                                if cached_keys.is_empty() {
                                    let end_pos = compact_buf.len();
                                    // Output is [...]\n, cache [...] (without \n)
                                    cached_output = compact_buf[before..end_pos-1].to_vec();
                                    let mut unsorted: Vec<(usize, usize)> = Vec::new();
                                    if let Some(_) = json_object_extract_keys_only(raw, 0, &mut unsorted) {
                                        cached_keys = unsorted.iter().map(|(s, e)| raw[*s..*e].to_vec()).collect();
                                    }
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_keys_unsorted {
                    let mut tmp = Vec::new();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if use_pretty_buf {
                            tmp.clear();
                            if json_object_keys_unsorted_to_buf(raw, 0, &mut tmp) {
                                let len = tmp.len();
                                if len > 0 && tmp[len-1] == b'\n' { tmp.truncate(len-1); }
                                push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else if !json_object_keys_unsorted_to_buf(raw, 0, &mut compact_buf) {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref hf) = has_field {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if let Some(found) = json_object_has_key(raw, 0, hf) {
                            compact_buf.extend_from_slice(if found { b"true\n" } else { b"false\n" });
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref hm_fields, hm_is_and)) = has_multi {
                    let field_refs: Vec<&str> = hm_fields.iter().map(|s| s.as_str()).collect();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        let result = if hm_is_and {
                            json_object_has_all_keys(raw, 0, &field_refs)
                        } else {
                            json_object_has_any_key(raw, 0, &field_refs)
                        };
                        if let Some(found) = result {
                            compact_buf.extend_from_slice(if found { b"true\n" } else { b"false\n" });
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_type {
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        compact_buf.extend_from_slice(json_type_byte(raw[0]));
                        compact_buf.push(b'\n');
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref df) = del_field {
                    let mut tmp = Vec::new();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if use_pretty_buf {
                            tmp.clear();
                            if json_object_del_field(raw, 0, df, &mut tmp) {
                                push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else if json_object_del_field(raw, 0, df, &mut compact_buf) {
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref merge_pairs) = obj_merge_lit {
                    let mut tmp = Vec::new();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if use_pretty_buf {
                            tmp.clear();
                            if json_object_merge_literal(raw, 0, merge_pairs, &mut tmp) {
                                push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else if json_object_merge_literal(raw, 0, merge_pairs, &mut compact_buf) {
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref out_key, ref nfields, ref arith)) = obj_merge_computed {
                    let mut tmp = Vec::new();
                    let mut merge_pair = vec![(out_key.clone(), Vec::new())];
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        let nf_count = nfields.len();
                        let ok = if nf_count == 1 {
                            if let Some(v) = json_object_get_num(raw, 0, &nfields[0]) {
                                let result = arith.eval(&[v]);
                                merge_pair[0].1.clear();
                                push_jq_number_bytes(&mut merge_pair[0].1, result);
                                true
                            } else { false }
                        } else if nf_count == 2 {
                            if let Some((a, b)) = json_object_get_two_nums(raw, 0, &nfields[0], &nfields[1]) {
                                let result = arith.eval(&[a, b]);
                                merge_pair[0].1.clear();
                                push_jq_number_bytes(&mut merge_pair[0].1, result);
                                true
                            } else { false }
                        } else { false };
                        if ok {
                            if use_pretty_buf {
                                tmp.clear();
                                if json_object_merge_literal(raw, 0, &merge_pair, &mut tmp) {
                                    push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                    compact_buf.push(b'\n');
                                } else {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            } else if json_object_merge_literal(raw, 0, &merge_pair, &mut compact_buf) {
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_each {
                    if use_pretty_buf {
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            if !json_each_value_cb(raw, 0, |vs, ve| {
                                let val = &raw[vs..ve];
                                emit_raw_ln!(&mut compact_buf, val);
                            }) {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                            Ok(())
                        })
                    } else {
                        json_stream_raw(&input_str, |start, end| {
                            let raw = &input_bytes[start..end];
                            if !json_each_value_raw(raw, 0, &mut compact_buf) {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                            Ok(())
                        })
                    }
                } else if is_sort_keys {
                    let mut pairs_buf: Vec<(usize, usize, usize, usize)> = Vec::new();
                    let mut tmp = Vec::new();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if use_pretty_buf {
                            tmp.clear();
                            if json_object_sort_keys(raw, 0, &mut tmp, &mut pairs_buf) {
                                push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            if json_object_sort_keys(raw, 0, &mut compact_buf, &mut pairs_buf) {
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_to_entries {
                    let mut tmp = Vec::new();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if use_pretty_buf {
                            tmp.clear();
                            if json_to_entries_raw(raw, 0, &mut tmp) {
                                let len = tmp.len();
                                if len > 0 && tmp[len-1] == b'\n' { tmp.truncate(len-1); }
                                push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else if !json_to_entries_raw(raw, 0, &mut compact_buf) {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref rte_pairs) = remap_to_entries {
                    // {k1:.f1, k2:.f2} | to_entries → emit entries array from raw fields
                    let rte_src: Vec<&str> = rte_pairs.iter().map(|(_, src)| src.as_str()).collect();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        // Extract all source field values
                        let mut vals: Vec<Option<(usize, usize)>> = Vec::with_capacity(rte_src.len());
                        for src in &rte_src {
                            vals.push(json_object_get_field_raw(raw, 0, src));
                        }
                        compact_buf.push(b'[');
                        let mut first = true;
                        for (i, (out_key, _)) in rte_pairs.iter().enumerate() {
                            if !first { compact_buf.push(b','); }
                            first = false;
                            compact_buf.extend_from_slice(b"{\"key\":");
                            // Write key as JSON string
                            compact_buf.push(b'"');
                            for &b in out_key.as_bytes() {
                                match b {
                                    b'"' => compact_buf.extend_from_slice(b"\\\""),
                                    b'\\' => compact_buf.extend_from_slice(b"\\\\"),
                                    _ => compact_buf.push(b),
                                }
                            }
                            compact_buf.extend_from_slice(b"\",\"value\":");
                            if let Some((vs, ve)) = vals[i] {
                                compact_buf.extend_from_slice(&raw[vs..ve]);
                            } else {
                                compact_buf.extend_from_slice(b"null");
                            }
                            compact_buf.push(b'}');
                        }
                        compact_buf.extend_from_slice(b"]\n");
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some((ref we_op, we_threshold)) = with_entries_select {
                    // with_entries(select(.value CMP N)) → filter object entries by value
                    use jq_jit::ir::BinOp;
                    let cmp_byte = match we_op {
                        BinOp::Gt => b'>',
                        BinOp::Ge => b'G',
                        BinOp::Lt => b'<',
                        BinOp::Le => b'L',
                        BinOp::Eq => b'=',
                        BinOp::Ne => b'!',
                        _ => b'>',
                    };
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        json_with_entries_select_value_cmp(raw, 0, cmp_byte, we_threshold, &mut compact_buf);
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref type_name) = with_entries_type {
                    // with_entries(select(.value | type == "type_name")) — filter by value type
                    let mut tmp = Vec::new();
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        if use_pretty_buf {
                            tmp.clear();
                            if json_object_filter_by_value_type(raw, 0, type_name, &mut tmp) {
                                push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else if json_object_filter_by_value_type(raw, 0, type_name, &mut compact_buf) {
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if is_tojson {
                    // tojson: single-pass compact + escape
                    json_stream_raw(&input_str, |start, end| {
                        let raw = &input_bytes[start..end];
                        push_tojson_raw(&mut compact_buf, raw);
                        compact_buf.push(b'\n');
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else if let Some(ref pf) = projection_fields {
                    let field_refs: Vec<&str> = pf.iter().map(|s| s.as_str()).collect();
                    json_stream_project(&input_str, &field_refs, |v| {
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        pool_value(v);
                        Ok(())
                    })
                } else if use_compact_buf {
                    json_stream_offsets(&input_str, |v, start, end| {
                        let raw = &input_bytes[start..end];
                        process_input(&v, Some(raw), &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        pool_value(v);
                        Ok(())
                    })
                } else {
                    json_stream(&input_str, |v| {
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        pool_value(v);
                        Ok(())
                    })
                };
                if let Err(e) = parse_result {
                    eprintln!("jq: error (at <stdin>:0): {}", e);
                    process::exit(2);
                }
            }
        }
    } else {
        // Process files
        let mut slurp_values: Vec<Value> = Vec::new();
        for file in &files {
            let f = match std::fs::File::open(file) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("jq: error: Could not open file {}: {}", file, e);
                    process::exit(2);
                }
            };
            let meta = f.metadata().unwrap();
            // Memory-map files to avoid heap allocation for file content
            let (mmap, content);
            if meta.len() > 0 {
                mmap = Some(unsafe { memmap2::Mmap::map(&f) }.unwrap_or_else(|e| {
                    eprintln!("jq: error: Could not mmap file {}: {}", file, e);
                    process::exit(2);
                }));
                // SAFETY: JSON is defined as UTF-8. Our parser validates structure
                // byte-by-byte, so we skip the upfront O(n) UTF-8 validation which
                // costs ~40% of total runtime on large files.
                content = unsafe { std::str::from_utf8_unchecked(mmap.as_ref().unwrap()) };
            } else {
                mmap = None;
                content = "";
            }
            let _ = &mmap; // keep mmap alive
            let parse_result = if slurp {
                // Slurp: collect all JSON values into an array, process once
                let mut values = Vec::new();
                let r = if raw_input {
                    for line in content.lines() {
                        values.push(Value::from_str(line));
                    }
                    Ok(())
                } else {
                    json_stream(content, |v| {
                        values.push(v);
                        Ok(())
                    })
                };
                if let Err(e) = r {
                    eprintln!("jq: error (at {}:0): {}", file, e);
                    process::exit(2);
                }
                slurp_values.extend(values);
                Ok(())
            } else if filter.is_empty() {
                // Empty fast path: just validate JSON structure, produce no output.
                json_stream_raw(content, |_, _| Ok(()))
            } else if let Some(ref lit) = literal_output {
                json_stream_raw(content, |_, _| {
                    compact_buf.extend_from_slice(lit);
                    compact_buf.push(b'\n');
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref aff_fields, ref aff_format)) = array_fields_format {
                let content_bytes = content.as_bytes();
                let aff_refs: Vec<&str> = aff_fields.iter().map(|s| s.as_str()).collect();
                let is_csv = aff_format == "csv";
                let mut ranges_buf = vec![(0usize, 0usize); aff_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &aff_refs, &mut ranges_buf) {
                        let mut has_escapes = false;
                        for (vs, ve) in &ranges_buf {
                            let val = &raw[*vs..*ve];
                            if val[0] == b'"' && val[1..val.len()-1].contains(&b'\\') {
                                has_escapes = true;
                                break;
                            }
                        }
                        if has_escapes {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        } else {
                            compact_buf.push(b'"');
                            for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                if i > 0 {
                                    if is_csv { compact_buf.push(b','); }
                                    else { compact_buf.extend_from_slice(b"\\t"); }
                                }
                                let val = &raw[*vs..*ve];
                                if val[0] == b'"' {
                                    let inner = &val[1..val.len()-1];
                                    if is_csv {
                                        compact_buf.extend_from_slice(b"\\\"");
                                        compact_buf.extend_from_slice(inner);
                                        compact_buf.extend_from_slice(b"\\\"");
                                    } else {
                                        compact_buf.extend_from_slice(inner);
                                    }
                                } else if val == b"null" {
                                    // null → empty
                                } else {
                                    compact_buf.extend_from_slice(val);
                                }
                            }
                            compact_buf.push(b'"');
                            compact_buf.push(b'\n');
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref rcf_fields, ref rcf_format)) = raw_csv_fields {
                let content_bytes = content.as_bytes();
                let rcf_refs: Vec<&str> = rcf_fields.iter().map(|s| s.as_str()).collect();
                let is_csv = rcf_format == "csv";
                let sep = if is_csv { b',' } else { b'\t' };
                let mut ranges_buf = vec![(0usize, 0usize); rcf_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &rcf_refs, &mut ranges_buf) {
                        for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                            if i > 0 { compact_buf.push(sep); }
                            let val = &raw[*vs..*ve];
                            if val[0] == b'"' {
                                let inner = &val[1..val.len()-1];
                                if is_csv {
                                    compact_buf.push(b'"');
                                    if inner.contains(&b'\\') {
                                        let decoded = json_unescape_bytes(inner);
                                        for &b in &decoded {
                                            if b == b'"' { compact_buf.push(b'"'); }
                                            compact_buf.push(b);
                                        }
                                    } else if inner.contains(&b'"') {
                                        for &b in inner.iter() {
                                            if b == b'"' { compact_buf.push(b'"'); }
                                            compact_buf.push(b);
                                        }
                                    } else {
                                        compact_buf.extend_from_slice(inner);
                                    }
                                    compact_buf.push(b'"');
                                } else {
                                    if inner.contains(&b'\\') {
                                        compact_buf.extend_from_slice(&json_unescape_bytes(inner));
                                    } else {
                                        compact_buf.extend_from_slice(inner);
                                    }
                                }
                            } else if val == b"null" {
                                // null → empty
                            } else if val == b"true" {
                                compact_buf.extend_from_slice(b"true");
                            } else if val == b"false" {
                                compact_buf.extend_from_slice(b"false");
                            } else {
                                compact_buf.extend_from_slice(val);
                            }
                        }
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref rev_field) = field_str_reverse {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, rev_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"' {
                            let inner = &val[1..val.len()-1];
                            if !inner.contains(&b'\\') && inner.iter().all(|&b| b < 0x80) {
                                compact_buf.push(b'"');
                                for &b in inner.iter().rev() {
                                    compact_buf.push(b);
                                }
                                compact_buf.extend_from_slice(b"\"\n");
                            } else {
                                let unescaped = json_unescape_bytes(inner);
                                if let Ok(s) = std::str::from_utf8(&unescaped) {
                                    let reversed: String = s.chars().rev().collect();
                                    compact_buf.push(b'"');
                                    compact_buf.extend_from_slice(&json_escape_bytes(reversed.as_bytes()));
                                    compact_buf.extend_from_slice(b"\"\n");
                                } else {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref fsj_field, ref fsj_split, ref fsj_join)) = field_split_join {
                let content_bytes = content.as_bytes();
                let split_bytes = fsj_split.as_bytes();
                let single_split = if split_bytes.len() == 1 { Some(split_bytes[0]) } else { None };
                let escaped_join = json_escape_bytes(fsj_join.as_bytes());
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, fsj_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && memchr::memchr(b'\\', &val[1..val.len()-1]).is_none()
                            && !split_bytes.is_empty()
                        {
                            let inner = &val[1..val.len()-1];
                            compact_buf.push(b'"');
                            let mut pos = 0;
                            let mut first = true;
                            loop {
                                let rest = &inner[pos..];
                                let found = if let Some(d) = single_split {
                                    memchr::memchr(d, rest)
                                } else {
                                    rest.windows(split_bytes.len()).position(|w| w == split_bytes)
                                };
                                if let Some(idx) = found {
                                    if !first { compact_buf.extend_from_slice(&escaped_join); }
                                    first = false;
                                    compact_buf.extend_from_slice(&rest[..idx]);
                                    pos += idx + split_bytes.len();
                                } else {
                                    if !first { compact_buf.extend_from_slice(&escaped_join); }
                                    compact_buf.extend_from_slice(rest);
                                    break;
                                }
                            }
                            compact_buf.push(b'"');
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sf_field, ref sf_delim)) = field_split_first {
                let content_bytes = content.as_bytes();
                let delim_bytes = sf_delim.as_bytes();
                let single_delim = if delim_bytes.len() == 1 { Some(delim_bytes[0]) } else { None };
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sf_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && memchr::memchr(b'\\', &val[1..val.len()-1]).is_none()
                            && !delim_bytes.is_empty()
                        {
                            let inner = &val[1..val.len()-1];
                            compact_buf.push(b'"');
                            let split_pos = if let Some(d) = single_delim {
                                memchr::memchr(d, inner)
                            } else {
                                inner.windows(delim_bytes.len()).position(|w| w == delim_bytes)
                            };
                            if let Some(idx) = split_pos {
                                compact_buf.extend_from_slice(&inner[..idx]);
                            } else {
                                compact_buf.extend_from_slice(inner);
                            }
                            compact_buf.push(b'"');
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sl_field, ref sl_delim)) = field_split_last {
                let content_bytes = content.as_bytes();
                let delim_bytes = sl_delim.as_bytes();
                let single_delim = if delim_bytes.len() == 1 { Some(delim_bytes[0]) } else { None };
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sl_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && memchr::memchr(b'\\', &val[1..val.len()-1]).is_none()
                            && !delim_bytes.is_empty()
                        {
                            let inner = &val[1..val.len()-1];
                            compact_buf.push(b'"');
                            let split_pos = if let Some(d) = single_delim {
                                memchr::memrchr(d, inner)
                            } else {
                                let mut last_pos = None;
                                let mut pos = 0;
                                while pos + delim_bytes.len() <= inner.len() {
                                    if &inner[pos..pos+delim_bytes.len()] == delim_bytes {
                                        last_pos = Some(pos);
                                    }
                                    pos += 1;
                                }
                                last_pos
                            };
                            if let Some(idx) = split_pos {
                                compact_buf.extend_from_slice(&inner[idx + delim_bytes.len()..]);
                            } else {
                                compact_buf.extend_from_slice(inner);
                            }
                            compact_buf.push(b'"');
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sn_field, ref sn_delim, sn_idx)) = field_split_nth {
                let content_bytes = content.as_bytes();
                let delim_bytes = sn_delim.as_bytes();
                let single_delim = if delim_bytes.len() == 1 { Some(delim_bytes[0]) } else { None };
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sn_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && memchr::memchr(b'\\', &val[1..val.len()-1]).is_none()
                            && !delim_bytes.is_empty()
                        {
                            let inner = &val[1..val.len()-1];
                            let segments: Vec<&[u8]> = if let Some(d) = single_delim {
                                inner.split(|&b| b == d).collect()
                            } else {
                                let mut segs = Vec::new();
                                let mut pos = 0;
                                while pos <= inner.len() {
                                    if let Some(p) = inner[pos..].windows(delim_bytes.len()).position(|w| w == delim_bytes) {
                                        segs.push(&inner[pos..pos+p]);
                                        pos = pos + p + delim_bytes.len();
                                    } else {
                                        segs.push(&inner[pos..]);
                                        break;
                                    }
                                }
                                segs
                            };
                            let actual_idx = if sn_idx >= 0 {
                                sn_idx as usize
                            } else {
                                let abs = (-sn_idx) as usize;
                                if abs <= segments.len() { segments.len() - abs } else { segments.len() }
                            };
                            if actual_idx < segments.len() {
                                compact_buf.push(b'"');
                                compact_buf.extend_from_slice(segments[actual_idx]);
                                compact_buf.push(b'"');
                                compact_buf.push(b'\n');
                            } else {
                                compact_buf.extend_from_slice(b"null\n");
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref spl_field, ref spl_delim)) = field_split_length {
                let content_bytes = content.as_bytes();
                let delim_bytes = spl_delim.as_bytes();
                let single_delim = if delim_bytes.len() == 1 { Some(delim_bytes[0]) } else { None };
                let mut ibuf = itoa::Buffer::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, spl_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && memchr::memchr(b'\\', &val[1..val.len()-1]).is_none()
                            && !delim_bytes.is_empty()
                        {
                            let inner = &val[1..val.len()-1];
                            let count = if let Some(d) = single_delim {
                                memchr::memchr_iter(d, inner).count() + 1
                            } else {
                                inner.windows(delim_bytes.len()).filter(|w| *w == delim_bytes).count() + 1
                            };
                            compact_buf.extend_from_slice(ibuf.format(count).as_bytes());
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sol_field, ref sol_op, ref sol_arg)) = field_strop_length {
                let content_bytes = content.as_bytes();
                let mut ibuf = itoa::Buffer::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sol_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && !val[1..val.len()-1].contains(&b'\\')
                        {
                            let inner = &val[1..val.len()-1];
                            let result = match sol_op.as_str() {
                                "ltrimstr" => {
                                    let prefix = sol_arg.as_ref().unwrap().as_bytes();
                                    if inner.starts_with(prefix) {
                                        inner[prefix.len()..].iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                    } else {
                                        inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                    }
                                }
                                "rtrimstr" => {
                                    let suffix = sol_arg.as_ref().unwrap().as_bytes();
                                    if inner.ends_with(suffix) {
                                        inner[..inner.len() - suffix.len()].iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                    } else {
                                        inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                    }
                                }
                                "identity_length" | "explode" => {
                                    inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
                                }
                                _ => {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                    return Ok(());
                                }
                            };
                            compact_buf.extend_from_slice(ibuf.format(result).as_bytes());
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref flc_field, flc_op, flc_n)) = field_length_cmp {
                // .field | length cmp N — string length comparison from raw bytes
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let threshold = flc_n as usize;
                let threshold_f = flc_n;
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, flc_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && !val[1..val.len()-1].contains(&b'\\')
                        {
                            let inner = &val[1..val.len()-1];
                            let cp_count = inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
                            let result = match flc_op {
                                BinOp::Gt => cp_count as f64 > threshold_f,
                                BinOp::Lt => (cp_count as f64) < threshold_f,
                                BinOp::Ge => cp_count as f64 >= threshold_f,
                                BinOp::Le => cp_count as f64 <= threshold_f,
                                BinOp::Eq => cp_count == threshold,
                                BinOp::Ne => cp_count != threshold,
                                _ => unreachable!(),
                            };
                            if result {
                                compact_buf.extend_from_slice(b"true\n");
                            } else {
                                compact_buf.extend_from_slice(b"false\n");
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref slcf_cond, slcf_op, slcf_n, ref slcf_out)) = select_length_cmp_field {
                // select(.field | length cmp N) | .output_field — stdin path
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let threshold = slcf_n as usize;
                let threshold_f = slcf_n;
                let fields: Vec<&str> = if slcf_cond == slcf_out {
                    vec![slcf_cond.as_str()]
                } else {
                    vec![slcf_cond.as_str(), slcf_out.as_str()]
                };
                let mut ranges_buf = vec![(0usize, 0usize); fields.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &fields, &mut ranges_buf) {
                        let cond_range = ranges_buf[0];
                        let cond_val = &raw[cond_range.0..cond_range.1];
                        if cond_val.len() >= 2 && cond_val[0] == b'"' && cond_val[cond_val.len()-1] == b'"'
                            && !cond_val[1..cond_val.len()-1].contains(&b'\\')
                        {
                            let inner = &cond_val[1..cond_val.len()-1];
                            let cp_count = inner.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
                            let pass = match slcf_op {
                                BinOp::Gt => cp_count as f64 > threshold_f,
                                BinOp::Lt => (cp_count as f64) < threshold_f,
                                BinOp::Ge => cp_count as f64 >= threshold_f,
                                BinOp::Le => cp_count as f64 <= threshold_f,
                                BinOp::Eq => cp_count == threshold,
                                BinOp::Ne => cp_count != threshold,
                                _ => false,
                            };
                            if pass {
                                let out_range = if fields.len() == 1 { ranges_buf[0] } else { ranges_buf[1] };
                                let out_val = &raw[out_range.0..out_range.1];
                                if use_pretty_buf && (out_val[0] == b'{' || out_val[0] == b'[') {
                                    push_json_pretty_raw(&mut compact_buf, out_val, 2, false);
                                } else {
                                    compact_buf.extend_from_slice(out_val);
                                }
                                compact_buf.push(b'\n');
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref m2_f1, ref m2_f2)) = min_two_fields {
                let content_bytes = content.as_bytes();
                let fields: Vec<&str> = vec![m2_f1.as_str(), m2_f2.as_str()];
                let mut ranges_buf = vec![(0usize, 0usize); 2];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &fields, &mut ranges_buf) {
                        if let (Some(a), Some(b)) = (
                            parse_json_num(&raw[ranges_buf[0].0..ranges_buf[0].1]),
                            parse_json_num(&raw[ranges_buf[1].0..ranges_buf[1].1]),
                        ) {
                            let r = if a <= b { a } else { b };
                            push_jq_number_bytes(&mut compact_buf, r);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref mm_f1, ref mm_f2, mm_is_max)) = minmax_two {
                let content_bytes = content.as_bytes();
                let fields: Vec<&str> = vec![mm_f1.as_str(), mm_f2.as_str()];
                let mut ranges_buf = vec![(0usize, 0usize); 2];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &fields, &mut ranges_buf) {
                        if let (Some(a), Some(b)) = (
                            parse_json_num(&raw[ranges_buf[0].0..ranges_buf[0].1]),
                            parse_json_num(&raw[ranges_buf[1].0..ranges_buf[1].1]),
                        ) {
                            let r = if mm_is_max { if a >= b { a } else { b } } else { if a <= b { a } else { b } };
                            push_jq_number_bytes(&mut compact_buf, r);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref mm_fields, mm_is_max)) = minmax_n {
                // [.f1, .f2, .f3, ...] | min/max — stdin N-field path
                let content_bytes = content.as_bytes();
                let field_refs: Vec<&str> = mm_fields.iter().map(|s| s.as_str()).collect();
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                        let mut result: Option<f64> = None;
                        let mut all_ok = true;
                        for &(vs, ve) in ranges_buf.iter() {
                            if let Some(v) = parse_json_num(&raw[vs..ve]) {
                                result = Some(match result {
                                    None => v,
                                    Some(cur) => if mm_is_max { if v > cur { v } else { cur } } else { if v < cur { v } else { cur } },
                                });
                            } else {
                                all_ok = false;
                                break;
                            }
                        }
                        if all_ok {
                            if let Some(r) = result {
                                push_jq_number_bytes(&mut compact_buf, r);
                                compact_buf.push(b'\n');
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sl_field, sl_from, sl_to)) = field_slice {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sl_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && val[1..val.len()-1].is_ascii()
                            && !val[1..val.len()-1].contains(&b'\\')
                        {
                            let inner = &val[1..val.len()-1];
                            let len = inner.len() as i64;
                            let f = match sl_from {
                                Some(v) => if v < 0 { (len + v).max(0) as usize } else { (v as usize).min(inner.len()) },
                                None => 0,
                            };
                            let t = match sl_to {
                                Some(v) => if v < 0 { (len + v).max(0) as usize } else { (v as usize).min(inner.len()) },
                                None => inner.len(),
                            };
                            compact_buf.push(b'"');
                            if t > f { compact_buf.extend_from_slice(&inner[f..t]); }
                            compact_buf.push(b'"');
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref dk_key, ref dk_val)) = dynamic_key_obj {
                let content_bytes = content.as_bytes();
                let fields: Vec<&str> = vec![dk_key.as_str(), dk_val.as_str()];
                let mut ranges_buf = vec![(0usize, 0usize); fields.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &fields, &mut ranges_buf) {
                        let (ks, ke) = ranges_buf[0];
                        let (vs, ve) = ranges_buf[1];
                        let key_val = &raw[ks..ke];
                        if key_val.len() >= 2 && key_val[0] == b'"' {
                            if use_pretty_buf {
                                compact_buf.extend_from_slice(b"{\n  ");
                                compact_buf.extend_from_slice(key_val);
                                compact_buf.extend_from_slice(b": ");
                                let val = &raw[vs..ve];
                                if val[0] == b'{' || val[0] == b'[' {
                                    push_json_pretty_raw_at(&mut compact_buf, val, 2, false, 1);
                                } else {
                                    compact_buf.extend_from_slice(val);
                                }
                                compact_buf.extend_from_slice(b"\n}\n");
                            } else {
                                compact_buf.push(b'{');
                                compact_buf.extend_from_slice(key_val);
                                compact_buf.push(b':');
                                compact_buf.extend_from_slice(&raw[vs..ve]);
                                compact_buf.extend_from_slice(b"}\n");
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref fu_field, ref fu_op, fu_n)) = field_update_num {
                let content_bytes = content.as_bytes();
                let mut tmp = Vec::with_capacity(256);
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if use_pretty_buf {
                        tmp.clear();
                        if json_object_update_field_num(raw, 0, fu_field, *fu_op, fu_n, &mut tmp) {
                            push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else if !json_object_update_field_num(raw, 0, fu_field, *fu_op, fu_n, &mut compact_buf) {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    } else {
                        compact_buf.push(b'\n');
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref fa_set_field, ref fa_set_val)) = field_assign_const {
                let content_bytes = content.as_bytes();
                let mut tmp = Vec::with_capacity(256);
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if use_pretty_buf {
                        tmp.clear();
                        if json_object_set_field_raw(raw, 0, fa_set_field, fa_set_val, &mut tmp) {
                            let len = tmp.len();
                            if len > 0 && tmp[len-1] == b'\n' { tmp.truncate(len-1); }
                            push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else if !json_object_set_field_raw(raw, 0, fa_set_field, fa_set_val, &mut compact_buf) {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref fa_field) = field_access {
                // Field access fast path: extract a single field's raw bytes, no full parse.
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if raw[0] == b'{' {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, fa_field) {
                            let val_bytes = &raw[vs..ve];
                            emit_raw_ln!(&mut compact_buf, val_bytes);
                        } else {
                            compact_buf.extend_from_slice(b"null\n");
                        }
                    } else {
                        compact_buf.extend_from_slice(b"null\n");
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref nf) = nested_field {
                let content_bytes = content.as_bytes();
                let nf_refs: Vec<&str> = nf.iter().map(|s| s.as_str()).collect();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_nested_field_raw(raw, 0, &nf_refs) {
                        let val_bytes = &raw[vs..ve];
                        emit_raw_ln!(&mut compact_buf, val_bytes);
                    } else {
                        compact_buf.extend_from_slice(b"null\n");
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref celems) = computed_array {
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                for rexpr in celems {
                    let names = remap_expr_fields(rexpr);
                    for name in names {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved = resolve_remap_exprs_array(celems, &field_idx);
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                        if use_pretty_buf {
                            compact_buf.extend_from_slice(b"[\n");
                            for (i, res) in resolved.iter().enumerate() {
                                if i > 0 { compact_buf.extend_from_slice(b",\n"); }
                                compact_buf.extend_from_slice(b"  ");
                                emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                            }
                            compact_buf.extend_from_slice(b"\n]\n");
                        } else {
                            compact_buf.push(b'[');
                            for (i, res) in resolved.iter().enumerate() {
                                if i > 0 { compact_buf.push(b','); }
                                emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                            }
                            compact_buf.extend_from_slice(b"]\n");
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref af) = array_field {
                let content_bytes = content.as_bytes();
                let af_refs: Vec<&str> = af.iter().map(|s| s.as_str()).collect();
                let mut ranges_buf = vec![(0usize, 0usize); af_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &af_refs, &mut ranges_buf) {
                        if use_pretty_buf {
                            compact_buf.extend_from_slice(b"[\n");
                            for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                if i > 0 { compact_buf.extend_from_slice(b",\n"); }
                                compact_buf.extend_from_slice(b"  ");
                                let val = &raw[*vs..*ve];
                                if val[0] == b'{' || val[0] == b'[' {
                                    push_json_pretty_raw_at(&mut compact_buf, val, 2, false, 1);
                                } else {
                                    compact_buf.extend_from_slice(val);
                                }
                            }
                            compact_buf.extend_from_slice(b"\n]\n");
                        } else {
                            compact_buf.push(b'[');
                            for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                if i > 0 { compact_buf.push(b','); }
                                compact_buf.extend_from_slice(&raw[*vs..*ve]);
                            }
                            compact_buf.extend_from_slice(b"]\n");
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref mf) = multi_field {
                let content_bytes = content.as_bytes();
                let mf_refs: Vec<&str> = mf.iter().map(|s| s.as_str()).collect();
                let mut ranges_buf = vec![(0usize, 0usize); mf_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &mf_refs, &mut ranges_buf) {
                        for (vs, ve) in &ranges_buf {
                            let val_bytes = &raw[*vs..*ve];
                            emit_raw_ln!(&mut compact_buf, val_bytes);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref op, threshold)) = select_cmp {
                // Select fast path: extract field without full parsing, copy raw bytes on match.
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(val) = json_object_get_num(raw, 0, field) {
                        let pass = match op {
                            BinOp::Gt => val > threshold,
                            BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold,
                            BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold,
                            BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        if pass {
                            emit_raw_ln!(&mut compact_buf, raw);
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                        }
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref arith_ops, ref cmp_op, threshold)) = select_arith_cmp {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(mut val) = json_object_get_num(raw, 0, field) {
                        for (aop, n) in arith_ops {
                            val = match aop {
                                BinOp::Add => val + n, BinOp::Sub => val - n,
                                BinOp::Mul => val * n, BinOp::Div => val / n,
                                BinOp::Mod => val % n, _ => val,
                            };
                        }
                        let pass = match cmp_op {
                            BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        if pass {
                            emit_raw_ln!(&mut compact_buf, raw);
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                        }
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref op, ref val)) = select_str {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let mut expected = Vec::with_capacity(val.len() + 2);
                expected.push(b'"');
                expected.extend_from_slice(val.as_bytes());
                expected.push(b'"');
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
                        let val_bytes = &raw[vs..ve];
                        let matches = val_bytes == expected.as_slice();
                        let pass = match op { BinOp::Eq => matches, BinOp::Ne => !matches, _ => false };
                        if pass {
                            emit_raw_ln!(&mut compact_buf, raw);
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                        }
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref builtin, ref arg)) = select_str_test {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[ve-vs-1] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                            let inner = &val[1..ve-vs-1];
                            let pass = match builtin.as_str() {
                                "startswith" => inner.starts_with(arg.as_bytes()),
                                "endswith" => inner.ends_with(arg.as_bytes()),
                                "contains" => bytes_contains(inner, arg.as_bytes()),
                                _ => false,
                            };
                            if pass {
                                emit_raw_ln!(&mut compact_buf, raw);
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref rt_field, ref rt_pattern, ref rt_flags)) = select_regex_test {
                let re_pattern = if let Some(flags) = rt_flags {
                    let mut prefix = String::from("(?");
                    for c in flags.chars() {
                        match c { 'i' | 'm' | 's' => prefix.push(c), _ => {} }
                    }
                    prefix.push(')');
                    prefix.push_str(rt_pattern);
                    prefix
                } else {
                    rt_pattern.clone()
                };
                if let Ok(re) = regex::Regex::new(&re_pattern) {
                    let content_bytes = content.as_bytes();
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, rt_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && !val[1..val.len()-1].contains(&b'\\')
                            {
                                let content_str = unsafe { std::str::from_utf8_unchecked(&val[1..val.len()-1]) };
                                if re.is_match(content_str) {
                                    emit_raw_ln!(&mut compact_buf, raw);
                                }
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else {
                    let content_bytes = content.as_bytes();
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        Ok(())
                    })
                }
            } else if let Some((ref fields, ref op, threshold)) = select_nested_cmp {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_nested_field_raw(raw, 0, &field_refs) {
                        if let Some(val) = parse_json_num(&raw[vs..ve]) {
                            let pass = match op {
                                BinOp::Gt => val > threshold,
                                BinOp::Lt => val < threshold,
                                BinOp::Ge => val >= threshold,
                                BinOp::Le => val <= threshold,
                                BinOp::Eq => val == threshold,
                                BinOp::Ne => val != threshold,
                                _ => false,
                            };
                            if pass {
                                emit_raw_ln!(&mut compact_buf, raw);
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref alt_field, ref fallback_bytes)) = field_alt {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, alt_field) {
                        let val = &raw[vs..ve];
                        if val == b"null" || val == b"false" {
                            compact_buf.extend_from_slice(fallback_bytes);
                        } else {
                            compact_buf.extend_from_slice(val);
                        }
                    } else {
                        compact_buf.extend_from_slice(fallback_bytes);
                    }
                    compact_buf.push(b'\n');
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref prim_field, ref fallback_field)) = field_field_alt {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let use_primary = if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, prim_field) {
                        let pval = &raw[vs..ve];
                        if pval != b"null" && pval != b"false" {
                            compact_buf.extend_from_slice(pval);
                            true
                        } else { false }
                    } else { false };
                    if !use_primary {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, fallback_field) {
                            compact_buf.extend_from_slice(&raw[vs..ve]);
                        } else {
                            compact_buf.extend_from_slice(b"null");
                        }
                    }
                    compact_buf.push(b'\n');
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref branches, ref else_output)) = cond_chain {
                use jq_jit::interpreter::{BranchOutput, CondRhs};
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();

                // Specialized path: single-branch with lazy output fetch (no remap outputs)
                if branches.len() == 1 && !matches!(branches[0].output, BranchOutput::Remap(_) | BranchOutput::Computed(_)) && !matches!(else_output, BranchOutput::Remap(_) | BranchOutput::Computed(_)) {
                    let br = &branches[0];
                    let cond_field = br.cond_field.as_str();
                    let cond_arith = &br.cond_arith_ops;
                    let cond_op = &br.cond_op;
                    let then_out = &br.output;
                    let rhs_field: Option<&str> = if let CondRhs::Field(ref f) = br.cond_rhs { Some(f.as_str()) } else { None };
                    let rhs_const: Option<f64> = if let CondRhs::Const(n) = br.cond_rhs { Some(n) } else { None };
                    let is_non_numeric_cmp = matches!(br.cond_rhs, CondRhs::Null | CondRhs::Str(_) | CondRhs::Bool(_) | CondRhs::Startswith(_) | CondRhs::Endswith(_) | CondRhs::Contains(_));
                    let rhs_str_json: Option<Vec<u8>> = if let CondRhs::Str(ref s) = br.cond_rhs {
                        let mut buf = Vec::with_capacity(s.len() + 2);
                        buf.push(b'"');
                        buf.extend_from_slice(s.as_bytes());
                        buf.push(b'"');
                        Some(buf)
                    } else { None };
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        // Non-numeric comparison (null/str/bool/strfunc)
                        if is_non_numeric_cmp {
                            let pass = if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, cond_field) {
                                let field_bytes = &raw[vs..ve];
                                match &br.cond_rhs {
                                    CondRhs::Null => match cond_op {
                                        BinOp::Eq => field_bytes == b"null",
                                        BinOp::Ne => field_bytes != b"null",
                                        _ => false,
                                    },
                                    CondRhs::Bool(expected) => {
                                        let val = if field_bytes == b"true" { Some(true) } else if field_bytes == b"false" { Some(false) } else { None };
                                        val.map_or(false, |v| match cond_op {
                                            BinOp::Eq => v == *expected,
                                            BinOp::Ne => v != *expected,
                                            _ => false,
                                        })
                                    }
                                    CondRhs::Str(_) => {
                                        let rhs_json = rhs_str_json.as_ref().unwrap();
                                        match cond_op {
                                            BinOp::Eq => field_bytes == rhs_json.as_slice(),
                                            BinOp::Ne => field_bytes != rhs_json.as_slice(),
                                            _ => false,
                                        }
                                    }
                                    CondRhs::Startswith(ref s) => {
                                        field_bytes.len() >= 2 && field_bytes[0] == b'"'
                                            && field_bytes[1..field_bytes.len()-1].starts_with(s.as_bytes())
                                    }
                                    CondRhs::Endswith(ref s) => {
                                        field_bytes.len() >= 2 && field_bytes[0] == b'"'
                                            && field_bytes[1..field_bytes.len()-1].ends_with(s.as_bytes())
                                    }
                                    CondRhs::Contains(ref s) => {
                                        field_bytes.len() >= 2 && field_bytes[0] == b'"' && {
                                            let inner = &field_bytes[1..field_bytes.len()-1];
                                            let sb = s.as_bytes();
                                            inner.windows(sb.len()).any(|w| w == sb)
                                        }
                                    }
                                    _ => false,
                                }
                            } else {
                                matches!(br.cond_rhs, CondRhs::Null) && matches!(cond_op, BinOp::Eq)
                            };
                            let out_br = if pass { then_out } else { else_output };
                            match out_br {
                                BranchOutput::Literal(ref bytes) => {
                                    compact_buf.extend_from_slice(bytes);
                                    compact_buf.push(b'\n');
                                }
                                BranchOutput::Field(ref f) => {
                                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, f) {
                                        let val = &raw[vs..ve];
                                        if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                            push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                        } else {
                                            compact_buf.extend_from_slice(val);
                                        }
                                        compact_buf.push(b'\n');
                                    } else {
                                        compact_buf.extend_from_slice(b"null\n");
                                    }
                                }
                                BranchOutput::Empty => {}
                                BranchOutput::Remap(_) | BranchOutput::Computed(_) => unreachable!(),
                            }
                            if compact_buf.len() >= 1 << 17 {
                                let _ = out.write_all(&compact_buf);
                                compact_buf.clear();
                            }
                            return Ok(());
                        }
                        let (lv, rv, got) = if let Some(rf) = rhs_field {
                            if let Some((l, r)) = json_object_get_two_nums(raw, 0, cond_field, rf) {
                                (l, r, true)
                            } else { (0.0, 0.0, false) }
                        } else {
                            let thr = rhs_const.unwrap();
                            if let Some(l) = json_object_get_num(raw, 0, cond_field) {
                                (l, thr, true)
                            } else { (0.0, 0.0, false) }
                        };
                        if got {
                            let orig_lv = lv;
                            let mut cmp_lv = lv;
                            for (aop, n) in cond_arith {
                                cmp_lv = match aop {
                                    BinOp::Add => cmp_lv + n, BinOp::Sub => cmp_lv - n,
                                    BinOp::Mul => cmp_lv * n, BinOp::Div => cmp_lv / n,
                                    BinOp::Mod => cmp_lv % n, _ => cmp_lv,
                                };
                            }
                            let pass = match cond_op {
                                BinOp::Gt => cmp_lv > rv, BinOp::Lt => cmp_lv < rv,
                                BinOp::Ge => cmp_lv >= rv, BinOp::Le => cmp_lv <= rv,
                                BinOp::Eq => cmp_lv == rv, BinOp::Ne => cmp_lv != rv,
                                _ => false,
                            };
                            let out_br = if pass { then_out } else { else_output };
                            match out_br {
                                BranchOutput::Literal(ref bytes) => {
                                    compact_buf.extend_from_slice(bytes);
                                    compact_buf.push(b'\n');
                                }
                                BranchOutput::Field(ref f) => {
                                    let fs = f.as_str();
                                    if fs == cond_field {
                                        push_jq_number_bytes(&mut compact_buf, orig_lv);
                                        compact_buf.push(b'\n');
                                    } else if rhs_field.map_or(false, |rf| fs == rf) {
                                        push_jq_number_bytes(&mut compact_buf, rv);
                                        compact_buf.push(b'\n');
                                    } else {
                                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, f) {
                                            let val = &raw[vs..ve];
                                            if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                                push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                            } else {
                                                compact_buf.extend_from_slice(val);
                                            }
                                            compact_buf.push(b'\n');
                                        }
                                    }
                                }
                                BranchOutput::Empty => {}
                                BranchOutput::Remap(_) | BranchOutput::Computed(_) => unreachable!(),
                            }
                        } else {
                            match else_output {
                                BranchOutput::Literal(ref bytes) => {
                                    compact_buf.extend_from_slice(bytes);
                                    compact_buf.push(b'\n');
                                }
                                BranchOutput::Field(ref f) => {
                                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, f) {
                                        let val = &raw[vs..ve];
                                        if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                            push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                        } else {
                                            compact_buf.extend_from_slice(val);
                                        }
                                        compact_buf.push(b'\n');
                                    }
                                }
                                BranchOutput::Empty => {}
                                BranchOutput::Remap(_) | BranchOutput::Computed(_) => unreachable!(),
                            }
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else {
                // General path
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                let ensure_field = |f: &String, all: &mut Vec<String>, idx: &mut std::collections::HashMap<String, usize>| {
                    if !idx.contains_key(f) {
                        idx.insert(f.clone(), all.len());
                        all.push(f.clone());
                    }
                };
                let collect_output_fields = |out: &BranchOutput, all: &mut Vec<String>, idx: &mut std::collections::HashMap<String, usize>| {
                    match out {
                        BranchOutput::Field(ref f) => { ensure_field(f, all, idx); }
                        BranchOutput::Remap(ref entries) => {
                            for (_, rexpr) in entries {
                                for name in remap_expr_fields(rexpr) {
                                    let s = name.to_string();
                                    ensure_field(&s, all, idx);
                                }
                            }
                        }
                        BranchOutput::Computed(ref rexpr) => {
                            for name in remap_expr_fields(rexpr) {
                                let s = name.to_string();
                                ensure_field(&s, all, idx);
                            }
                        }
                        _ => {}
                    }
                };
                for br in branches {
                    ensure_field(&br.cond_field, &mut all_fields, &mut field_idx);
                    if let CondRhs::Field(ref f) = br.cond_rhs {
                        ensure_field(f, &mut all_fields, &mut field_idx);
                    }
                    collect_output_fields(&br.output, &mut all_fields, &mut field_idx);
                }
                collect_output_fields(else_output, &mut all_fields, &mut field_idx);
                // Pre-resolve computed exprs
                let branch_computed2: Vec<Option<ResolvedRemap>> = branches.iter().map(|br| {
                    if let BranchOutput::Computed(ref rexpr) = br.output {
                        Some(resolve_one_remap(rexpr, &field_idx))
                    } else { None }
                }).collect();
                let else_computed2 = if let BranchOutput::Computed(ref rexpr) = else_output {
                    Some(resolve_one_remap(rexpr, &field_idx))
                } else { None };
                // Pre-resolve remap exprs for each branch
                let branch_resolved: Vec<Option<(Vec<Vec<u8>>, Vec<ResolvedRemap>, &[u8])>> = branches.iter().map(|br| {
                    if let BranchOutput::Remap(ref entries) = br.output {
                        let kp = if use_pretty_buf {
                            build_obj_key_prefixes_pretty(entries.iter().map(|(k, _)| k.as_str()))
                        } else {
                            build_obj_key_prefixes(entries.iter().map(|(k, _)| k.as_str()))
                        };
                        let res = resolve_remap_exprs(entries, &field_idx);
                        Some((kp, res, if use_pretty_buf { &b"\n}\n"[..] } else { &b"}\n"[..] }))
                    } else { None }
                }).collect();
                let else_resolved = if let BranchOutput::Remap(ref entries) = else_output {
                    let kp = if use_pretty_buf {
                        build_obj_key_prefixes_pretty(entries.iter().map(|(k, _)| k.as_str()))
                    } else {
                        build_obj_key_prefixes(entries.iter().map(|(k, _)| k.as_str()))
                    };
                    let res = resolve_remap_exprs(entries, &field_idx);
                    Some((kp, res, if use_pretty_buf { &b"\n}\n"[..] } else { &b"}\n"[..] }))
                } else { None };
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                        let mut output = None;
                        let mut output_idx = 0usize;
                        for (i, br) in branches.iter().enumerate() {
                            let idx = field_idx[&br.cond_field];
                            let (vs, ve) = ranges_buf[idx];
                            let field_bytes = &raw[vs..ve];
                            let pass = match &br.cond_rhs {
                                CondRhs::Null => match br.cond_op {
                                    BinOp::Eq => field_bytes == b"null",
                                    BinOp::Ne => field_bytes != b"null",
                                    _ => false,
                                },
                                CondRhs::Bool(expected) => {
                                    let val = if field_bytes == b"true" { Some(true) } else if field_bytes == b"false" { Some(false) } else { None };
                                    val.map_or(false, |v| match br.cond_op {
                                        BinOp::Eq => v == *expected,
                                        BinOp::Ne => v != *expected,
                                        _ => false,
                                    })
                                }
                                CondRhs::Str(ref s) => {
                                    let rhs_json = {
                                        let mut buf = Vec::with_capacity(s.len() + 2);
                                        buf.push(b'"');
                                        buf.extend_from_slice(s.as_bytes());
                                        buf.push(b'"');
                                        buf
                                    };
                                    match br.cond_op {
                                        BinOp::Eq => field_bytes == rhs_json.as_slice(),
                                        BinOp::Ne => field_bytes != rhs_json.as_slice(),
                                        _ => false,
                                    }
                                }
                                CondRhs::Startswith(ref s) => {
                                    field_bytes.len() >= 2 && field_bytes[0] == b'"'
                                        && field_bytes[1..field_bytes.len()-1].starts_with(s.as_bytes())
                                }
                                CondRhs::Endswith(ref s) => {
                                    field_bytes.len() >= 2 && field_bytes[0] == b'"'
                                        && field_bytes[1..field_bytes.len()-1].ends_with(s.as_bytes())
                                }
                                CondRhs::Contains(ref s) => {
                                    field_bytes.len() >= 2 && field_bytes[0] == b'"' && {
                                        let inner = &field_bytes[1..field_bytes.len()-1];
                                        let sb = s.as_bytes();
                                        inner.windows(sb.len()).any(|w| w == sb)
                                    }
                                }
                                _ => {
                                    if let Some(mut val) = parse_json_num(field_bytes) {
                                        for (aop, n) in &br.cond_arith_ops {
                                            val = match aop {
                                                BinOp::Add => val + n, BinOp::Sub => val - n,
                                                BinOp::Mul => val * n, BinOp::Div => val / n,
                                                BinOp::Mod => val % n, _ => val,
                                            };
                                        }
                                        let rhs_val = match &br.cond_rhs {
                                            CondRhs::Const(n) => *n,
                                            CondRhs::Field(ref f) => {
                                                let ri = field_idx[f];
                                                let (rs, re) = ranges_buf[ri];
                                                match parse_json_num(&raw[rs..re]) { Some(v) => v, None => { continue; } }
                                            }
                                            _ => unreachable!(),
                                        };
                                        match br.cond_op {
                                            BinOp::Gt => val > rhs_val, BinOp::Lt => val < rhs_val,
                                            BinOp::Ge => val >= rhs_val, BinOp::Le => val <= rhs_val,
                                            BinOp::Eq => val == rhs_val, BinOp::Ne => val != rhs_val,
                                            _ => false,
                                        }
                                    } else { false }
                                }
                            };
                            if pass { output = Some(&br.output); output_idx = i; break; }
                        }
                        let out_branch = output.unwrap_or(else_output);
                        match out_branch {
                            BranchOutput::Literal(ref bytes) => {
                                compact_buf.extend_from_slice(bytes);
                                compact_buf.push(b'\n');
                            }
                            BranchOutput::Field(ref f) => {
                                let idx = field_idx[f];
                                let (vs, ve) = ranges_buf[idx];
                                let val = &raw[vs..ve];
                                if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                    push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                } else {
                                    compact_buf.extend_from_slice(val);
                                }
                                compact_buf.push(b'\n');
                            }
                            BranchOutput::Remap(_) => {
                                let resolved_data = if output.is_some() {
                                    branch_resolved[output_idx].as_ref()
                                } else {
                                    else_resolved.as_ref()
                                };
                                if let Some((ref kp, ref res, close)) = resolved_data {
                                    for (j, rv) in res.iter().enumerate() {
                                        compact_buf.extend_from_slice(&kp[j]);
                                        emit_resolved_value(&mut compact_buf, rv, raw, &ranges_buf);
                                    }
                                    compact_buf.extend_from_slice(close);
                                }
                            }
                            BranchOutput::Computed(_) => {
                                let resolved_remap = if output.is_some() {
                                    branch_computed2[output_idx].as_ref()
                                } else {
                                    else_computed2.as_ref()
                                };
                                if let Some(rv) = resolved_remap {
                                    emit_resolved_value(&mut compact_buf, rv, raw, &ranges_buf);
                                    compact_buf.push(b'\n');
                                }
                            }
                            BranchOutput::Empty => { /* produce no output */ }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
                } // end general path
            } else if let Some((ref field, ref op, threshold, ref t_bytes, ref f_bytes)) = cmp_branch_lit {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(val) = json_object_get_num(raw, 0, field) {
                        let pass = match op {
                            BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        compact_buf.extend_from_slice(if pass { t_bytes } else { f_bytes });
                        compact_buf.push(b'\n');
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref arith_ops, ref cmp_op, threshold, ref t_bytes, ref f_bytes)) = arith_cmp_branch_lit {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(mut val) = json_object_get_num(raw, 0, field) {
                        for (aop, n) in arith_ops {
                            val = match aop {
                                BinOp::Add => val + n, BinOp::Sub => val - n,
                                BinOp::Mul => val * n, BinOp::Div => val / n,
                                BinOp::Mod => val % n, _ => val,
                            };
                        }
                        let pass = match cmp_op {
                            BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        compact_buf.extend_from_slice(if pass { t_bytes } else { f_bytes });
                        compact_buf.push(b'\n');
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref f1, ref cmp_op, ref f2, ref t_bytes, ref f_bytes)) = field_field_cmp_branch {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, f1, f2) {
                        let pass = match cmp_op {
                            BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                            BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                            BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                            _ => false,
                        };
                        compact_buf.extend_from_slice(if pass { t_bytes } else { f_bytes });
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref iacond, ref then_arr, ref else_arr)) = if_cmp_arrays {
                use jq_jit::ir::BinOp;
                use jq_jit::interpreter::IfArrayCond;
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                match iacond {
                    IfArrayCond::FieldConst(f, _, _) => {
                        field_idx.insert(f.clone(), 0);
                        all_fields.push(f.clone());
                    }
                    IfArrayCond::FieldField(f1, _, f2) => {
                        field_idx.insert(f1.clone(), 0);
                        all_fields.push(f1.clone());
                        if !field_idx.contains_key(f2) {
                            field_idx.insert(f2.clone(), 1);
                            all_fields.push(f2.clone());
                        }
                    }
                }
                for rexpr in then_arr.iter().chain(else_arr.iter()) {
                    for name in remap_expr_fields(rexpr) {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved_then: Vec<ResolvedRemap> = then_arr.iter()
                    .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                    .collect();
                let resolved_else: Vec<ResolvedRemap> = else_arr.iter()
                    .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                    .collect();
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                        let pass = match iacond {
                            IfArrayCond::FieldConst(f, op, threshold) => {
                                let idx = field_idx[f];
                                let r = &ranges_buf[idx];
                                if r.0 < r.1 {
                                    if let Some(v) = parse_json_num(&raw[r.0..r.1]) {
                                        match op {
                                            BinOp::Gt => v > *threshold, BinOp::Lt => v < *threshold,
                                            BinOp::Ge => v >= *threshold, BinOp::Le => v <= *threshold,
                                            BinOp::Eq => v == *threshold, BinOp::Ne => v != *threshold,
                                            _ => false,
                                        }
                                    } else { false }
                                } else { false }
                            }
                            IfArrayCond::FieldField(f1, op, f2) => {
                                let i1 = field_idx[f1];
                                let i2 = field_idx[f2];
                                let r1 = &ranges_buf[i1];
                                let r2 = &ranges_buf[i2];
                                if r1.0 < r1.1 && r2.0 < r2.1 {
                                    if let (Some(v1), Some(v2)) = (
                                        parse_json_num(&raw[r1.0..r1.1]),
                                        parse_json_num(&raw[r2.0..r2.1]),
                                    ) {
                                        match op {
                                            BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                            BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                            BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                            _ => false,
                                        }
                                    } else { false }
                                } else { false }
                            }
                        };
                        let resolved = if pass { &resolved_then } else { &resolved_else };
                        compact_buf.push(b'[');
                        for (i, res) in resolved.iter().enumerate() {
                            if i > 0 { compact_buf.push(b','); }
                            emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                        }
                        compact_buf.extend_from_slice(b"]\n");
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref conj, ref cmps)) = select_compound {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let is_and = matches!(conj, BinOp::And);
                let two_field = if cmps.len() == 2 && cmps[0].0 != cmps[1].0 {
                    Some((cmps[0].0.as_str(), cmps[1].0.as_str()))
                } else { None };
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let pass = if let Some((f1, f2)) = two_field {
                        if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, f1, f2) {
                            let c1 = match &cmps[0].1 {
                                BinOp::Gt => v1 > cmps[0].2, BinOp::Lt => v1 < cmps[0].2,
                                BinOp::Ge => v1 >= cmps[0].2, BinOp::Le => v1 <= cmps[0].2,
                                BinOp::Eq => v1 == cmps[0].2, BinOp::Ne => v1 != cmps[0].2,
                                _ => false,
                            };
                            let c2 = match &cmps[1].1 {
                                BinOp::Gt => v2 > cmps[1].2, BinOp::Lt => v2 < cmps[1].2,
                                BinOp::Ge => v2 >= cmps[1].2, BinOp::Le => v2 <= cmps[1].2,
                                BinOp::Eq => v2 == cmps[1].2, BinOp::Ne => v2 != cmps[1].2,
                                _ => false,
                            };
                            if is_and { c1 && c2 } else { c1 || c2 }
                        } else { false }
                    } else if is_and {
                        cmps.iter().all(|(field, op, threshold)| {
                            json_object_get_num(raw, 0, field).map_or(false, |val| match op {
                                BinOp::Gt => val > *threshold, BinOp::Lt => val < *threshold,
                                BinOp::Ge => val >= *threshold, BinOp::Le => val <= *threshold,
                                BinOp::Eq => val == *threshold, BinOp::Ne => val != *threshold,
                                _ => false,
                            })
                        })
                    } else {
                        cmps.iter().any(|(field, op, threshold)| {
                            json_object_get_num(raw, 0, field).map_or(false, |val| match op {
                                BinOp::Gt => val > *threshold, BinOp::Lt => val < *threshold,
                                BinOp::Ge => val >= *threshold, BinOp::Le => val <= *threshold,
                                BinOp::Eq => val == *threshold, BinOp::Ne => val != *threshold,
                                _ => false,
                            })
                        })
                    };
                    if pass {
                        emit_raw_ln!(&mut compact_buf, raw);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref conj, ref cmps, ref out_field)) = select_compound_field {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let is_and = matches!(conj, BinOp::And);
                let two_field_lazy = if cmps.len() == 2 && cmps[0].0 != cmps[1].0
                    && cmps[0].0 != *out_field && cmps[1].0 != *out_field {
                    Some((cmps[0].0.as_str(), cmps[1].0.as_str(), out_field.as_str()))
                } else { None };
                let mut all_fields: Vec<String> = cmps.iter().map(|(f, _, _)| f.clone()).collect();
                if !all_fields.contains(out_field) { all_fields.push(out_field.clone()); }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let out_idx = all_fields.iter().position(|f| f == out_field).unwrap();
                let cmp_indices: Vec<(usize, BinOp, f64)> = cmps.iter().map(|(f, op, thr)| {
                    (all_fields.iter().position(|af| af == f).unwrap(), *op, *thr)
                }).collect();
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((f1, f2, of)) = two_field_lazy {
                        if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, f1, f2) {
                            let c1 = match &cmps[0].1 {
                                BinOp::Gt => v1 > cmps[0].2, BinOp::Lt => v1 < cmps[0].2,
                                BinOp::Ge => v1 >= cmps[0].2, BinOp::Le => v1 <= cmps[0].2,
                                BinOp::Eq => v1 == cmps[0].2, BinOp::Ne => v1 != cmps[0].2,
                                _ => false,
                            };
                            let c2 = match &cmps[1].1 {
                                BinOp::Gt => v2 > cmps[1].2, BinOp::Lt => v2 < cmps[1].2,
                                BinOp::Ge => v2 >= cmps[1].2, BinOp::Le => v2 <= cmps[1].2,
                                BinOp::Eq => v2 == cmps[1].2, BinOp::Ne => v2 != cmps[1].2,
                                _ => false,
                            };
                            let pass = if is_and { c1 && c2 } else { c1 || c2 };
                            if pass {
                                if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, of) {
                                    let val = &raw[vs..ve];
                                    if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                        push_json_pretty_raw(&mut compact_buf, val, 2, false);
                                    } else {
                                        compact_buf.extend_from_slice(val);
                                    }
                                    compact_buf.push(b'\n');
                                }
                            }
                        }
                    } else {
                        if !json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            return Ok(());
                        }
                        let check = |idx: usize, op: &BinOp, thr: &f64| -> bool {
                            let (vs, ve) = ranges_buf[idx];
                            parse_json_num(&raw[vs..ve]).map_or(false, |val| match op {
                                BinOp::Gt => val > *thr, BinOp::Lt => val < *thr,
                                BinOp::Ge => val >= *thr, BinOp::Le => val <= *thr,
                                BinOp::Eq => val == *thr, BinOp::Ne => val != *thr,
                                _ => false,
                            })
                        };
                        let pass = if is_and {
                            cmp_indices.iter().all(|(idx, op, thr)| check(*idx, op, thr))
                        } else {
                            cmp_indices.iter().any(|(idx, op, thr)| check(*idx, op, thr))
                        };
                        if pass {
                            let (vs, ve) = ranges_buf[out_idx];
                            let val = &raw[vs..ve];
                            if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                push_json_pretty_raw(&mut compact_buf, val, 2, false);
                            } else {
                                compact_buf.extend_from_slice(val);
                            }
                            compact_buf.push(b'\n');
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref conj, ref cmps, ref remap)) = select_compound_remap {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let is_and = matches!(conj, BinOp::And);
                let two_cmp_lazy = if cmps.len() == 2 && cmps[0].0 != cmps[1].0 {
                    Some((cmps[0].0.as_str(), cmps[1].0.as_str()))
                } else { None };
                let remap_fields: Vec<&str> = {
                    let mut rf = Vec::new();
                    for (_, f) in remap { if !rf.contains(&f.as_str()) { rf.push(f.as_str()); } }
                    rf
                };
                let remap_field_map: std::collections::HashMap<&str, usize> =
                    remap_fields.iter().enumerate().map(|(i, f)| (*f, i)).collect();
                let remap_out_indices: Vec<(&str, usize)> = remap.iter().map(|(k, f)| {
                    (k.as_str(), remap_field_map[f.as_str()])
                }).collect();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                let ensure_field = |f: &String, all: &mut Vec<String>, idx: &mut std::collections::HashMap<String, usize>| {
                    if !idx.contains_key(f) { idx.insert(f.clone(), all.len()); all.push(f.clone()); }
                };
                for (f, _, _) in cmps { ensure_field(f, &mut all_fields, &mut field_idx); }
                for (_, f) in remap { ensure_field(f, &mut all_fields, &mut field_idx); }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let cmp_indices: Vec<(usize, BinOp, f64)> = cmps.iter().map(|(f, op, thr)| {
                    (field_idx[f], *op, *thr)
                }).collect();
                let gen_remap_indices: Vec<(&str, usize)> = remap.iter().map(|(k, f)| {
                    (k.as_str(), field_idx[f])
                }).collect();
                let mut ranges_buf = vec![(0usize, 0usize); std::cmp::max(field_refs.len(), remap_fields.len())];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((f1, f2)) = two_cmp_lazy {
                        if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, f1, f2) {
                            let c1 = match &cmps[0].1 {
                                BinOp::Gt => v1 > cmps[0].2, BinOp::Lt => v1 < cmps[0].2,
                                BinOp::Ge => v1 >= cmps[0].2, BinOp::Le => v1 <= cmps[0].2,
                                BinOp::Eq => v1 == cmps[0].2, BinOp::Ne => v1 != cmps[0].2,
                                _ => false,
                            };
                            let c2 = match &cmps[1].1 {
                                BinOp::Gt => v2 > cmps[1].2, BinOp::Lt => v2 < cmps[1].2,
                                BinOp::Ge => v2 >= cmps[1].2, BinOp::Le => v2 <= cmps[1].2,
                                BinOp::Eq => v2 == cmps[1].2, BinOp::Ne => v2 != cmps[1].2,
                                _ => false,
                            };
                            let pass = if is_and { c1 && c2 } else { c1 || c2 };
                            if pass {
                                if json_object_get_fields_raw_buf(raw, 0, &remap_fields, &mut ranges_buf) {
                                    compact_buf.push(b'{');
                                    for (i, (key, fidx)) in remap_out_indices.iter().enumerate() {
                                        if i > 0 { compact_buf.push(b','); }
                                        compact_buf.push(b'"');
                                        compact_buf.extend_from_slice(key.as_bytes());
                                        compact_buf.extend_from_slice(b"\":");
                                        let (vs, ve) = ranges_buf[*fidx];
                                        compact_buf.extend_from_slice(&raw[vs..ve]);
                                    }
                                    compact_buf.extend_from_slice(b"}\n");
                                }
                            }
                        }
                    } else {
                        if !json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            return Ok(());
                        }
                        let check = |idx: usize, op: &BinOp, thr: &f64| -> bool {
                            let (vs, ve) = ranges_buf[idx];
                            parse_json_num(&raw[vs..ve]).map_or(false, |val| match op {
                                BinOp::Gt => val > *thr, BinOp::Lt => val < *thr,
                                BinOp::Ge => val >= *thr, BinOp::Le => val <= *thr,
                                BinOp::Eq => val == *thr, BinOp::Ne => val != *thr,
                                _ => false,
                            })
                        };
                        let pass = if is_and {
                            cmp_indices.iter().all(|(idx, op, thr)| check(*idx, op, thr))
                        } else {
                            cmp_indices.iter().any(|(idx, op, thr)| check(*idx, op, thr))
                        };
                        if pass {
                            compact_buf.push(b'{');
                            for (i, (key, fidx)) in gen_remap_indices.iter().enumerate() {
                                if i > 0 { compact_buf.push(b','); }
                                compact_buf.push(b'"');
                                compact_buf.extend_from_slice(key.as_bytes());
                                compact_buf.extend_from_slice(b"\":");
                                let (vs, ve) = ranges_buf[*fidx];
                                compact_buf.extend_from_slice(&raw[vs..ve]);
                            }
                            compact_buf.extend_from_slice(b"}\n");
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref shm_fields, shm_is_and)) = select_has_multi {
                let field_refs: Vec<&str> = shm_fields.iter().map(|s| s.as_str()).collect();
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let pass = if field_refs.len() == 1 {
                        json_object_has_key(raw, 0, field_refs[0]).unwrap_or(false)
                    } else if shm_is_and {
                        json_object_has_all_keys(raw, 0, &field_refs).unwrap_or(false)
                    } else {
                        json_object_has_any_key(raw, 0, &field_refs).unwrap_or(false)
                    };
                    if pass {
                        emit_raw_ln!(&mut compact_buf, raw);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref op, threshold, ref out_field)) = select_cmp_field {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                        let pass = match op {
                            BinOp::Gt => val > threshold,
                            BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold,
                            BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold,
                            BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        if pass {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                compact_buf.extend_from_slice(&raw[vs..ve]);
                                compact_buf.push(b'\n');
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref arith_ops, ref cmp_op, threshold, ref out_field)) = select_arith_cmp_field {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(mut val) = json_object_get_num(raw, 0, sel_field) {
                        for (aop, n) in arith_ops {
                            val = match aop {
                                BinOp::Add => val + n, BinOp::Sub => val - n,
                                BinOp::Mul => val * n, BinOp::Div => val / n,
                                BinOp::Mod => val % n, _ => val,
                            };
                        }
                        let pass = match cmp_op {
                            BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        if pass {
                            if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                compact_buf.extend_from_slice(&raw[vs..ve]);
                                compact_buf.push(b'\n');
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref sel_op, threshold, ref out_field, ref uop)) = select_cmp_field_unary {
                use jq_jit::ir::{BinOp, UnaryOp};
                let is_string_op = matches!(uop, UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase);
                let is_length = matches!(uop, UnaryOp::Length | UnaryOp::Utf8ByteLength);
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                        let pass = match sel_op {
                            BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        if pass {
                            if is_length {
                                if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                    let fval = &raw[vs..ve];
                                    if fval.len() >= 2 && fval[0] == b'"' && fval[fval.len()-1] == b'"' {
                                        let inner = &fval[1..fval.len()-1];
                                        let len = if matches!(uop, UnaryOp::Utf8ByteLength) {
                                            inner.iter().filter(|&&b| b != b'\\').count()
                                        } else if !inner.contains(&b'\\') && inner.iter().all(|&b| b < 0x80) {
                                            inner.len()
                                        } else {
                                            let unescaped = json_unescape_bytes(inner);
                                            if let Ok(s) = std::str::from_utf8(&unescaped) { s.chars().count() } else { unescaped.len() }
                                        };
                                        compact_buf.extend_from_slice(itoa::Buffer::new().format(len).as_bytes());
                                        compact_buf.push(b'\n');
                                    } else if fval[0] == b'[' || fval[0] == b'{' {
                                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                    } else if let Some(n) = parse_json_num(fval) {
                                        push_jq_number_bytes(&mut compact_buf, n.abs());
                                        compact_buf.push(b'\n');
                                    } else {
                                        compact_buf.extend_from_slice(b"null\n");
                                    }
                                }
                            } else if is_string_op {
                                if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                    let fval = &raw[vs..ve];
                                    if fval.len() >= 2 && fval[0] == b'"' && !fval[1..fval.len()-1].contains(&b'\\') {
                                        compact_buf.push(b'"');
                                        for &b in &fval[1..fval.len()-1] {
                                            compact_buf.push(match uop {
                                                UnaryOp::AsciiDowncase => if b >= b'A' && b <= b'Z' { b + 32 } else { b },
                                                UnaryOp::AsciiUpcase => if b >= b'a' && b <= b'z' { b - 32 } else { b },
                                                _ => b,
                                            });
                                        }
                                        compact_buf.extend_from_slice(b"\"\n");
                                    } else {
                                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                    }
                                }
                            } else if matches!(uop, UnaryOp::ToString) {
                                if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                                    let fval = &raw[vs..ve];
                                    if fval[0] == b'"' {
                                        compact_buf.extend_from_slice(fval);
                                        compact_buf.push(b'\n');
                                    } else if let Some(n) = parse_json_num(fval) {
                                        compact_buf.push(b'"');
                                        let i = n as i64;
                                        if i as f64 == n {
                                            compact_buf.extend_from_slice(itoa::Buffer::new().format(i).as_bytes());
                                        } else {
                                            compact_buf.extend_from_slice(ryu::Buffer::new().format(n).as_bytes());
                                        }
                                        compact_buf.extend_from_slice(b"\"\n");
                                    } else {
                                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                    }
                                }
                            } else {
                                if let Some(n) = json_object_get_num(raw, 0, out_field) {
                                    let result = match uop {
                                        UnaryOp::Floor => n.floor(), UnaryOp::Ceil => n.ceil(),
                                        UnaryOp::Sqrt => n.sqrt(), UnaryOp::Fabs | UnaryOp::Abs => n.abs(),
                                        _ => n,
                                    };
                                    push_jq_number_bytes(&mut compact_buf, result);
                                    compact_buf.push(b'\n');
                                } else {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref op, threshold, ref pairs)) = select_cmp_remap {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let remap_fields: Vec<&str> = {
                    let mut v: Vec<&str> = Vec::new();
                    for (_, src) in pairs { if !v.contains(&src.as_str()) { v.push(src.as_str()); } }
                    v
                };
                let pair_indices: Vec<usize> = pairs.iter().map(|(_, src)| {
                    remap_fields.iter().position(|&f| f == src.as_str()).unwrap()
                }).collect();
                let key_prefixes = if use_pretty_buf {
                    build_obj_key_prefixes_pretty(pairs.iter().map(|(k, _)| k.as_str()))
                } else {
                    build_obj_key_prefixes(pairs.iter().map(|(k, _)| k.as_str()))
                };
                let obj_close: &[u8] = if use_pretty_buf { b"\n}\n" } else { b"}\n" };
                let mut ranges_buf = vec![(0usize, 0usize); remap_fields.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                        let pass = match op {
                            BinOp::Gt => val > threshold,
                            BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold,
                            BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold,
                            BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        if pass {
                            if json_object_get_fields_raw_buf(raw, 0, &remap_fields, &mut ranges_buf) {
                                for (i, &idx) in pair_indices.iter().enumerate() {
                                    compact_buf.extend_from_slice(&key_prefixes[i]);
                                    let (vs, ve) = ranges_buf[idx];
                                    if use_pretty_buf {
                                        let val_bytes = &raw[vs..ve];
                                        if val_bytes[0] == b'{' || val_bytes[0] == b'[' {
                                            push_json_pretty_raw_at(&mut compact_buf, val_bytes, 2, false, 1);
                                        } else {
                                            compact_buf.extend_from_slice(val_bytes);
                                        }
                                    } else {
                                        compact_buf.extend_from_slice(&raw[vs..ve]);
                                    }
                                }
                                compact_buf.extend_from_slice(obj_close);
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref sel_op, threshold, ref cremap)) = select_cmp_cremap {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                field_idx.insert(sel_field.clone(), 0);
                all_fields.push(sel_field.clone());
                for (_, rexpr) in cremap {
                    for name in remap_expr_fields(rexpr) {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved = resolve_remap_exprs(cremap, &field_idx);
                let key_prefixes = if use_pretty_buf {
                    build_obj_key_prefixes_pretty(cremap.iter().map(|(k, _)| k.as_str()))
                } else {
                    build_obj_key_prefixes(cremap.iter().map(|(k, _)| k.as_str()))
                };
                let obj_close: &[u8] = if use_pretty_buf { b"\n}\n" } else { b"}\n" };
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    // Lazy fetch: check select condition first with fast single-field lookup
                    if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                        let pass = match sel_op {
                            BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        if pass {
                            // Only now extract all remap fields
                            if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                                for (i, res) in resolved.iter().enumerate() {
                                    compact_buf.extend_from_slice(&key_prefixes[i]);
                                    emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                }
                                compact_buf.extend_from_slice(obj_close);
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref sel_op, threshold, ref arr_elems)) = select_cmp_array {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                field_idx.insert(sel_field.clone(), 0);
                all_fields.push(sel_field.clone());
                for rexpr in arr_elems {
                    for name in remap_expr_fields(rexpr) {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                    .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                    .collect();
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                        let pass = match sel_op {
                            BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        if pass {
                            if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                                compact_buf.push(b'[');
                                for (i, res) in resolved.iter().enumerate() {
                                    if i > 0 { compact_buf.push(b','); }
                                    emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                }
                                compact_buf.extend_from_slice(b"]\n");
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref arith_ops, ref sel_op, threshold, ref arr_elems)) = select_arith_cmp_array {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                field_idx.insert(sel_field.clone(), 0);
                all_fields.push(sel_field.clone());
                for rexpr in arr_elems {
                    for name in remap_expr_fields(rexpr) {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                    .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                    .collect();
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(mut val) = json_object_get_num(raw, 0, sel_field) {
                        for (aop, n) in arith_ops {
                            val = match aop {
                                BinOp::Add => val + n, BinOp::Sub => val - n,
                                BinOp::Mul => val * n, BinOp::Div => val / n,
                                BinOp::Mod => if n.is_finite() && *n != 0.0 { val % n } else { val },
                                _ => val,
                            };
                        }
                        let pass = match sel_op {
                            BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                            BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                            BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                            _ => false,
                        };
                        if pass {
                            if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                                compact_buf.push(b'[');
                                for (i, res) in resolved.iter().enumerate() {
                                    if i > 0 { compact_buf.push(b','); }
                                    emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                }
                                compact_buf.extend_from_slice(b"]\n");
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref sel_op, threshold, ref out_rexpr)) = select_cmp_value {
                use jq_jit::interpreter::RemapExpr;
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let fused_mode: u8 = match out_rexpr {
                    RemapExpr::FieldOpConst(f, _, _) if f == sel_field => 1,
                    RemapExpr::FieldOpField(f1, _, f2) if f1 == sel_field || f2 == sel_field => 2,
                    RemapExpr::ConstOpField(_, _, f) if f == sel_field => 3,
                    _ => 0,
                };
                let other_field: Option<&str> = if fused_mode == 2 {
                    match out_rexpr {
                        RemapExpr::FieldOpField(f1, _, f2) => {
                            if f1 == sel_field { Some(f2.as_str()) } else { Some(f1.as_str()) }
                        }
                        _ => None,
                    }
                } else { None };
                // Pre-compute fields for fused_mode 0 (general path)
                let mut gen_all_fields: Vec<String> = Vec::new();
                let mut gen_field_idx = std::collections::HashMap::new();
                if fused_mode == 0 {
                    for name in remap_expr_fields(out_rexpr) {
                        if !gen_field_idx.contains_key(name) {
                            gen_field_idx.insert(name.to_string(), gen_all_fields.len());
                            gen_all_fields.push(name.to_string());
                        }
                    }
                }
                let gen_field_refs: Vec<&str> = gen_all_fields.iter().map(|s| s.as_str()).collect();
                let mut ranges_buf = vec![(0usize, 0usize); gen_field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    match fused_mode {
                        1 => {
                            if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                                let pass = match sel_op {
                                    BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                    BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                    BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                    _ => false,
                                };
                                if pass {
                                    if let RemapExpr::FieldOpConst(_, op, n) = out_rexpr {
                                        let r = match op { BinOp::Add => val + n, BinOp::Sub => val - n, BinOp::Mul => val * n, BinOp::Div => val / n, BinOp::Mod => val % n, _ => unreachable!() };
                                        push_jq_number_bytes(&mut compact_buf, r);
                                    }
                                    compact_buf.push(b'\n');
                                }
                            }
                        }
                        2 => {
                            let of = other_field.unwrap();
                            if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, sel_field, of) {
                                let pass = match sel_op {
                                    BinOp::Gt => v1 > threshold, BinOp::Lt => v1 < threshold,
                                    BinOp::Ge => v1 >= threshold, BinOp::Le => v1 <= threshold,
                                    BinOp::Eq => v1 == threshold, BinOp::Ne => v1 != threshold,
                                    _ => false,
                                };
                                if pass {
                                    if let RemapExpr::FieldOpField(f1, op, _) = out_rexpr {
                                        let (lhs_val, rhs_val) = if f1 == sel_field { (v1, v2) } else { (v2, v1) };
                                        let r = match op { BinOp::Add => lhs_val + rhs_val, BinOp::Sub => lhs_val - rhs_val, BinOp::Mul => lhs_val * rhs_val, BinOp::Div => lhs_val / rhs_val, BinOp::Mod => lhs_val % rhs_val, _ => unreachable!() };
                                        push_jq_number_bytes(&mut compact_buf, r);
                                    }
                                    compact_buf.push(b'\n');
                                }
                            }
                        }
                        3 => {
                            if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                                let pass = match sel_op {
                                    BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                    BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                    BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                    _ => false,
                                };
                                if pass {
                                    if let RemapExpr::ConstOpField(n, op, _) = out_rexpr {
                                        let r = match op { BinOp::Add => n + val, BinOp::Sub => n - val, BinOp::Mul => n * val, BinOp::Div => n / val, BinOp::Mod => n % val, _ => unreachable!() };
                                        push_jq_number_bytes(&mut compact_buf, r);
                                    }
                                    compact_buf.push(b'\n');
                                }
                            }
                        }
                        _ => {
                            if let Some(val) = json_object_get_num(raw, 0, sel_field) {
                                let pass = match sel_op {
                                    BinOp::Gt => val > threshold, BinOp::Lt => val < threshold,
                                    BinOp::Ge => val >= threshold, BinOp::Le => val <= threshold,
                                    BinOp::Eq => val == threshold, BinOp::Ne => val != threshold,
                                    _ => false,
                                };
                                if pass {
                                    if json_object_get_fields_raw_buf(raw, 0, &gen_field_refs, &mut ranges_buf) {
                                        emit_remap_value(&mut compact_buf, out_rexpr, raw, &ranges_buf, &gen_field_idx);
                                    } else {
                                        compact_buf.extend_from_slice(b"null");
                                    }
                                    compact_buf.push(b'\n');
                                }
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sff_f1, sff_op, ref sff_f2, ref sff_out)) = select_ff_cmp_field {
                // select(.f1 cmp .f2) | .output — stdin path
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let out_is_separate = sff_out != sff_f1 && sff_out != sff_f2;
                let mut all_fields: Vec<&str> = Vec::new();
                let mut idx = std::collections::HashMap::new();
                for f in [sff_f1.as_str(), sff_f2.as_str(), sff_out.as_str()] {
                    if !idx.contains_key(f) { idx.insert(f, all_fields.len()); all_fields.push(f); }
                }
                let mut ranges_buf = vec![(0usize, 0usize); all_fields.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if out_is_separate {
                        if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, sff_f1, sff_f2) {
                            let pass = match sff_op {
                                BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                _ => false,
                            };
                            if pass {
                                if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sff_out) {
                                    let out_val = &raw[vs..ve];
                                    if use_pretty_buf && (out_val[0] == b'{' || out_val[0] == b'[') {
                                        push_json_pretty_raw(&mut compact_buf, out_val, 2, false);
                                    } else {
                                        compact_buf.extend_from_slice(out_val);
                                    }
                                    compact_buf.push(b'\n');
                                }
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else if json_object_get_fields_raw_buf(raw, 0, &all_fields, &mut ranges_buf) {
                        let i1 = idx[sff_f1.as_str()];
                        let i2 = idx[sff_f2.as_str()];
                        if let (Some(v1), Some(v2)) = (
                            parse_json_num(&raw[ranges_buf[i1].0..ranges_buf[i1].1]),
                            parse_json_num(&raw[ranges_buf[i2].0..ranges_buf[i2].1]),
                        ) {
                            let pass = match sff_op {
                                BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                _ => false,
                            };
                            if pass {
                                let oi = idx[sff_out.as_str()];
                                let (vs, ve) = ranges_buf[oi];
                                let out_val = &raw[vs..ve];
                                if use_pretty_buf && (out_val[0] == b'{' || out_val[0] == b'[') {
                                    push_json_pretty_raw(&mut compact_buf, out_val, 2, false);
                                } else {
                                    compact_buf.extend_from_slice(out_val);
                                }
                                compact_buf.push(b'\n');
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sff_f1, sff_op, ref sff_f2)) = select_ff_cmp {
                // select(.f1 cmp .f2) — stdin path, output whole object
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((v1, v2)) = json_object_get_two_nums(raw, 0, sff_f1, sff_f2) {
                        let pass = match sff_op {
                            BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                            BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                            BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                            _ => false,
                        };
                        if pass {
                            emit_raw_ln!(&mut compact_buf, raw);
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sff_f1, ref sff_op, ref sff_f2, ref cremap)) = select_ff_cmp_cremap {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                for f in [sff_f1, sff_f2] {
                    if !field_idx.contains_key(f) {
                        field_idx.insert(f.clone(), all_fields.len());
                        all_fields.push(f.clone());
                    }
                }
                for (_, rexpr) in cremap {
                    for name in remap_expr_fields(rexpr) {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved = resolve_remap_exprs(cremap, &field_idx);
                let f1_idx = field_idx[sff_f1];
                let f2_idx = field_idx[sff_f2];
                let key_prefixes = if use_pretty_buf {
                    build_obj_key_prefixes_pretty(cremap.iter().map(|(k, _)| k.as_str()))
                } else {
                    build_obj_key_prefixes(cremap.iter().map(|(k, _)| k.as_str()))
                };
                let obj_close: &[u8] = if use_pretty_buf { b"\n}\n" } else { b"}\n" };
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                        let r1 = &ranges_buf[f1_idx];
                        let r2 = &ranges_buf[f2_idx];
                        if r1.0 < r1.1 && r2.0 < r2.1 {
                            if let (Some(v1), Some(v2)) = (
                                parse_json_num(&raw[r1.0..r1.1]),
                                parse_json_num(&raw[r2.0..r2.1]),
                            ) {
                                let pass = match sff_op {
                                    BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                    BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                    BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                    _ => false,
                                };
                                if pass {
                                    for (i, res) in resolved.iter().enumerate() {
                                        compact_buf.extend_from_slice(&key_prefixes[i]);
                                        emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                    }
                                    compact_buf.extend_from_slice(obj_close);
                                }
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sff_f1, ref sff_op, ref sff_f2, ref out_rexpr)) = select_ff_cmp_value {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                for f in [sff_f1, sff_f2] {
                    if !field_idx.contains_key(f) {
                        field_idx.insert(f.clone(), all_fields.len());
                        all_fields.push(f.clone());
                    }
                }
                for name in remap_expr_fields(out_rexpr) {
                    if !field_idx.contains_key(name) {
                        field_idx.insert(name.to_string(), all_fields.len());
                        all_fields.push(name.to_string());
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved = resolve_one_remap(out_rexpr, &field_idx);
                let f1_idx = field_idx[sff_f1];
                let f2_idx = field_idx[sff_f2];
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                        let r1 = &ranges_buf[f1_idx];
                        let r2 = &ranges_buf[f2_idx];
                        if r1.0 < r1.1 && r2.0 < r2.1 {
                            if let (Some(v1), Some(v2)) = (
                                parse_json_num(&raw[r1.0..r1.1]),
                                parse_json_num(&raw[r2.0..r2.1]),
                            ) {
                                let pass = match sff_op {
                                    BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                    BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                    BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                    _ => false,
                                };
                                if pass {
                                    emit_resolved_value(&mut compact_buf, &resolved, raw, &ranges_buf);
                                    compact_buf.push(b'\n');
                                }
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sff_f1, ref sff_op, ref sff_f2, ref arr_elems)) = select_ff_cmp_array {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                for f in [sff_f1, sff_f2] {
                    if !field_idx.contains_key(f) {
                        field_idx.insert(f.clone(), all_fields.len());
                        all_fields.push(f.clone());
                    }
                }
                for rexpr in arr_elems {
                    for name in remap_expr_fields(rexpr) {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                    .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                    .collect();
                let f1_idx = field_idx[sff_f1];
                let f2_idx = field_idx[sff_f2];
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                        let r1 = &ranges_buf[f1_idx];
                        let r2 = &ranges_buf[f2_idx];
                        if r1.0 < r1.1 && r2.0 < r2.1 {
                            if let (Some(v1), Some(v2)) = (
                                parse_json_num(&raw[r1.0..r1.1]),
                                parse_json_num(&raw[r2.0..r2.1]),
                            ) {
                                let pass = match sff_op {
                                    BinOp::Gt => v1 > v2, BinOp::Lt => v1 < v2,
                                    BinOp::Ge => v1 >= v2, BinOp::Le => v1 <= v2,
                                    BinOp::Eq => v1 == v2, BinOp::Ne => v1 != v2,
                                    _ => false,
                                };
                                if pass {
                                    compact_buf.push(b'[');
                                    for (i, res) in resolved.iter().enumerate() {
                                        if i > 0 { compact_buf.push(b','); }
                                        emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                                    }
                                    compact_buf.extend_from_slice(b"]\n");
                                }
                            }
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref conj, ref cmps, ref arr_elems)) = select_compound_array {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                for (f, _, _) in cmps {
                    if !field_idx.contains_key(f) {
                        field_idx.insert(f.clone(), all_fields.len());
                        all_fields.push(f.clone());
                    }
                }
                for rexpr in arr_elems {
                    for name in remap_expr_fields(rexpr) {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                    .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                    .collect();
                let resolved_cmps: Vec<(usize, BinOp, f64)> = cmps.iter()
                    .map(|(f, op, n)| (field_idx[f], *op, *n))
                    .collect();
                let is_and = matches!(conj, BinOp::And);
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                        let mut pass = is_and;
                        for (idx, op, threshold) in &resolved_cmps {
                            let r = &ranges_buf[*idx];
                            let cmp_pass = if r.0 < r.1 {
                                if let Some(v) = parse_json_num(&raw[r.0..r.1]) {
                                    match op {
                                        BinOp::Gt => v > *threshold, BinOp::Lt => v < *threshold,
                                        BinOp::Ge => v >= *threshold, BinOp::Le => v <= *threshold,
                                        BinOp::Eq => v == *threshold, BinOp::Ne => v != *threshold,
                                        _ => false,
                                    }
                                } else { false }
                            } else { false };
                            if is_and {
                                if !cmp_pass { pass = false; break; }
                            } else {
                                if cmp_pass { pass = true; break; }
                            }
                        }
                        if pass {
                            compact_buf.push(b'[');
                            for (i, res) in resolved.iter().enumerate() {
                                if i > 0 { compact_buf.push(b','); }
                                emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                            }
                            compact_buf.extend_from_slice(b"]\n");
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref test_type, ref test_arg, ref out_field)) = select_str_field {
                let content_bytes = content.as_bytes();
                let expected_eq = if test_type == "eq" || test_type == "ne" {
                    let mut e = Vec::with_capacity(test_arg.len() + 2);
                    e.push(b'"'); e.extend_from_slice(test_arg.as_bytes()); e.push(b'"');
                    Some(e)
                } else { None };
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let pass = if let Some(ref expected) = expected_eq {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                            let val_bytes = &raw[vs..ve];
                            let m = val_bytes == expected.as_slice();
                            if test_type == "eq" { m } else { !m }
                        } else { false }
                    } else {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[ve-vs-1] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                let inner = &val[1..ve-vs-1];
                                match test_type.as_str() {
                                    "startswith" => inner.starts_with(test_arg.as_bytes()),
                                    "endswith" => inner.ends_with(test_arg.as_bytes()),
                                    "contains" => bytes_contains(inner, test_arg.as_bytes()),
                                    _ => false,
                                }
                            } else { false }
                        } else { false }
                    };
                    if pass {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, out_field) {
                            let val = &raw[vs..ve];
                            if use_pretty_buf && (val[0] == b'{' || val[0] == b'[') {
                                push_json_pretty_raw(&mut compact_buf, val, 2, false);
                            } else {
                                compact_buf.extend_from_slice(val);
                            }
                            compact_buf.push(b'\n');
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref chain_field, ref chain_ops, ref chain_terminal)) = field_string_chain {
                use jq_jit::interpreter::{StringChainOp, StringChainTerminal};
                let content_bytes = content.as_bytes();
                let mut tmp_str = Vec::<u8>::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, chain_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && !val[1..val.len()-1].contains(&b'\\')
                        {
                            tmp_str.clear();
                            tmp_str.extend_from_slice(&val[1..val.len()-1]);
                            for op in chain_ops.iter() {
                                match op {
                                    StringChainOp::AsciiDowncase => {
                                        for b in tmp_str.iter_mut() {
                                            if *b >= b'A' && *b <= b'Z' { *b += 32; }
                                        }
                                    }
                                    StringChainOp::AsciiUpcase => {
                                        for b in tmp_str.iter_mut() {
                                            if *b >= b'a' && *b <= b'z' { *b -= 32; }
                                        }
                                    }
                                    StringChainOp::Ltrimstr(ref prefix) => {
                                        let pb = prefix.as_bytes();
                                        if tmp_str.len() >= pb.len() && &tmp_str[..pb.len()] == pb {
                                            tmp_str.drain(..pb.len());
                                        }
                                    }
                                    StringChainOp::Rtrimstr(ref suffix) => {
                                        let sb = suffix.as_bytes();
                                        if tmp_str.len() >= sb.len() && &tmp_str[tmp_str.len()-sb.len()..] == sb {
                                            let new_len = tmp_str.len() - sb.len();
                                            tmp_str.truncate(new_len);
                                        }
                                    }
                                }
                            }
                            match chain_terminal {
                                StringChainTerminal::None => {
                                    compact_buf.push(b'"');
                                    compact_buf.extend_from_slice(&tmp_str);
                                    compact_buf.extend_from_slice(b"\"\n");
                                }
                                StringChainTerminal::Startswith(ref arg) => {
                                    let ab = arg.as_bytes();
                                    compact_buf.extend_from_slice(if tmp_str.len() >= ab.len() && &tmp_str[..ab.len()] == ab { b"true\n" } else { b"false\n" });
                                }
                                StringChainTerminal::Endswith(ref arg) => {
                                    let ab = arg.as_bytes();
                                    compact_buf.extend_from_slice(if tmp_str.len() >= ab.len() && &tmp_str[tmp_str.len()-ab.len()..] == ab { b"true\n" } else { b"false\n" });
                                }
                                StringChainTerminal::Contains(ref arg) => {
                                    compact_buf.extend_from_slice(if bytes_contains(&tmp_str, arg.as_bytes()) { b"true\n" } else { b"false\n" });
                                }
                                StringChainTerminal::Length => {
                                    let count = tmp_str.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
                                    push_jq_number_bytes(&mut compact_buf, count as f64);
                                    compact_buf.push(b'\n');
                                }
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref test_type, ref test_arg, ref remap_pairs)) = select_str_cremap {
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                all_fields.push(sel_field.clone());
                for (_, rexpr) in remap_pairs {
                    for f in remap_expr_fields(rexpr) {
                        if !all_fields.iter().any(|x| x == f) {
                            all_fields.push(f.to_string());
                        }
                    }
                }
                let field_strs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let mut field_idx = std::collections::HashMap::new();
                for (i, f) in all_fields.iter().enumerate() { field_idx.insert(f.clone(), i); }
                let resolved = resolve_remap_exprs(remap_pairs, &field_idx);
                let obj_keys: Vec<&str> = remap_pairs.iter().map(|(k, _)| k.as_str()).collect();
                let key_prefixes = if use_pretty_buf {
                    build_obj_key_prefixes_pretty(obj_keys.iter().copied())
                } else {
                    build_obj_key_prefixes(obj_keys.iter().copied())
                };
                let obj_close: &[u8] = if use_pretty_buf { b"\n}\n" } else { b"}\n" };
                let expected_eq = if test_type == "eq" || test_type == "ne" {
                    let mut e = Vec::with_capacity(test_arg.len() + 2);
                    e.push(b'"'); e.extend_from_slice(test_arg.as_bytes()); e.push(b'"');
                    Some(e)
                } else { None };
                let mut ranges = vec![(0usize, 0usize); all_fields.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let pass = if let Some(ref expected) = expected_eq {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                            let val_bytes = &raw[vs..ve];
                            let m = val_bytes == expected.as_slice();
                            if test_type == "eq" { m } else { !m }
                        } else { false }
                    } else {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[ve-vs-1] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                let inner = &val[1..ve-vs-1];
                                match test_type.as_str() {
                                    "startswith" => inner.starts_with(test_arg.as_bytes()),
                                    "endswith" => inner.ends_with(test_arg.as_bytes()),
                                    "contains" => bytes_contains(inner, test_arg.as_bytes()),
                                    _ => false,
                                }
                            } else { false }
                        } else { false }
                    };
                    if pass {
                        if json_object_get_fields_raw_buf(raw, 0, &field_strs, &mut ranges) {
                            for (prefix, res) in key_prefixes.iter().zip(resolved.iter()) {
                                compact_buf.extend_from_slice(prefix);
                                emit_resolved_value(&mut compact_buf, res, raw, &ranges);
                            }
                            compact_buf.extend_from_slice(obj_close);
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sel_field, ref test_type, ref test_arg, ref arr_elems)) = select_str_array {
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                all_fields.push(sel_field.clone());
                for rexpr in arr_elems {
                    for f in remap_expr_fields(rexpr) {
                        if !all_fields.iter().any(|x| x == f) {
                            all_fields.push(f.to_string());
                        }
                    }
                }
                let field_strs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let mut field_idx = std::collections::HashMap::new();
                for (i, f) in all_fields.iter().enumerate() { field_idx.insert(f.clone(), i); }
                let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                    .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                    .collect();
                let expected_eq = if test_type == "eq" || test_type == "ne" {
                    let mut e = Vec::with_capacity(test_arg.len() + 2);
                    e.push(b'"'); e.extend_from_slice(test_arg.as_bytes()); e.push(b'"');
                    Some(e)
                } else { None };
                let mut ranges = vec![(0usize, 0usize); all_fields.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let pass = if let Some(ref expected) = expected_eq {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                            let val_bytes = &raw[vs..ve];
                            let m = val_bytes == expected.as_slice();
                            if test_type == "eq" { m } else { !m }
                        } else { false }
                    } else {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sel_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[ve-vs-1] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                let inner = &val[1..ve-vs-1];
                                match test_type.as_str() {
                                    "startswith" => inner.starts_with(test_arg.as_bytes()),
                                    "endswith" => inner.ends_with(test_arg.as_bytes()),
                                    "contains" => bytes_contains(inner, test_arg.as_bytes()),
                                    _ => false,
                                }
                            } else { false }
                        } else { false }
                    };
                    if pass {
                        if json_object_get_fields_raw_buf(raw, 0, &field_strs, &mut ranges) {
                            compact_buf.push(b'[');
                            for (i, res) in resolved.iter().enumerate() {
                                if i > 0 { compact_buf.push(b','); }
                                emit_resolved_value(&mut compact_buf, res, raw, &ranges);
                            }
                            compact_buf.extend_from_slice(b"]\n");
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref remap) = field_remap {
                let content_bytes = content.as_bytes();
                let input_fields: Vec<&str> = remap.iter().map(|(_, f)| f.as_str()).collect();
                let mut ranges_buf = vec![(0usize, 0usize); input_fields.len()];
                if use_pretty_buf {
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &input_fields, &mut ranges_buf) {
                            compact_buf.extend_from_slice(b"{\n");
                            for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                if i > 0 { compact_buf.extend_from_slice(b",\n"); }
                                compact_buf.extend_from_slice(b"  \"");
                                compact_buf.extend_from_slice(remap[i].0.as_bytes());
                                compact_buf.extend_from_slice(b"\": ");
                                let val = &raw[*vs..*ve];
                                if val[0] == b'{' || val[0] == b'[' {
                                    push_json_pretty_raw_at(&mut compact_buf, val, 2, false, 1);
                                } else {
                                    compact_buf.extend_from_slice(val);
                                }
                            }
                            compact_buf.extend_from_slice(b"\n}\n");
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else {
                    let mut key_prefixes: Vec<Vec<u8>> = Vec::with_capacity(remap.len());
                    for (i, (out_key, _)) in remap.iter().enumerate() {
                        let mut prefix = Vec::new();
                        if i == 0 { prefix.push(b'{'); } else { prefix.push(b','); }
                        prefix.push(b'"');
                        prefix.extend_from_slice(out_key.as_bytes());
                        prefix.extend_from_slice(b"\":");
                        key_prefixes.push(prefix);
                    }
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        if json_object_get_fields_raw_buf(raw, 0, &input_fields, &mut ranges_buf) {
                            for (i, (vs, ve)) in ranges_buf.iter().enumerate() {
                                compact_buf.extend_from_slice(&key_prefixes[i]);
                                compact_buf.extend_from_slice(&raw[*vs..*ve]);
                            }
                            compact_buf.extend_from_slice(b"}\n");
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                }
            } else if let Some(ref cremap) = computed_remap {
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                for (_, rexpr) in cremap {
                    for name in remap_expr_fields(rexpr) {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved = resolve_remap_exprs(cremap, &field_idx);
                let key_prefixes = if use_pretty_buf {
                    build_obj_key_prefixes_pretty(cremap.iter().map(|(k, _)| k.as_str()))
                } else {
                    build_obj_key_prefixes(cremap.iter().map(|(k, _)| k.as_str()))
                };
                let obj_close: &[u8] = if use_pretty_buf { b"\n}\n" } else { b"}\n" };
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                        for (i, res) in resolved.iter().enumerate() {
                            compact_buf.extend_from_slice(&key_prefixes[i]);
                            emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                        }
                        compact_buf.extend_from_slice(obj_close);
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref arr_elems) = standalone_array {
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                let mut field_idx = std::collections::HashMap::new();
                for rexpr in arr_elems {
                    for name in remap_expr_fields(rexpr) {
                        if !field_idx.contains_key(name) {
                            field_idx.insert(name.to_string(), all_fields.len());
                            all_fields.push(name.to_string());
                        }
                    }
                }
                let field_refs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let resolved: Vec<ResolvedRemap> = arr_elems.iter()
                    .map(|rexpr| resolve_one_remap(rexpr, &field_idx))
                    .collect();
                let mut ranges_buf = vec![(0usize, 0usize); field_refs.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                        compact_buf.push(b'[');
                        for (i, res) in resolved.iter().enumerate() {
                            if i > 0 { compact_buf.push(b','); }
                            emit_resolved_value(&mut compact_buf, res, raw, &ranges_buf);
                        }
                        compact_buf.extend_from_slice(b"]\n");
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref f1, ref op, ref f2)) = field_binop {
                // Arithmetic fast path: extract two numeric fields, compute, output.
                // Falls back to normal JIT if fields are missing or non-numeric.
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((a, b)) = json_object_get_two_nums(raw, 0, f1, f2) {
                        let result = match op {
                            BinOp::Add => a + b,
                            BinOp::Sub => a - b,
                            BinOp::Mul => a * b,
                            BinOp::Div => if b == 0.0 { f64::NAN } else { a / b },
                            BinOp::Mod => { let r = a % b; if r.is_finite() { r } else { f64::NAN } },
                            _ => unreachable!(),
                        };
                        if result.is_nan() {
                            compact_buf.extend_from_slice(b"null\n");
                        } else {
                            push_jq_number_bytes(&mut compact_buf, result);
                            compact_buf.push(b'\n');
                        }
                    } else {
                        // Field missing or non-numeric: fall back to parse + JIT
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref f1, ref op, ref f2)) = field_binop_tostring {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((a, b)) = json_object_get_two_nums(raw, 0, f1, f2) {
                        let result = match op {
                            BinOp::Add => a + b,
                            BinOp::Sub => a - b,
                            BinOp::Mul => a * b,
                            BinOp::Div => a / b,
                            BinOp::Mod => a % b,
                            _ => unreachable!(),
                        };
                        if result.is_finite() {
                            compact_buf.push(b'"');
                            let i = result as i64;
                            if i as f64 == result {
                                compact_buf.extend_from_slice(itoa::Buffer::new().format(i).as_bytes());
                            } else {
                                compact_buf.extend_from_slice(ryu::Buffer::new().format(result).as_bytes());
                            }
                            compact_buf.extend_from_slice(b"\"\n");
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref uop)) = field_unary_num {
                use jq_jit::ir::UnaryOp;
                let is_string_op = matches!(uop, UnaryOp::AsciiDowncase | UnaryOp::AsciiUpcase);
                let is_length_op = matches!(uop, UnaryOp::Length | UnaryOp::Utf8ByteLength);
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if is_string_op {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && !val[1..val.len()-1].contains(&b'\\') {
                                compact_buf.push(b'"');
                                for &byte in &val[1..val.len()-1] {
                                    compact_buf.push(match uop {
                                        UnaryOp::AsciiDowncase => if byte >= b'A' && byte <= b'Z' { byte + 32 } else { byte },
                                        UnaryOp::AsciiUpcase => if byte >= b'a' && byte <= b'z' { byte - 32 } else { byte },
                                        _ => unreachable!(),
                                    });
                                }
                                compact_buf.push(b'"');
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else if is_length_op {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, field) {
                            let val = &raw[vs..ve];
                            match val[0] {
                                b'"' => {
                                    let inner = &val[1..ve-vs-1];
                                    if !inner.contains(&b'\\') {
                                        let len = if matches!(uop, UnaryOp::Utf8ByteLength) {
                                            inner.len()
                                        } else if inner.is_ascii() { inner.len() }
                                        else { unsafe { std::str::from_utf8_unchecked(inner) }.chars().count() };
                                        push_jq_number_bytes(&mut compact_buf, len as f64);
                                        compact_buf.push(b'\n');
                                    } else {
                                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                    }
                                }
                                b'n' => { compact_buf.extend_from_slice(b"0\n"); }
                                b'[' | b'{' => {
                                    if let Some(len) = json_value_length(val, 0) {
                                        push_jq_number_bytes(&mut compact_buf, len as f64);
                                        compact_buf.push(b'\n');
                                    } else {
                                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                    }
                                }
                                _ => {
                                    if let Some(n) = json_object_get_num(raw, 0, field) {
                                        push_jq_number_bytes(&mut compact_buf, n.abs());
                                    } else { compact_buf.extend_from_slice(b"null"); }
                                    compact_buf.push(b'\n');
                                }
                            }
                        } else { compact_buf.extend_from_slice(b"0\n"); }
                    } else if let Some(n) = json_object_get_num(raw, 0, field) {
                        if matches!(uop, UnaryOp::ToString) {
                            compact_buf.push(b'"');
                            push_jq_number_bytes(&mut compact_buf, n);
                            compact_buf.push(b'"');
                            compact_buf.push(b'\n');
                        } else {
                            let result = match uop {
                                UnaryOp::Floor => n.floor(),
                                UnaryOp::Ceil => n.ceil(),
                                UnaryOp::Sqrt => n.sqrt(),
                                UnaryOp::Fabs | UnaryOp::Abs => n.abs(),
                                _ => unreachable!(),
                            };
                            push_jq_number_bytes(&mut compact_buf, result);
                            compact_buf.push(b'\n');
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref bop, cval, ref uop_opt, const_left)) = field_binop_const_unary {
                use jq_jit::ir::BinOp;
                use jq_jit::ir::UnaryOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(n) = json_object_get_num(raw, 0, field) {
                        let (a, b) = if const_left { (cval, n) } else { (n, cval) };
                        let mid = match bop {
                            BinOp::Add => a + b,
                            BinOp::Sub => a - b,
                            BinOp::Mul => a * b,
                            BinOp::Div => a / b,
                            BinOp::Mod => a % b,
                            _ => unreachable!(),
                        };
                        let result = if let Some(uop) = uop_opt {
                            match uop {
                                UnaryOp::Floor => mid.floor(),
                                UnaryOp::Ceil => mid.ceil(),
                                UnaryOp::Sqrt => mid.sqrt(),
                                UnaryOp::Fabs | UnaryOp::Abs => mid.abs(),
                                _ => unreachable!(),
                            }
                        } else { mid };
                        push_jq_number_bytes(&mut compact_buf, result);
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref ops)) = field_arith_chain {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(n) = json_object_get_num(raw, 0, field) {
                        let mut result = n;
                        for &(ref op, c) in ops.iter() {
                            result = match op {
                                BinOp::Add => result + c,
                                BinOp::Sub => result - c,
                                BinOp::Mul => result * c,
                                BinOp::Div => result / c,
                                BinOp::Mod => result % c,
                                _ => unreachable!(),
                            };
                        }
                        push_jq_number_bytes(&mut compact_buf, result);
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref ops)) = field_arith_tostring {
                // .field arith_chain | tostring — stdin path
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(n) = json_object_get_num(raw, 0, field) {
                        let mut result = n;
                        for &(ref op, c) in ops.iter() {
                            result = match op {
                                BinOp::Add => result + c,
                                BinOp::Sub => result - c,
                                BinOp::Mul => result * c,
                                BinOp::Div => result / c,
                                BinOp::Mod => result % c,
                                _ => unreachable!(),
                            };
                        }
                        compact_buf.push(b'"');
                        let i = result as i64;
                        if i as f64 == result {
                            compact_buf.extend_from_slice(itoa::Buffer::new().format(i).as_bytes());
                        } else {
                            compact_buf.extend_from_slice(ryu::Buffer::new().format(result).as_bytes());
                        }
                        compact_buf.extend_from_slice(b"\"\n");
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref nfields, ref arith)) = numeric_expr {
                let nf_count = nfields.len();
                let content_bytes = content.as_bytes();
                let field_refs: Vec<&str> = nfields.iter().map(|s| s.as_str()).collect();
                let mut ranges_buf = vec![(0usize, 0usize); nf_count];
                let mut vals_buf: Vec<f64> = vec![0.0; nf_count];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let ok = if nf_count == 1 {
                        if let Some(v) = json_object_get_num(raw, 0, &nfields[0]) {
                            vals_buf[0] = v; true
                        } else { false }
                    } else if nf_count == 2 {
                        if let Some((a, b)) = json_object_get_two_nums(raw, 0, &nfields[0], &nfields[1]) {
                            vals_buf[0] = a; vals_buf[1] = b; true
                        } else { false }
                    } else {
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            let mut all_ok = true;
                            for (i, &(s, e)) in ranges_buf.iter().enumerate() {
                                match fast_float::parse::<f64, _>(unsafe { std::str::from_utf8_unchecked(&raw[s..e]) }) {
                                    Ok(n) => vals_buf[i] = n,
                                    Err(_) => { all_ok = false; break; }
                                }
                            }
                            all_ok
                        } else { false }
                    };
                    if ok {
                        let result = arith.eval(&vals_buf);
                        push_jq_number_bytes(&mut compact_buf, result);
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref nfields, ref arith, math_op)) = numeric_expr_unary {
                let nf_count = nfields.len();
                let content_bytes = content.as_bytes();
                let field_refs: Vec<&str> = nfields.iter().map(|s| s.as_str()).collect();
                let mut ranges_buf = vec![(0usize, 0usize); nf_count];
                let mut vals_buf: Vec<f64> = vec![0.0; nf_count];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let ok = if nf_count == 1 {
                        if let Some(v) = json_object_get_num(raw, 0, &nfields[0]) {
                            vals_buf[0] = v; true
                        } else { false }
                    } else if nf_count == 2 {
                        if let Some((a, b)) = json_object_get_two_nums(raw, 0, &nfields[0], &nfields[1]) {
                            vals_buf[0] = a; vals_buf[1] = b; true
                        } else { false }
                    } else {
                        if json_object_get_fields_raw_buf(raw, 0, &field_refs, &mut ranges_buf) {
                            let mut all_ok = true;
                            for (i, &(s, e)) in ranges_buf.iter().enumerate() {
                                match fast_float::parse::<f64, _>(unsafe { std::str::from_utf8_unchecked(&raw[s..e]) }) {
                                    Ok(n) => vals_buf[i] = n,
                                    Err(_) => { all_ok = false; break; }
                                }
                            }
                            all_ok
                        } else { false }
                    };
                    if ok {
                        let result = apply_math_unary(math_op, arith.eval(&vals_buf));
                        push_jq_number_bytes(&mut compact_buf, result);
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref f1, ref cmp_op, ref f2)) = field_field_cmp {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((n1, n2)) = json_object_get_two_nums(raw, 0, f1, f2) {
                        let result = match cmp_op {
                            BinOp::Gt => n1 > n2, BinOp::Lt => n1 < n2,
                            BinOp::Ge => n1 >= n2, BinOp::Le => n1 <= n2,
                            BinOp::Eq => n1 == n2, BinOp::Ne => n1 != n2,
                            _ => unreachable!(),
                        };
                        compact_buf.extend_from_slice(if result { b"true\n" } else { b"false\n" });
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref cmp_op, cval)) = field_const_cmp {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(n) = json_object_get_num(raw, 0, field) {
                        let result = match cmp_op {
                            BinOp::Gt => n > cval, BinOp::Lt => n < cval,
                            BinOp::Ge => n >= cval, BinOp::Le => n <= cval,
                            BinOp::Eq => n == cval, BinOp::Ne => n != cval,
                            _ => unreachable!(),
                        };
                        compact_buf.extend_from_slice(if result { b"true\n" } else { b"false\n" });
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref field, ref arith_ops, ref cmp_op, threshold)) = arith_chain_cmp {
                use jq_jit::ir::BinOp;
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(mut n) = json_object_get_num(raw, 0, field) {
                        for (aop, val) in arith_ops {
                            n = match aop {
                                BinOp::Add => n + val, BinOp::Sub => n - val,
                                BinOp::Mul => n * val, BinOp::Div => n / val,
                                BinOp::Mod => n % val, _ => n,
                            };
                        }
                        let result = match cmp_op {
                            BinOp::Gt => n > threshold, BinOp::Lt => n < threshold,
                            BinOp::Ge => n >= threshold, BinOp::Le => n <= threshold,
                            BinOp::Eq => n == threshold, BinOp::Ne => n != threshold,
                            _ => unreachable!(),
                        };
                        compact_buf.extend_from_slice(if result { b"true\n" } else { b"false\n" });
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref conjunct, ref cmps)) = compound_field_cmp {
                use jq_jit::ir::BinOp;
                let is_and = matches!(conjunct, BinOp::And);
                let mut field_names: Vec<String> = Vec::new();
                let mut field_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                for (f, _, _) in cmps {
                    if !field_idx.contains_key(f) {
                        field_idx.insert(f.clone(), field_names.len());
                        field_names.push(f.clone());
                    }
                }
                let cmp_spec: Vec<(usize, BinOp, f64)> = cmps.iter().map(|(f, op, n)| (field_idx[f], *op, *n)).collect();
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let got = if field_names.len() == 2 {
                        json_object_get_two_nums(raw, 0, &field_names[0], &field_names[1])
                            .map(|(a, b)| vec![a, b])
                    } else {
                        let mut vals = vec![f64::NAN; field_names.len()];
                        let mut ok = true;
                        for (i, fname) in field_names.iter().enumerate() {
                            if let Some(n) = json_object_get_num(raw, 0, fname) {
                                vals[i] = n;
                            } else { ok = false; break; }
                        }
                        if ok { Some(vals) } else { None }
                    };
                    if let Some(vals) = got {
                        let mut result = is_and;
                        for (idx, op, threshold) in &cmp_spec {
                            let v = vals[*idx];
                            let cmp_result = match op {
                                BinOp::Gt => v > *threshold, BinOp::Lt => v < *threshold,
                                BinOp::Ge => v >= *threshold, BinOp::Le => v <= *threshold,
                                BinOp::Eq => v == *threshold, BinOp::Ne => v != *threshold,
                                _ => unreachable!(),
                            };
                            if is_and {
                                if !cmp_result { result = false; break; }
                            } else {
                                if cmp_result { result = true; break; }
                            }
                        }
                        compact_buf.extend_from_slice(if result { b"true\n" } else { b"false\n" });
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sb_field, ref sb_name, ref sb_arg)) = field_str_builtin {
                let arg_bytes = sb_arg.as_bytes();
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sb_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && !val[1..val.len()-1].contains(&b'\\')
                        {
                            let content = &val[1..val.len()-1];
                            match sb_name.as_str() {
                                "startswith" => {
                                    if content.len() >= arg_bytes.len() && &content[..arg_bytes.len()] == arg_bytes {
                                        compact_buf.extend_from_slice(b"true\n");
                                    } else {
                                        compact_buf.extend_from_slice(b"false\n");
                                    }
                                }
                                "endswith" => {
                                    if content.len() >= arg_bytes.len() && &content[content.len()-arg_bytes.len()..] == arg_bytes {
                                        compact_buf.extend_from_slice(b"true\n");
                                    } else {
                                        compact_buf.extend_from_slice(b"false\n");
                                    }
                                }
                                "ltrimstr" => {
                                    compact_buf.push(b'"');
                                    if content.len() >= arg_bytes.len() && &content[..arg_bytes.len()] == arg_bytes {
                                        compact_buf.extend_from_slice(&content[arg_bytes.len()..]);
                                    } else {
                                        compact_buf.extend_from_slice(content);
                                    }
                                    compact_buf.push(b'"');
                                    compact_buf.push(b'\n');
                                }
                                "rtrimstr" => {
                                    compact_buf.push(b'"');
                                    if content.len() >= arg_bytes.len() && &content[content.len()-arg_bytes.len()..] == arg_bytes {
                                        compact_buf.extend_from_slice(&content[..content.len()-arg_bytes.len()]);
                                    } else {
                                        compact_buf.extend_from_slice(content);
                                    }
                                    compact_buf.push(b'"');
                                    compact_buf.push(b'\n');
                                }
                                "split" => {
                                    compact_buf.push(b'[');
                                    if arg_bytes.is_empty() {
                                        for (j, &byte) in content.iter().enumerate() {
                                            if j > 0 { compact_buf.push(b','); }
                                            compact_buf.push(b'"');
                                            compact_buf.push(byte);
                                            compact_buf.push(b'"');
                                        }
                                    } else {
                                        let mut pos = 0;
                                        let mut first = true;
                                        while pos <= content.len() {
                                            if !first { compact_buf.push(b','); }
                                            first = false;
                                            let next = if pos + arg_bytes.len() <= content.len() {
                                                content[pos..].windows(arg_bytes.len())
                                                    .position(|w| w == arg_bytes)
                                                    .map(|i| pos + i)
                                            } else { None };
                                            compact_buf.push(b'"');
                                            if let Some(found) = next {
                                                compact_buf.extend_from_slice(&content[pos..found]);
                                                compact_buf.push(b'"');
                                                pos = found + arg_bytes.len();
                                            } else {
                                                compact_buf.extend_from_slice(&content[pos..]);
                                                compact_buf.push(b'"');
                                                break;
                                            }
                                        }
                                    }
                                    compact_buf.extend_from_slice(b"]\n");
                                }
                                _ => {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref ft_field, ref ft_pattern, ref ft_flags)) = field_test {
                let re_pattern = if let Some(flags) = ft_flags {
                    let mut prefix = String::from("(?");
                    for c in flags.chars() {
                        match c { 'i' | 'm' | 's' => prefix.push(c), _ => {} }
                    }
                    prefix.push(')');
                    prefix.push_str(ft_pattern);
                    prefix
                } else {
                    ft_pattern.clone()
                };
                let re = regex::Regex::new(&re_pattern);
                if let Ok(re) = re {
                    let content_bytes = content.as_bytes();
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, ft_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && !val[1..val.len()-1].contains(&b'\\')
                            {
                                let content = &val[1..val.len()-1];
                                let content_str = unsafe { std::str::from_utf8_unchecked(content) };
                                if re.is_match(content_str) {
                                    compact_buf.extend_from_slice(b"true\n");
                                } else {
                                    compact_buf.extend_from_slice(b"false\n");
                                }
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else {
                    // If for some reason we have a file path with invalid regex, fall through to JIT
                    let content_bytes = content.as_bytes();
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        Ok(())
                    })
                }
            } else if let Some((ref gs_field, gs_global, ref gs_pattern, ref gs_replacement, ref gs_flags)) = field_gsub {
                let re_pattern = if let Some(flags) = gs_flags {
                    let mut prefix = String::from("(?");
                    for c in flags.chars() {
                        match c { 'i' | 'm' | 's' => prefix.push(c), _ => {} }
                    }
                    prefix.push(')');
                    prefix.push_str(gs_pattern);
                    prefix
                } else {
                    gs_pattern.clone()
                };
                if let Ok(re) = regex::Regex::new(&re_pattern) {
                    let repl = gs_replacement.as_str();
                    let content_bytes = content.as_bytes();
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, gs_field) {
                            let val = &raw[vs..ve];
                            if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                                && !val[1..val.len()-1].contains(&b'\\')
                            {
                                let content_str = unsafe { std::str::from_utf8_unchecked(&val[1..val.len()-1]) };
                                let result = if gs_global {
                                    re.replace_all(content_str, repl)
                                } else {
                                    re.replace(content_str, repl)
                                };
                                compact_buf.push(b'"');
                                for &b in result.as_bytes() {
                                    match b {
                                        b'"' => compact_buf.extend_from_slice(b"\\\""),
                                        b'\\' => compact_buf.extend_from_slice(b"\\\\"),
                                        b'\n' => compact_buf.extend_from_slice(b"\\n"),
                                        b'\r' => compact_buf.extend_from_slice(b"\\r"),
                                        b'\t' => compact_buf.extend_from_slice(b"\\t"),
                                        c if c < 0x20 => { use std::io::Write; let _ = write!(compact_buf, "\\u{:04x}", c); }
                                        _ => compact_buf.push(b),
                                    }
                                }
                                compact_buf.extend_from_slice(b"\"\n");
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else {
                    let content_bytes = content.as_bytes();
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        Ok(())
                    })
                }
            } else if let Some((ref ff_field, ref ff_format)) = field_format {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, ff_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && !val[1..val.len()-1].contains(&b'\\')
                        {
                            let content = &val[1..val.len()-1];
                            match ff_format.as_str() {
                                "text" => {
                                    compact_buf.extend_from_slice(val);
                                    compact_buf.push(b'\n');
                                }
                                "json" => {
                                    push_tojson_raw(&mut compact_buf, val);
                                    compact_buf.push(b'\n');
                                }
                                _ => {
                                    compact_buf.push(b'"');
                                    match ff_format.as_str() {
                                        "base64" => base64_encode_to(content, &mut compact_buf),
                                        "uri" => uri_encode_to(content, &mut compact_buf),
                                        "html" => html_encode_to(content, &mut compact_buf),
                                        _ => {}
                                    }
                                    compact_buf.extend_from_slice(b"\"\n");
                                }
                            }
                        } else {
                            // Non-string field values (numbers, booleans, null)
                            match ff_format.as_str() {
                                "json" | "text" => {
                                    compact_buf.push(b'"');
                                    compact_buf.extend_from_slice(val);
                                    compact_buf.extend_from_slice(b"\"\n");
                                }
                                _ => {
                                    let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                                }
                            }
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref lt_field, ref lt_prefix)) = field_ltrimstr_tonumber {
                let prefix_bytes = lt_prefix.as_bytes();
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, lt_field) {
                        let val = &raw[vs..ve];
                        if val.len() >= 2 && val[0] == b'"' && val[val.len()-1] == b'"'
                            && !val[1..val.len()-1].contains(&b'\\')
                        {
                            let content = &val[1..val.len()-1];
                            let num_str = if content.len() >= prefix_bytes.len() && &content[..prefix_bytes.len()] == prefix_bytes {
                                &content[prefix_bytes.len()..]
                            } else {
                                content
                            };
                            if let Ok(n) = fast_float::parse::<f64, _>(num_str) {
                                push_jq_number_bytes(&mut compact_buf, n);
                                compact_buf.push(b'\n');
                            } else {
                                compact_buf.extend_from_slice(b"null\n");
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref sc_field, ref suffix)) = field_str_concat {
                // String concat fast path: extract field's raw string bytes and append suffix.
                // Only works when the field value is a non-escaped string.
                let content_bytes = content.as_bytes();
                // Pre-escape suffix for JSON output
                let suffix_needs_escape = suffix.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20);
                let suffix_escaped: Vec<u8> = if suffix_needs_escape {
                    let mut buf = Vec::new();
                    for &b in suffix.as_bytes() {
                        match b {
                            b'"' => buf.extend_from_slice(b"\\\""),
                            b'\\' => buf.extend_from_slice(b"\\\\"),
                            b'\n' => buf.extend_from_slice(b"\\n"),
                            b'\r' => buf.extend_from_slice(b"\\r"),
                            b'\t' => buf.extend_from_slice(b"\\t"),
                            c if c < 0x20 => { let _ = std::fmt::Write::write_fmt(&mut String::new(), format_args!("\\u{:04x}", c)); }
                            _ => buf.push(b),
                        }
                    }
                    buf
                } else {
                    suffix.as_bytes().to_vec()
                };
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if raw[0] == b'{' {
                        if let Some((vs, ve)) = json_object_get_field_raw(raw, 0, sc_field) {
                            let val = &raw[vs..ve];
                            // Only use fast path for simple strings (quoted, no backslash)
                            if val[0] == b'"' && !val[1..ve-vs-1].contains(&b'\\') {
                                // Copy everything except trailing quote, append suffix + quote + newline
                                compact_buf.extend_from_slice(&val[..val.len()-1]);
                                compact_buf.extend_from_slice(&suffix_escaped);
                                compact_buf.extend_from_slice(b"\"\n");
                            } else {
                                // Fall back for non-string or escaped string values
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else {
                            // Field not found → null + "str" = "str"
                            compact_buf.push(b'"');
                            compact_buf.extend_from_slice(&suffix_escaped);
                            compact_buf.extend_from_slice(b"\"\n");
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref join_parts, ref join_sep)) = array_join {
                let content_bytes = content.as_bytes();
                let field_names: Vec<&str> = join_parts.iter()
                    .filter(|(is_lit, _)| !*is_lit)
                    .map(|(_, name)| name.as_str())
                    .collect();
                let sep_bytes = join_sep.as_bytes();
                let escaped_sep = json_escape_bytes(sep_bytes);
                let escaped_lits: Vec<Option<Vec<u8>>> = join_parts.iter().map(|(is_lit, s)| {
                    if *is_lit { Some(json_escape_bytes(s.as_bytes())) } else { None }
                }).collect();
                let mut ranges_buf = vec![(0usize, 0usize); field_names.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if raw[0] != b'{' {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        return Ok(());
                    }
                    if json_object_get_fields_raw_buf(raw, 0, &field_names, &mut ranges_buf) {
                        compact_buf.push(b'"');
                        let mut field_idx = 0;
                        for (i, (is_lit, _)) in join_parts.iter().enumerate() {
                            if i > 0 { compact_buf.extend_from_slice(&escaped_sep); }
                            if *is_lit {
                                compact_buf.extend_from_slice(escaped_lits[i].as_ref().unwrap());
                            } else {
                                let (vs, ve) = ranges_buf[field_idx];
                                field_idx += 1;
                                let val = &raw[vs..ve];
                                if val[0] == b'"' && val.len() >= 2 {
                                    compact_buf.extend_from_slice(&val[1..val.len()-1]);
                                } else {
                                    compact_buf.extend_from_slice(val);
                                }
                            }
                        }
                        compact_buf.push(b'"');
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref remap_exprs, ref join_sep)) = remap_tostring_join {
                let content_bytes = content.as_bytes();
                let mut all_fields: Vec<String> = Vec::new();
                for rexpr in remap_exprs {
                    for f in remap_expr_fields(rexpr) {
                        if !all_fields.iter().any(|x| x == f) {
                            all_fields.push(f.to_string());
                        }
                    }
                }
                let field_strs: Vec<&str> = all_fields.iter().map(|s| s.as_str()).collect();
                let mut field_idx = std::collections::HashMap::new();
                for (i, f) in all_fields.iter().enumerate() { field_idx.insert(f.clone(), i); }
                let resolved: Vec<ResolvedRemap> = remap_exprs.iter().map(|rexpr| resolve_one_remap(rexpr, &field_idx)).collect();
                let sep_bytes = join_sep.as_bytes();
                let escaped_sep = json_escape_bytes(sep_bytes);
                let mut ranges = vec![(0usize, 0usize); all_fields.len()];
                let mut num_buf = Vec::<u8>::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if raw[0] != b'{' {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        return Ok(());
                    }
                    if json_object_get_fields_raw_buf(raw, 0, &field_strs, &mut ranges) {
                        compact_buf.push(b'"');
                        for (i, res) in resolved.iter().enumerate() {
                            if i > 0 { compact_buf.extend_from_slice(&escaped_sep); }
                            num_buf.clear();
                            emit_resolved_value(&mut num_buf, res, raw, &ranges);
                            if num_buf.len() >= 2 && num_buf[0] == b'"' && num_buf[num_buf.len()-1] == b'"' {
                                compact_buf.extend_from_slice(&num_buf[1..num_buf.len()-1]);
                            } else {
                                compact_buf.extend_from_slice(&json_escape_bytes(&num_buf));
                            }
                        }
                        compact_buf.push(b'"');
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref interp_parts) = string_interp_fields {
                let content_bytes = content.as_bytes();
                let field_names: Vec<&str> = interp_parts.iter()
                    .filter(|(is_lit, _)| !*is_lit)
                    .map(|(_, name)| name.as_str())
                    .collect();
                let escaped_lits: Vec<Option<Vec<u8>>> = interp_parts.iter().map(|(is_lit, s)| {
                    if *is_lit {
                        let mut buf = Vec::new();
                        for &b in s.as_bytes() {
                            match b {
                                b'"' => buf.extend_from_slice(b"\\\""),
                                b'\\' => buf.extend_from_slice(b"\\\\"),
                                b'\n' => buf.extend_from_slice(b"\\n"),
                                b'\r' => buf.extend_from_slice(b"\\r"),
                                b'\t' => buf.extend_from_slice(b"\\t"),
                                c if c < 0x20 => { let _ = write!(buf, "\\u{:04x}", c); }
                                _ => buf.push(b),
                            }
                        }
                        Some(buf)
                    } else {
                        None
                    }
                }).collect();
                let mut ranges_buf = vec![(0usize, 0usize); field_names.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if raw[0] != b'{' {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        return Ok(());
                    }
                    if json_object_get_fields_raw_buf(raw, 0, &field_names, &mut ranges_buf) {
                        compact_buf.push(b'"');
                        let mut field_idx = 0;
                        for (i, (is_lit, _)) in interp_parts.iter().enumerate() {
                            if *is_lit {
                                compact_buf.extend_from_slice(escaped_lits[i].as_ref().unwrap());
                            } else {
                                let (vs, ve) = ranges_buf[field_idx];
                                field_idx += 1;
                                let val = &raw[vs..ve];
                                if val[0] == b'"' && val.len() >= 2 {
                                    compact_buf.extend_from_slice(&val[1..val.len()-1]);
                                } else {
                                    compact_buf.extend_from_slice(val);
                                }
                            }
                        }
                        compact_buf.push(b'"');
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref sac_parts) = string_add_chain {
                use jq_jit::interpreter::StringAddPart;
                let content_bytes = content.as_bytes();
                let mut field_names: Vec<&str> = Vec::new();
                let mut field_idx_map = std::collections::HashMap::new();
                for part in sac_parts.iter() {
                    let name = match part {
                        StringAddPart::Field(f) | StringAddPart::FieldToString(f) => f.as_str(),
                        _ => continue,
                    };
                    if !field_idx_map.contains_key(name) {
                        field_idx_map.insert(name, field_names.len());
                        field_names.push(name);
                    }
                }
                // Pre-compute actions for hot loop
                let mut actions: Vec<(usize, bool)> = Vec::with_capacity(sac_parts.len());
                let mut lit_bufs: Vec<Vec<u8>> = Vec::new();
                for part in sac_parts.iter() {
                    match part {
                        StringAddPart::Literal(s) => {
                            let mut buf = Vec::new();
                            for &b in s.as_bytes() {
                                match b {
                                    b'"' => buf.extend_from_slice(b"\\\""),
                                    b'\\' => buf.extend_from_slice(b"\\\\"),
                                    b'\n' => buf.extend_from_slice(b"\\n"),
                                    b'\r' => buf.extend_from_slice(b"\\r"),
                                    b'\t' => buf.extend_from_slice(b"\\t"),
                                    c if c < 0x20 => { let _ = write!(buf, "\\u{:04x}", c); }
                                    _ => buf.push(b),
                                }
                            }
                            let lit_idx = lit_bufs.len();
                            lit_bufs.push(buf);
                            actions.push((lit_idx, false));
                        }
                        StringAddPart::Field(f) | StringAddPart::FieldToString(f) => {
                            actions.push((field_idx_map[f.as_str()], true));
                        }
                    }
                }
                let mut ranges_buf = vec![(0usize, 0usize); field_names.len()];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if json_object_get_fields_raw_buf(raw, 0, &field_names, &mut ranges_buf) {
                        compact_buf.push(b'"');
                        for &(idx, is_field) in &actions {
                            if is_field {
                                let (vs, ve) = ranges_buf[idx];
                                let val = &raw[vs..ve];
                                if val[0] == b'"' && val.len() >= 2 {
                                    compact_buf.extend_from_slice(&val[1..val.len()-1]);
                                } else {
                                    compact_buf.extend_from_slice(val);
                                }
                            } else {
                                compact_buf.extend_from_slice(&lit_bufs[idx]);
                            }
                        }
                        compact_buf.extend_from_slice(b"\"\n");
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_length {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(len) = json_value_length(raw, 0) {
                        push_jq_number_bytes(&mut compact_buf, len as f64);
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_keys {
                let content_bytes = content.as_bytes();
                let mut tmp = Vec::new();
                let mut keys_buf: Vec<(usize, usize)> = Vec::new();
                let mut cached_output: Vec<u8> = Vec::new();
                let mut cached_keys: Vec<Vec<u8>> = Vec::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if !cached_keys.is_empty() {
                        if let Some(extracted) = json_object_extract_keys_only(raw, 0, &mut keys_buf) {
                            if extracted == cached_keys.len() {
                                let mut same = true;
                                for (i, (ks, ke)) in keys_buf.iter().enumerate() {
                                    if &raw[*ks..*ke] != cached_keys[i].as_slice() {
                                        same = false;
                                        break;
                                    }
                                }
                                if same {
                                    if use_pretty_buf {
                                        push_json_pretty_raw(&mut compact_buf, &cached_output, 2, false);
                                        compact_buf.push(b'\n');
                                    } else {
                                        compact_buf.extend_from_slice(&cached_output);
                                        compact_buf.push(b'\n');
                                    }
                                    if compact_buf.len() >= 1 << 17 {
                                        let _ = out.write_all(&compact_buf);
                                        compact_buf.clear();
                                    }
                                    return Ok(());
                                }
                            }
                        }
                    }
                    if use_pretty_buf {
                        tmp.clear();
                        if json_object_keys_to_buf_reuse(raw, 0, &mut tmp, &mut keys_buf) {
                            let len = tmp.len();
                            if len > 0 && tmp[len-1] == b'\n' { tmp.truncate(len-1); }
                            if cached_keys.is_empty() {
                                cached_output = tmp.clone();
                                let mut unsorted: Vec<(usize, usize)> = Vec::new();
                                if let Some(_) = json_object_extract_keys_only(raw, 0, &mut unsorted) {
                                    cached_keys = unsorted.iter().map(|(s, e)| raw[*s..*e].to_vec()).collect();
                                }
                            }
                            push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let before = compact_buf.len();
                        if json_object_keys_to_buf_reuse(raw, 0, &mut compact_buf, &mut keys_buf) {
                            if cached_keys.is_empty() {
                                let end_pos = compact_buf.len();
                                cached_output = compact_buf[before..end_pos-1].to_vec();
                                let mut unsorted: Vec<(usize, usize)> = Vec::new();
                                if let Some(_) = json_object_extract_keys_only(raw, 0, &mut unsorted) {
                                    cached_keys = unsorted.iter().map(|(s, e)| raw[*s..*e].to_vec()).collect();
                                }
                            }
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_keys_unsorted {
                let content_bytes = content.as_bytes();
                let mut tmp = Vec::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if use_pretty_buf {
                        tmp.clear();
                        if json_object_keys_unsorted_to_buf(raw, 0, &mut tmp) {
                            let len = tmp.len();
                            if len > 0 && tmp[len-1] == b'\n' { tmp.truncate(len-1); }
                            push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else if !json_object_keys_unsorted_to_buf(raw, 0, &mut compact_buf) {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref hf) = has_field {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if let Some(found) = json_object_has_key(raw, 0, hf) {
                        compact_buf.extend_from_slice(if found { b"true\n" } else { b"false\n" });
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref hm_fields, hm_is_and)) = has_multi {
                let field_refs: Vec<&str> = hm_fields.iter().map(|s| s.as_str()).collect();
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let result = if hm_is_and {
                        json_object_has_all_keys(raw, 0, &field_refs)
                    } else {
                        json_object_has_any_key(raw, 0, &field_refs)
                    };
                    if let Some(found) = result {
                        compact_buf.extend_from_slice(if found { b"true\n" } else { b"false\n" });
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_type {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    compact_buf.extend_from_slice(json_type_byte(raw[0]));
                    compact_buf.push(b'\n');
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref df) = del_field {
                let content_bytes = content.as_bytes();
                let mut tmp = Vec::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if use_pretty_buf {
                        tmp.clear();
                        if json_object_del_field(raw, 0, df, &mut tmp) {
                            push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else if json_object_del_field(raw, 0, df, &mut compact_buf) {
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref merge_pairs) = obj_merge_lit {
                let content_bytes = content.as_bytes();
                let mut tmp = Vec::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if use_pretty_buf {
                        tmp.clear();
                        if json_object_merge_literal(raw, 0, merge_pairs, &mut tmp) {
                            push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else if json_object_merge_literal(raw, 0, merge_pairs, &mut compact_buf) {
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref out_key, ref nfields, ref arith)) = obj_merge_computed {
                let content_bytes = content.as_bytes();
                let mut tmp = Vec::new();
                let mut merge_pair = vec![(out_key.clone(), Vec::new())];
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let nf_count = nfields.len();
                    let ok = if nf_count == 1 {
                        if let Some(v) = json_object_get_num(raw, 0, &nfields[0]) {
                            let result = arith.eval(&[v]);
                            merge_pair[0].1.clear();
                            push_jq_number_bytes(&mut merge_pair[0].1, result);
                            true
                        } else { false }
                    } else if nf_count == 2 {
                        if let Some((a, b)) = json_object_get_two_nums(raw, 0, &nfields[0], &nfields[1]) {
                            let result = arith.eval(&[a, b]);
                            merge_pair[0].1.clear();
                            push_jq_number_bytes(&mut merge_pair[0].1, result);
                            true
                        } else { false }
                    } else { false };
                    if ok {
                        if use_pretty_buf {
                            tmp.clear();
                            if json_object_merge_literal(raw, 0, &merge_pair, &mut tmp) {
                                push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                                compact_buf.push(b'\n');
                            } else {
                                let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                                process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                            }
                        } else if json_object_merge_literal(raw, 0, &merge_pair, &mut compact_buf) {
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_each {
                let content_bytes = content.as_bytes();
                if use_pretty_buf {
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        if !json_each_value_cb(raw, 0, |vs, ve| {
                            let val = &raw[vs..ve];
                            emit_raw_ln!(&mut compact_buf, val);
                        }) {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                } else {
                    json_stream_raw(content, |start, end| {
                        let raw = &content_bytes[start..end];
                        if !json_each_value_raw(raw, 0, &mut compact_buf) {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                        if compact_buf.len() >= 1 << 17 {
                            let _ = out.write_all(&compact_buf);
                            compact_buf.clear();
                        }
                        Ok(())
                    })
                }
            } else if is_sort_keys {
                let content_bytes = content.as_bytes();
                let mut pairs_buf: Vec<(usize, usize, usize, usize)> = Vec::new();
                let mut tmp = Vec::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if use_pretty_buf {
                        tmp.clear();
                        if json_object_sort_keys(raw, 0, &mut tmp, &mut pairs_buf) {
                            push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else {
                        if json_object_sort_keys(raw, 0, &mut compact_buf, &mut pairs_buf) {
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_to_entries {
                let content_bytes = content.as_bytes();
                let mut tmp = Vec::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if use_pretty_buf {
                        tmp.clear();
                        if json_to_entries_raw(raw, 0, &mut tmp) {
                            let len = tmp.len();
                            if len > 0 && tmp[len-1] == b'\n' { tmp.truncate(len-1); }
                            push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else if !json_to_entries_raw(raw, 0, &mut compact_buf) {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref rte_pairs) = remap_to_entries {
                let content_bytes = content.as_bytes();
                let rte_src: Vec<&str> = rte_pairs.iter().map(|(_, src)| src.as_str()).collect();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    let mut vals: Vec<Option<(usize, usize)>> = Vec::with_capacity(rte_src.len());
                    for src in &rte_src {
                        vals.push(json_object_get_field_raw(raw, 0, src));
                    }
                    compact_buf.push(b'[');
                    let mut first = true;
                    for (i, (out_key, _)) in rte_pairs.iter().enumerate() {
                        if !first { compact_buf.push(b','); }
                        first = false;
                        compact_buf.extend_from_slice(b"{\"key\":");
                        compact_buf.push(b'"');
                        for &b in out_key.as_bytes() {
                            match b {
                                b'"' => compact_buf.extend_from_slice(b"\\\""),
                                b'\\' => compact_buf.extend_from_slice(b"\\\\"),
                                _ => compact_buf.push(b),
                            }
                        }
                        compact_buf.extend_from_slice(b"\",\"value\":");
                        if let Some((vs, ve)) = vals[i] {
                            compact_buf.extend_from_slice(&raw[vs..ve]);
                        } else {
                            compact_buf.extend_from_slice(b"null");
                        }
                        compact_buf.push(b'}');
                    }
                    compact_buf.extend_from_slice(b"]\n");
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some((ref we_op, we_threshold)) = with_entries_select {
                let content_bytes = content.as_bytes();
                use jq_jit::ir::BinOp;
                let cmp_byte = match we_op {
                    BinOp::Gt => b'>',
                    BinOp::Ge => b'G',
                    BinOp::Lt => b'<',
                    BinOp::Le => b'L',
                    BinOp::Eq => b'=',
                    BinOp::Ne => b'!',
                    _ => b'>',
                };
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    json_with_entries_select_value_cmp(raw, 0, cmp_byte, we_threshold, &mut compact_buf);
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref type_name) = with_entries_type {
                let content_bytes = content.as_bytes();
                let mut tmp = Vec::new();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if use_pretty_buf {
                        tmp.clear();
                        if json_object_filter_by_value_type(raw, 0, type_name, &mut tmp) {
                            push_json_pretty_raw(&mut compact_buf, &tmp, 2, false);
                            compact_buf.push(b'\n');
                        } else {
                            let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                            process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                        }
                    } else if json_object_filter_by_value_type(raw, 0, type_name, &mut compact_buf) {
                        compact_buf.push(b'\n');
                    } else {
                        let v = json_to_value(unsafe { std::str::from_utf8_unchecked(raw) })?;
                        process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if is_tojson {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    push_tojson_raw(&mut compact_buf, raw);
                    compact_buf.push(b'\n');
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if filter.is_identity() && use_compact_buf && !exit_status {
                // Identity fast path: skip JSON parsing entirely, just validate structure
                // and copy raw bytes directly. Falls back to parse+serialize for non-compact input.
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    if is_json_compact(raw) {
                        compact_buf.extend_from_slice(raw);
                        compact_buf.push(b'\n');
                    } else {
                        push_json_compact_raw(&mut compact_buf, raw);
                        compact_buf.push(b'\n');
                    }
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if filter.is_identity() && use_pretty_buf && !exit_status {
                let content_bytes = content.as_bytes();
                json_stream_raw(content, |start, end| {
                    let raw = &content_bytes[start..end];
                    push_json_pretty_raw(&mut compact_buf, raw, indent_n, tab);
                    compact_buf.push(b'\n');
                    if compact_buf.len() >= 1 << 17 {
                        let _ = out.write_all(&compact_buf);
                        compact_buf.clear();
                    }
                    Ok(())
                })
            } else if let Some(ref pf) = projection_fields {
                let field_refs: Vec<&str> = pf.iter().map(|s| s.as_str()).collect();
                json_stream_project(content, &field_refs, |v| {
                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    pool_value(v);
                    Ok(())
                })
            } else if use_compact_buf {
                let content_bytes = content.as_bytes();
                json_stream_offsets(content, |v, start, end| {
                    let raw = &content_bytes[start..end];
                    process_input(&v, Some(raw), &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    pool_value(v);
                    Ok(())
                })
            } else {
                json_stream(content, |v| {
                    process_input(&v, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
                    pool_value(v);
                    Ok(())
                })
            };
            if let Err(e) = parse_result {
                eprintln!("jq: error: {}", e);
                process::exit(2);
            }
        }
        if slurp && !slurp_values.is_empty() {
            let arr = Value::Arr(std::rc::Rc::new(slurp_values));
            process_input(&arr, None, &mut out, &mut compact_buf, &mut any_output_false, &mut had_error);
        }
    }

    if !compact_buf.is_empty() {
        let _ = out.write_all(&compact_buf);
    }
    let _ = out.flush();

    if had_error {
        process::exit(5);
    }
    if exit_status && any_output_false {
        process::exit(5);
    }
}

fn print_usage() {
    eprintln!("Usage: jq-jit [OPTIONS] <FILTER> [FILE...]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -c, --compact-output   Compact output");
    eprintln!("  -r, --raw-output       Raw output (strings without quotes)");
    eprintln!("  -R, --raw-input        Raw input (each line is a string)");
    eprintln!("  -n, --null-input        Use null as input");
    eprintln!("  -s, --slurp            Slurp all inputs into array");
    eprintln!("  -S, --sort-keys        Sort object keys");
    eprintln!("  -e, --exit-status      Exit with non-zero if last output is false/null");
    eprintln!("  -f, --from-file FILE   Read filter from file");
    eprintln!("  --tab                  Use tabs for indentation");
    eprintln!("  --indent N             Use N spaces for indentation");
    eprintln!("  --arg NAME VALUE       Set variable $NAME to VALUE (string)");
    eprintln!("  --argjson NAME VALUE   Set variable $NAME to VALUE (JSON)");
    eprintln!("  --version              Show version");
    eprintln!("  -h, --help             Show this help");
}
