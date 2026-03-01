//! Criterion benchmarks for jq-jit: JIT vs libjq comparison.
//!
//! Measures:
//! - JIT compile time (bytecode → native code)
//! - JIT execution time (native code only)
//! - libjq execution time (compile + execute via jq_start/jq_next)
//!
//! Run with: `cargo bench`

use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};

use jq_jit::bytecode::JqState;
use jq_jit::codegen::compile_expr;
use jq_jit::compiler::bytecode_to_ir;
use jq_jit::jq_runner::run_jq;
use jq_jit::value::json_to_value;

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

fn generate_array(n: usize) -> String {
    let arr: Vec<String> = (0..n).map(|i| i.to_string()).collect();
    format!("[{}]", arr.join(","))
}

fn generate_objects(n: usize) -> String {
    let objs: Vec<String> = (0..n)
        .map(|i| {
            format!(
                r#"{{"name":"user{}","age":{},"score":{}}}"#,
                i,
                20 + i % 60,
                i * 10 % 100
            )
        })
        .collect();
    format!("[{}]", objs.join(","))
}

fn generate_csv_string(n: usize) -> String {
    let letters: Vec<String> = (0..n).map(|i| {
        let c = (b'a' + (i as u8 % 26)) as char;
        c.to_string()
    }).collect();
    format!("\"{}\"", letters.join(","))
}

// ---------------------------------------------------------------------------
// Helper: compile a filter to JIT (reusable across benchmarks)
// ---------------------------------------------------------------------------

struct JitCompiled {
    filter: jq_jit::codegen::JitFilter,
}

fn compile_jit(filter_str: &str) -> Option<JitCompiled> {
    let mut jq = JqState::new().ok()?;
    let bc = jq.compile(filter_str).ok()?;
    let ir = bytecode_to_ir(&bc).ok()?;
    let (jit_filter, _) = compile_expr(&ir).ok()?;
    Some(JitCompiled { filter: jit_filter })
}

// ---------------------------------------------------------------------------
// A. JIT compile time
// ---------------------------------------------------------------------------

fn bench_compile_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_time");

    let filters = [
        (". + 1", "simple_add"),
        (".foo", "field_access"),
        (".[] | . + 1", "each_add"),
        ("if . > 0 then . * 2 else . * -1 end", "if_then_else"),
        ("map(. * 2) | add", "map_mul_add"),
        ("reduce .[] as $x (0; . + $x)", "reduce_sum"),
    ];

    for (filter, name) in &filters {
        group.bench_with_input(BenchmarkId::new("jit_compile", name), filter, |b, f| {
            b.iter(|| {
                let mut jq = JqState::new().unwrap();
                let bc = jq.compile(f).unwrap();
                let ir = bytecode_to_ir(&bc).unwrap();
                let (jit_filter, _) = compile_expr(&ir).unwrap();
                std::hint::black_box(jit_filter);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// B. Execution time: JIT vs jq
// ---------------------------------------------------------------------------

// Category 1: Scalar operations
fn bench_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar");

    // `. + 1` on 42
    {
        let filter = ". + 1";
        let input_json = "42";
        let input = json_to_value(input_json).unwrap();

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "dot_plus_1"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "dot_plus_1"), |b| {
            b.iter(|| {
                std::hint::black_box(run_jq(filter, input_json).unwrap());
            });
        });
    }

    // `.foo + .bar * 2` on {"foo": 10, "bar": 20}
    {
        let filter = ".foo + .bar * 2";
        let input_json = r#"{"foo":10,"bar":20}"#;
        let input = json_to_value(input_json).unwrap();

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "field_arith"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "field_arith"), |b| {
            b.iter(|| {
                std::hint::black_box(run_jq(filter, input_json).unwrap());
            });
        });
    }

    group.finish();
}

// Category 2: Conditional
fn bench_conditional(c: &mut Criterion) {
    let mut group = c.benchmark_group("conditional");

    let filter = "if . > 0 then . * 2 else . * -1 end";
    let input_json = "5";
    let input = json_to_value(input_json).unwrap();

    if let Some(compiled) = compile_jit(filter) {
        group.bench_function(BenchmarkId::new("jit", "if_then_else"), |b| {
            b.iter(|| {
                std::hint::black_box(compiled.filter.execute(&input));
            });
        });
    }

    group.bench_function(BenchmarkId::new("libjq", "if_then_else"), |b| {
        b.iter(|| {
            std::hint::black_box(run_jq(filter, input_json).unwrap());
        });
    });

    group.finish();
}

// Category 3: Generator + pipe
fn bench_generator(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator");

    let arr100 = generate_array(100);
    let input100 = json_to_value(&arr100).unwrap();

    // `.[] | . + 1` on 100 elements
    {
        let filter = ".[] | . + 1";

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "each_add_100"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input100));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "each_add_100"), |b| {
            let json = arr100.clone();
            b.iter(|| {
                std::hint::black_box(run_jq(filter, &json).unwrap());
            });
        });
    }

    // `.[] | select(. > 50) | . * 2` on 100 elements
    {
        let filter = ".[] | select(. > 50) | . * 2";

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "each_select_mul_100"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input100));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "each_select_mul_100"), |b| {
            let json = arr100.clone();
            b.iter(|| {
                std::hint::black_box(run_jq(filter, &json).unwrap());
            });
        });
    }

    group.finish();
}

// Category 4: Aggregation
fn bench_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregation");

    let arr100 = generate_array(100);
    let arr1000 = generate_array(1000);
    let input100 = json_to_value(&arr100).unwrap();
    let input1000 = json_to_value(&arr1000).unwrap();

    // `map(. * 2)` on 100 elements
    {
        let filter = "map(. * 2)";

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "map_mul_100"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input100));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "map_mul_100"), |b| {
            let json = arr100.clone();
            b.iter(|| {
                std::hint::black_box(run_jq(filter, &json).unwrap());
            });
        });
    }

    // `map(. * 2) | add` on 100 elements
    {
        let filter = "map(. * 2) | add";

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "map_mul_add_100"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input100));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "map_mul_add_100"), |b| {
            let json = arr100.clone();
            b.iter(|| {
                std::hint::black_box(run_jq(filter, &json).unwrap());
            });
        });
    }

    // `reduce .[] as $x (0; . + $x)` on 1000 elements
    {
        let filter = "reduce .[] as $x (0; . + $x)";

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "reduce_sum_1000"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input1000));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "reduce_sum_1000"), |b| {
            let json = arr1000.clone();
            b.iter(|| {
                std::hint::black_box(run_jq(filter, &json).unwrap());
            });
        });
    }

    group.finish();
}

// Category 5: String processing
fn bench_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("string");

    // `split(",") | length` on "a,b,c,...,z"
    {
        let filter = r#"split(",") | length"#;
        let input_json = &generate_csv_string(26);
        let input = json_to_value(input_json).unwrap();

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "split_length"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "split_length"), |b| {
            let json = input_json.clone();
            b.iter(|| {
                std::hint::black_box(run_jq(filter, &json).unwrap());
            });
        });
    }

    // `ascii_downcase` on "HELLO WORLD"
    {
        let filter = "ascii_downcase";
        let input_json = r#""HELLO WORLD""#;
        let input = json_to_value(input_json).unwrap();

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "ascii_downcase"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "ascii_downcase"), |b| {
            b.iter(|| {
                std::hint::black_box(run_jq(filter, input_json).unwrap());
            });
        });
    }

    group.finish();
}

// Category 6: Large-scale data
fn bench_large_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_data");
    // Increase sample size time for large data benchmarks
    group.sample_size(30);

    let objs10k = generate_objects(10000);
    let input10k = json_to_value(&objs10k).unwrap();

    // `.[] | .name` on 10000 objects
    {
        let filter = ".[] | .name";

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "each_name_10k"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input10k));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "each_name_10k"), |b| {
            let json = objs10k.clone();
            b.iter(|| {
                std::hint::black_box(run_jq(filter, &json).unwrap());
            });
        });
    }

    // `[.[] | select(.age > 30)]` on 10000 objects
    // Note: This uses array constructor + each + select, which may not be supported.
    // Falls back to `.[] | select(.age > 30)` (without array wrapping) if compile fails.
    {
        let filter_wrapped = "[.[] | select(.age > 30)]";
        let filter_unwrapped = ".[] | select(.age > 30)";

        // Try wrapped first, fall back to unwrapped
        let (filter, label) = if compile_jit(filter_wrapped).is_some() {
            (filter_wrapped, "select_age_10k_arr")
        } else {
            (filter_unwrapped, "select_age_10k")
        };

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", label), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&input10k));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", label), |b| {
            let json = objs10k.clone();
            b.iter(|| {
                std::hint::black_box(run_jq(filter, &json).unwrap());
            });
        });
    }

    // `.[] | . + 1` on a single large object (1000 keys)
    // This tests object iteration with rt_iter_prepare optimization
    {
        let filter = ".[] | . + 1";

        // Generate a single object with 1000 numeric keys
        let kvs: Vec<String> = (0..1000)
            .map(|i| format!(r#""k{}": {}"#, i, i))
            .collect();
        let obj_json = format!("{{{}}}", kvs.join(","));
        let obj_input = json_to_value(&obj_json).unwrap();

        if let Some(compiled) = compile_jit(filter) {
            group.bench_function(BenchmarkId::new("jit", "obj_iter_1k"), |b| {
                b.iter(|| {
                    std::hint::black_box(compiled.filter.execute(&obj_input));
                });
            });
        }

        group.bench_function(BenchmarkId::new("libjq", "obj_iter_1k"), |b| {
            let json = obj_json.clone();
            b.iter(|| {
                std::hint::black_box(run_jq(filter, &json).unwrap());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups and main
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_compile_time,
    bench_scalar,
    bench_conditional,
    bench_generator,
    bench_aggregation,
    bench_string,
    bench_large_data,
);
criterion_main!(benches);
