fn main() {
    // Homebrew library search path (macOS ARM64)
    println!("cargo:rustc-link-search=native=/opt/homebrew/lib");

    // Link libjq (dynamic)
    println!("cargo:rustc-link-lib=dylib=jq");

    // Link oniguruma (jq dependency for regex support)
    println!("cargo:rustc-link-lib=dylib=onig");

    // Include path for jq headers (used by bindgen in task 1-2)
    println!("cargo:include=/opt/homebrew/include");

    // Capture native static libs for AOT linking
    // Use `rustc --print native-static-libs` to discover what system libraries
    // are needed when linking a Rust staticlib.
    // We must compile a minimal crate (empty stdin) with --crate-type staticlib
    // because --print native-static-libs requires an actual compilation.
    use std::process::Command;
    let output = Command::new("rustc")
        .args([
            "--print",
            "native-static-libs",
            "--crate-type",
            "staticlib",
            "-o",
            "/dev/null",
            "-",
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .expect("failed to run rustc");
    let stderr = String::from_utf8_lossy(&output.stderr);
    // The output looks like: "note: native-static-libs: -lSystem -lc -lm"
    let libs = stderr
        .lines()
        .find(|line| line.contains("native-static-libs:"))
        .and_then(|line| line.split("native-static-libs:").nth(1))
        .unwrap_or("")
        .trim()
        .to_string();
    println!("cargo:rustc-env=NATIVE_STATIC_LIBS={}", libs);
}
