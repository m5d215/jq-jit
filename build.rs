fn main() {
    // Try pkg-config first for portable library discovery
    let have_pkg_config = try_pkg_config();

    if !have_pkg_config {
        // Fallback: platform-specific library search paths
        if cfg!(target_os = "macos") {
            // Homebrew on Apple Silicon
            if std::path::Path::new("/opt/homebrew/lib").exists() {
                println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
                println!("cargo:include=/opt/homebrew/include");
            }
            // Homebrew on Intel Mac
            if std::path::Path::new("/usr/local/lib").exists() {
                println!("cargo:rustc-link-search=native=/usr/local/lib");
                println!("cargo:include=/usr/local/include");
            }
        } else if cfg!(target_os = "linux") {
            // Standard Linux library paths
            for path in &[
                "/usr/lib",
                "/usr/lib64",
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib/aarch64-linux-gnu",
                "/usr/local/lib",
            ] {
                if std::path::Path::new(path).exists() {
                    println!("cargo:rustc-link-search=native={}", path);
                }
            }
            for path in &["/usr/include", "/usr/local/include"] {
                if std::path::Path::new(path).exists() {
                    println!("cargo:include={}", path);
                }
            }
        }

        // Link libjq and oniguruma
        println!("cargo:rustc-link-lib=dylib=jq");
        println!("cargo:rustc-link-lib=dylib=onig");
    }
}

/// Try to find libjq and libonig via pkg-config.
/// Returns true if both were found.
fn try_pkg_config() -> bool {
    let jq = std::process::Command::new("pkg-config")
        .args(["--libs", "--cflags", "libjq"])
        .output();
    let onig = std::process::Command::new("pkg-config")
        .args(["--libs", "--cflags", "oniguruma"])
        .output();

    match (jq, onig) {
        (Ok(jq_out), Ok(onig_out)) if jq_out.status.success() && onig_out.status.success() => {
            // Parse and emit pkg-config flags
            for output in &[jq_out.stdout, onig_out.stdout] {
                let flags = String::from_utf8_lossy(output);
                for flag in flags.split_whitespace() {
                    if let Some(path) = flag.strip_prefix("-L") {
                        println!("cargo:rustc-link-search=native={}", path);
                    } else if let Some(lib) = flag.strip_prefix("-l") {
                        println!("cargo:rustc-link-lib=dylib={}", lib);
                    } else if let Some(path) = flag.strip_prefix("-I") {
                        println!("cargo:include={}", path);
                    }
                }
            }
            true
        }
        _ => false,
    }
}
