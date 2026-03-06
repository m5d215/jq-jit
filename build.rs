fn main() {
    // Homebrew library/include search path (macOS)
    if cfg!(target_os = "macos") {
        if std::path::Path::new("/opt/homebrew/lib").exists() {
            println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
            println!("cargo:include=/opt/homebrew/include");
        } else if std::path::Path::new("/usr/local/lib").exists() {
            println!("cargo:rustc-link-search=native=/usr/local/lib");
            println!("cargo:include=/usr/local/include");
        }
    }

    // Link libjq (dynamic)
    println!("cargo:rustc-link-lib=dylib=jq");

    // Link oniguruma (jq dependency for regex support)
    println!("cargo:rustc-link-lib=dylib=onig");
}
