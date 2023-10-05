use std::env;

fn main() {
    // need to find a good default value for the compiler
    let compiler = env::var("CXX").unwrap_or("g++".to_string());

    // messy but works
    let ompflags = if compiler.contains("g++") {
        // g++ flavor
        "-fopenmp"
    } else if compiler.contains("clang++") {
        // clang++ flavor
        "-fopenmp=libomp"
    } else {
        panic!()
    };

    cxx_build::bridge("src/lib.rs")
        .compiler(compiler)
        .file("src/hello.cpp")
        .flag_if_supported("-std=c++20")
        .flag(ompflags) // clang
        .compile("poc-cc");

    // --- linking shenanigans ---
    println!("cargo:rustc-link-arg=-lomp");
    if env::consts::OS == "macos" {
        println!("cargo:rustc-link-arg=-L/opt/homebrew/opt/libomp/lib");
        println!("cargo:rustc-link-arg=-ld_classic");
    }
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/hello.cpp");
    println!("cargo:rerun-if-changed=include/hello.hpp");
}
