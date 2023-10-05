use std::env;

fn main() {
    // need to find a good default value for the compiler
    let compiler = env::var("CXX").unwrap_or("clang++".to_string());
    println!("{}", compiler);
    cxx_build::bridge("main.rs")
        .compiler(compiler)
        .file("hello.cpp")
        .flag_if_supported("-std=c++20")
        // compiler omp flags
        .flag_if_supported("-fopenmp") // gcc
        .flag_if_supported("-fopenmp=libomp") // clang
        .compile("poc-cc");

    // linker omp flags
    println!("cargo:rustc-link-arg=-L/opt/homebrew/Cellar/llvm/17.0.2/lib");
    println!("cargo:rustc-link-arg=-lomp");

    println!("cargo:rerun-if-changed=main.rs");
    println!("cargo:rerun-if-changed=many_hello.cpp");
    println!("cargo:rerun-if-changed=many_hello.hpp");
}
