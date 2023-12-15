use std::env;

fn main() {
    // need to find a good default value for the compiler
    let compiler = env::var("CXX").unwrap_or("g++".to_string());

    // messy but works
    let ompflags = if compiler.contains("clang++") {
        // clang++ flavor
        "-fopenmp=libomp"
    } else if compiler.contains("g++") {
        // g++ flavor
        "-fopenmp"
    } else {
        unimplemented!()
    };

    cxx_build::bridge("src/lib.rs")
        .compiler(compiler)
        .file("src/cpp/hello.cpp")
        .flag_if_supported("-std=c++20")
        .flag(ompflags) // clang
        .compile("poc-cc");

    // --- linking shenanigans ---
    match env::consts::OS {
        "macos" => {
            println!("cargo:rustc-link-arg=-L/opt/homebrew/opt/libomp/lib");
            //println!("cargo:rustc-link-arg=-ld_classic");
            println!("cargo:rustc-link-arg=-lomp");
        }
        "linux" => {
            println!("cargo:rustc-link-arg=-I/usr/include");
            println!("cargo:rustc-link-arg=-L/usr/lib");
            println!("cargo:rustc-link-arg=-lgomp");
        }
        _ => unimplemented!(),
    }
    // main
    println!("cargo:rerun-if-changed=src/main.rs");
    // cpp files
    println!("cargo:rerun-if-changed=src/cpp/hello.cpp");
    // header files
    println!("cargo:rerun-if-changed=src/include/hello.hpp");
}
