use std::env;

fn main() {
    // need to find a good default value for the compiler
    let compiler = env::var("CXX").unwrap_or("clang++".to_string());
    println!("{}", compiler);
    cxx_build::bridge("src/lib.rs")
        .compiler(compiler)
        .file("src/hello.cpp")
        .flag_if_supported("-std=c++20")
        .flag("-fopenmp") // gcc not working ? missing this flag when compiling with g++
        .flag_if_supported("-fopenmp=libomp") // clang
        .compile("poc-cc");

    println!("cargo:rustc-link-arg=-L/opt/homebrew/Cellar/llvm/17.0.2/lib");
    println!("cargo:rustc-link-arg=-lomp");
    println!("cargo:rustc-link-arg=-ld_classic"); // xcode 15.0 broke the linker :))))))))
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/hello.cpp");
    println!("cargo:rerun-if-changed=include/hello.hpp");
}
