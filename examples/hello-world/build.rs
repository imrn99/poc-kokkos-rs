use std::env;

fn main() {
    // need to find a good default value for the compiler
    let compiler = env::var("CXX").unwrap_or("clang++".to_string());
    println!("{}", compiler);
    cxx_build::bridge("main.rs")
        .compiler(compiler)
        .file("hello.cpp")
        .flag_if_supported("-std=c++20")
        .compile("poc-cc");

    println!("cargo:rerun-if-changed=main.rs");
    println!("cargo:rerun-if-changed=hello.cpp");
    println!("cargo:rerun-if-changed=hello.hpp");
}
