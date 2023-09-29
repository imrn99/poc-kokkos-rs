fn main() {
    cxx_build::bridge("src/main.rs")
        .file("src/hello.cpp")
        .flag_if_supported("-std=c++20")
        .compile("cxxbridge-demo");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/hello.cpp");
    println!("cargo:rerun-if-changed=include/hello.hpp");
}
