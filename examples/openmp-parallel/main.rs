#[cxx::bridge(namespace = "")]
mod ffi {
    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("openmp-parallel/many_hello.hpp");

        fn say_many_hello();
    }
}

fn main() {
    println!("Hello from Rust!");

    ffi::say_many_hello();
}
