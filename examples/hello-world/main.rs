#[cxx::bridge(namespace = "")]
mod ffi {
    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("hello-world/hello.hpp");

        fn say_hello();
    }
}

fn main() {
    println!("Hello from Rust!");

    ffi::say_hello();
}
