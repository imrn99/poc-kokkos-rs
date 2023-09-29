#[cxx::bridge(namespace = "")]
mod ffi {

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("poc-kokkos-rs/include/hello.hpp");

        fn say_hello();
    }
}

fn main() {
    ffi::say_hello()
}
