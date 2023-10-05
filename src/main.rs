use poc_kokkos_rs::ffi;

fn main() {
    ffi::say_hello();
    println!("Hello from Rust!");
    ffi::say_many_hello()
}
