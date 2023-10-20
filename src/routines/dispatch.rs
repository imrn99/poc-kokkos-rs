//!
//!
//!
//!

pub const HANDLER: Handler = Handler {};

pub enum DispatchError {}

pub struct Handler {}

impl Handler {
    pub fn dispatch_serial() -> Result<(), DispatchError> {
        todo!()
    }

    cfg_if::cfg_if! {
        if #[cfg(threads)] {
            // OS threads backend
            pub fn dispatch_cpu() -> Result<(), DispatchError> {
                todo!()
            }
        } else if #[cfg(rayon)] {
            // rayon backend
            pub fn dispatch_cpu() -> Result<(), DispatchError> {
                todo!()
            }
        } else {
            // fallback impl
            pub fn dispatch_cpu() -> Result<(), DispatchError> {
                todo!()
            }
        }
    }

    pub fn dispatch_gpu() -> Result<(), DispatchError> {
        unimplemented!()
    }
}
