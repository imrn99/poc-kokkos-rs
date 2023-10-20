//!
//!
//!
//!

use std::fmt::Display;

// enums

#[derive(Debug)]
pub enum DispatchError {
    Serial,
    CPU,
    GPU,
}

impl Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::Serial => todo!(),
            DispatchError::CPU => todo!(),
            DispatchError::GPU => todo!(),
        }
    }
}

impl std::error::Error for DispatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DispatchError::Serial => todo!(),
            DispatchError::CPU => todo!(),
            DispatchError::GPU => todo!(),
        }
    }
}

// dispatch routines

pub fn serial() -> Result<(), DispatchError> {
    todo!()
}

cfg_if::cfg_if! {
    if #[cfg(threads)] {
        // OS threads backend
        pub fn cpu() -> Result<(), DispatchError> {
            todo!()
        }
    } else if #[cfg(rayon)] {
        // rayon backend
        pub fn cpu() -> Result<(), DispatchError> {
            todo!()
        }
    } else {
        // fallback impl
        pub fn cpu() -> Result<(), DispatchError> {
            todo!()
        }
    }
}

pub fn gpu() -> Result<(), DispatchError> {
    unimplemented!()
}
