//!
//!
//!
//!

use std::fmt::Display;

use super::parameters::ExecutionPolicy;

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
            DispatchError::Serial => write!(f, "error during serial dispatch"),
            DispatchError::CPU => write!(f, "error during cpu dispatch"),
            DispatchError::GPU => write!(f, "error during gpu dispatch"),
        }
    }
}

impl std::error::Error for DispatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DispatchError::Serial => None,
            DispatchError::CPU => None,
            DispatchError::GPU => None,
        }
    }
}

// dispatch routines

pub fn serial<const N: usize, F, Args>(
    execp: ExecutionPolicy<N>,
    func: F,
) -> Result<(), DispatchError>
where
    F: FnMut(Args),
{
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
