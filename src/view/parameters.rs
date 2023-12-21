//! view parameterization code
//!
//! This module contains all code used to parameterize views as well as internal
//! types used for [`View`][super::View] implementation. Some parameters are direct
//! Kokkos replicates while others are Rust-specific.
//!
//! Currently supported parameters include:
//!
//! - Memory layout
//! - Memory space
//!

use std::{
    alloc::{alloc, dealloc, Layout},
    fmt::Debug,
};

#[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
use atomic::Atomic;

use super::ViewError;

/// Maximum possible depth (i.e. number of dimensions) for a view.
pub const MAX_VIEW_DEPTH: usize = 8;

// ~~~~~~~~~ Raw data ~~~~~~~~~

/// Supertrait with common trait that elements of a View should implement.
pub trait DataTraits: Debug + Clone + Copy + Default + Sized {}

impl DataTraits for f64 {}
impl DataTraits for f32 {}

#[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
/// Generic alias for elements of type `T` of a View.
///
/// This alias automatically changes according to features to ensure thread-safety
/// of Views. There are two possible values:
///
/// - any feature enabled: `InnerDataType<T> = Atomic<T>`. By adding the atomic wrapping,
/// operations on views can be implemented using thread-safe methods.
/// - no feature enabled: `InnerDataType<T> = T`.
///
/// **Current version**: no feature
pub type InnerDataType<T> = T;

#[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
/// Generic alias for elements of type `T` of a View.
///
/// This alias automatically changes according to features to ensure thread-safety
/// of Views. There are two possible values:
///
/// - any feature enabled: `InnerDataType<T> = Atomic<T>`. By adding the atomic wrapping,
/// operations on views can be implemented using thread-safe methods.
/// - no feature enabled: `InnerDataType<T> = T`.
///
/// **Current version**: thread-safe
pub type InnerDataType<T> = Atomic<T>;

/// Internal data holding type for views
///
/// From a memory perspective, views are simply flat arrays. In order to handle
/// runtime-known dimensions,the implementation requires something more flexible
/// than an actual array as the storage type.
///
/// This structure is a stripped down version of a vector implementation, with
/// enough utilities for our purpose. It operates using a pointer and save the
/// [Layout] used for allocation to make sure deallocation is handled properly.
///
/// The pointer and layout fields are set as private since direct manipulation
/// of these would create safety issues.
#[derive(Debug)]
pub struct ViewData<T>
where
    T: DataTraits,
{
    #[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
    /// Pointer to the start of the allocated space.
    ///
    /// This field can have two different type according to enabled features:
    ///
    /// - any feature enabled: `ptr: *const InnerDataType<T>`
    /// - no feature enabled: `ptr: *mut InnerDataType<T>`
    ///
    /// current version: `*mut`
    ptr: *mut InnerDataType<T>,
    #[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
    /// Pointer to the start of the allocated space.
    ///
    /// This field can have two different type according to enabled features:
    ///
    /// - any feature enabled: `ptr: *const InnerDataType<T>`
    /// - no feature enabled: `ptr: *mut InnerDataType<T>`
    ///
    /// current version: `*const`
    ptr: *const InnerDataType<T>,
    /// Layout passed to the `alloc` method. Stored and used for the deallocation
    /// at the end of the structure's lifetime.
    lyt: Layout,
    /// Number of elements allocated. The actual memory size is
    /// `size * size_of::<T>()`.
    pub size: usize,
    /// Boolean used to identify if the current structure is held by a mirror
    /// view, in which case the memory should not be deallocated when the mirror
    /// is destroyed.
    pub mirror: bool,
}

impl<T: DataTraits> ViewData<T> {
    /// Constructor.
    ///
    /// Eventually, this constructor could call one allocator or the other
    /// according to the specified memory space. The [Allocator][std::alloc::Allocator]
    /// trait is not yet stabilized though, so no device-specific allocators are
    /// implementable at the moment.
    pub fn new(size: usize, memspace: MemorySpace) -> Self {
        let (ptr, lyt) = allocate_block::<InnerDataType<T>>(size, memspace).unwrap();
        Self {
            ptr,
            size,
            lyt,
            mirror: false,
        }
    }

    #[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
    /// Reading interface.
    ///
    /// Two different implementations of this method are defined in order to keep a
    /// consistent user API across features:
    ///
    /// - any feature enabled: implictly use an atomic load operation on top of pointer
    /// arithmetic. The load currently uses relaxed ordering, this may change.
    /// - no feature enabled: uses pointer arithmetic.
    ///
    /// The pointer arithmetic is simply an addition. It is preceded by a check to
    /// ensure that the provided index lands in the allocated space.
    ///
    /// **Current version**: no-feature
    pub fn get(&self, idx: usize) -> T {
        assert!(idx < self.size);
        unsafe { *self.ptr.add(idx).as_ref().unwrap() }
    }

    #[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
    /// Reading interface.
    ///
    /// Two different implementations of this method are defined in order to keep a
    /// consistent user API across features:
    ///
    /// - any feature enabled: implictly use an atomic load operation on top of pointer
    /// arithmetic. The load currently uses relaxed ordering, this may change.
    /// - no feature enabled: uses pointer arithmetic.
    ///
    /// The pointer arithmetic is simply an addition. It is preceded by a check to
    /// ensure that the provided index lands in the allocated space.
    ///
    /// **Current version**: thread-safe
    pub fn get(&self, idx: usize) -> T {
        assert!(idx < self.size);
        unsafe {
            self.ptr
                .add(idx)
                .as_ref()
                .unwrap()
                .load(atomic::Ordering::Relaxed)
        }
    }

    #[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
    /// Writing interface.
    ///
    /// Two different implementations of this method are defined in order to satisfy
    /// the (im)mutability requirements when using parallelization features & keep a
    /// consistent user API:
    ///
    /// - any feature enabled: implictly use an atomic store operation on top of pointer
    /// arithmetic. The store currently uses relaxed ordering, this may change.
    /// - no feature enabled: uses pointer arithmetic.
    ///
    /// The pointer arithmetic is simply an addition. It is preceded by a check to
    /// ensure that the provided index lands in the allocated space.
    ///
    /// **Current version**: no-feature
    pub fn set(&mut self, idx: usize, val: T) {
        assert!(idx < self.size);
        let targ = unsafe { self.ptr.add(idx) };
        unsafe { *targ = val };
    }

    #[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
    /// Writing interface.
    ///
    /// Two different implementations of this method are defined in order to satisfy
    /// the (im)mutability requirements when using parallelization features & keep a
    /// consistent user API:
    ///
    /// - any feature enabled: implictly use an atomic store operation on top of pointer
    /// arithmetic. The store currently uses relaxed ordering, this may change.
    /// - no feature enabled: uses pointer arithmetic.
    ///
    /// The pointer arithmetic is simply an addition. It is preceded by a check to
    /// ensure that the provided index lands in the allocated space.
    ///
    /// **Current version**: thread-safe
    pub fn set(&self, idx: usize, val: T) {
        assert!(idx < self.size);
        let targ = unsafe { self.ptr.add(idx) };
        unsafe { (*targ).store(val, atomic::Ordering::Relaxed) };
    }

    /// Convenience method. Fill the allocated space with the specified value.
    pub fn fill(&mut self, val: T) {
        for idx in 0..self.size {
            self.set(idx, val);
        }
    }

    /// Convenience method. Fill the allocated space with the specified values.
    /// The slice passed should have the same size as the allocated space.
    pub fn take_vals(&mut self, vals: &[T]) {
        assert_eq!(vals.len(), self.size);
        vals.iter()
            .enumerate()
            .for_each(|(idx, val)| self.set(idx, *val))
    }

    /// Convenience method. Checks whether two [ViewData] structures point to
    /// the same memory space.
    pub fn ptr_is_eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T: DataTraits> Drop for ViewData<T> {
    fn drop(&mut self) {
        // we should deallocate memory only if we're not a mirror
        if !self.mirror {
            unsafe { dealloc(self.ptr as *mut u8, self.lyt) }
        }
    }
}

/// Cheesy fix used to share raw pointers across threads.
///
/// This should not pose any problems as long as raw pointers are
/// not accessible.
unsafe impl<T: DataTraits> Sync for ViewData<T> {}

/// Cheesy fix used to share raw pointers across threads.
///
/// This should not pose any problems as long as raw pointers are
/// not accessible.
unsafe impl<T: DataTraits> Send for ViewData<T> {}

// ~~~~~~~~~ Memory layout ~~~~~~~~~

/// Enum used to represent data layout. Struct enums is used in order to increase
/// readability.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum MemoryLayout<const N: usize> {
    /// Highest stride for the first index, decreasing stride as index increases.
    /// Exact stride for each index can be computed from dimensions at view initialization.
    #[default]
    Right,
    /// Lowest stride for the first index, increasing stride as index decreases.
    /// Exact stride for each index can be computed from dimensions at view initialization.
    Left,
    /// Custom stride for each index. Must be compatible with dimensions.
    Stride { s: [usize; N] },
}

/// Compute correct strides of each index using dimensions and specified layout.
pub fn compute_stride<const N: usize>(dim: &[usize; N], layout: &MemoryLayout<N>) -> [usize; N] {
    assert_eq!(N.clamp(1, MAX_VIEW_DEPTH), N); // 1 <= N <= MAX_N
    match layout {
        MemoryLayout::Right => {
            let mut stride = [1; N];

            let mut tmp: usize = 1;
            for i in (1..N).rev() {
                tmp *= dim[i];
                stride[N - i] = tmp;
            }

            stride.reverse();
            stride
        }
        MemoryLayout::Left => {
            let mut stride = [1; N];

            let mut tmp: usize = 1;
            for i in 0..N - 1 {
                tmp *= dim[i];
                stride[i + 1] = tmp;
            }

            stride
        }
        MemoryLayout::Stride { s } => *s,
    }
}

// ~~~~~~~~~ Memory space ~~~~~~~~~

/// Enum used to represent target memory space for views.
///
/// By using device-specific allocators, it may be possible to mimic
/// the templating implemented in Kokkos. As of now (1.74.1), the
/// [Allocator][std::alloc::Allocator] is not yet stabilized, but it
/// could make for some elegant wrapper around external tools (e.g.
/// CUDA).
pub enum MemorySpace {
    /// Allocate on CPU.
    CPU,
    /// Allocate on GPU. UNUSED
    GPU,
}

fn allocate_block<'a, T>(
    size: usize,
    memspace: MemorySpace,
) -> Result<(*mut T, Layout), ViewError<'a>> {
    // this is a required check to use the layout obtained below in the alloc()
    assert_ne!(size, 0);
    let res = Layout::array::<T>(size);
    match res {
        Ok(layout) => {
            // the idea here is that we can use one allocator or the other
            // according to the specified memory space.
            // the Allocator trait is not yet stabilized though, so no
            // device-specific allocators are implementable atm (1.74.1)
            match memspace {
                MemorySpace::CPU => {
                    let ptr = unsafe { alloc(layout) };
                    if ptr.is_null() {
                        return Err(ViewError::Allocation(
                            "OOM / mismatch allocator's size or alignment constraints",
                        ));
                    }
                    Ok((ptr as *mut T, layout)) // need to return both for deallocation later
                }
                MemorySpace::GPU => {
                    unimplemented!()
                }
            }
        }
        Err(_) => {
            // this happens "on arithmetic overflow"
            // or "when the total size would exceed isize::MAX"
            // according to std::alloc::Layout::array documentation
            Err(ViewError::Allocation(
                "arithmetic overflow / total size > isize::MAX",
            ))
        }
    }
}

// ~~~~~~~~~ Tests ~~~~~~~~~

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_right() {
        // dim = [n0, n1, n2, n3]
        let dim = [3, 4, 5, 6];

        let cmp_stride = compute_stride(&dim, &MemoryLayout::Right);
        // n3 * n2 * n1, n3 * n2, n3, 1
        let ref_stride: [usize; 4] = [6 * 5 * 4, 6 * 5, 6, 1];

        assert_eq!(cmp_stride, ref_stride);
    }

    #[test]
    fn stride_left() {
        // dim = [n0, n1, n2, n3]
        let dim = [3, 4, 5, 6];

        let cmp_stride = compute_stride(&dim, &MemoryLayout::Left);
        // 1, n0, n0 * n1, n0 * n1 * n2
        let ref_stride: [usize; 4] = [1, 3, 3 * 4, 3 * 4 * 5];

        assert_eq!(cmp_stride, ref_stride);
    }

    #[test]
    fn one_d_stride() {
        // 1d view (vector) of length 1
        let dim: [usize; 1] = [8];
        let ref_stride: [usize; 1] = [1];
        let mut cmp_stride = compute_stride(&dim, &MemoryLayout::Right);
        assert_eq!(ref_stride, cmp_stride);
        cmp_stride = compute_stride(&dim, &MemoryLayout::Left);
        assert_eq!(ref_stride, cmp_stride);
    }
}

#[cfg(test)]
mod viewdata {
    use super::*;

    const SIZE: usize = 10;
    const OOB: usize = 12;
    const IB: usize = 7;

    #[test]
    #[should_panic]
    fn out_of_bounds() {
        let dat: ViewData<f64> = ViewData::new(SIZE, MemorySpace::CPU);
        dat.get(OOB);
    }

    #[test]
    fn set_and_get() {
        cfg_if::cfg_if! {
            if #[cfg(any(feature = "threads", feature = "rayon", feature = "gpu"))] {
                let dat: ViewData<f64> = ViewData::new(SIZE, MemorySpace::CPU);
            } else {
                let mut dat: ViewData<f64> = ViewData::new(SIZE, MemorySpace::CPU);
            }
        }
        dat.set(IB, 10382.2891);
        assert_eq!(dat.get(IB), 10382.2891);
    }

    #[test]
    fn default_value() {
        let dat: ViewData<f64> = ViewData::new(SIZE, MemorySpace::CPU);
        (0..SIZE).for_each(|idx| assert_eq!(f64::default(), dat.get(idx)))
    }

    #[test]
    fn fill() {
        let mut dat: ViewData<f64> = ViewData::new(SIZE, MemorySpace::CPU);
        (0..SIZE).for_each(|idx| assert_eq!(f64::default(), dat.get(idx)));
        dat.fill(9534.284);
        (0..SIZE).for_each(|idx| assert_eq!(9534.284, dat.get(idx)));
    }
}
