use std::sync::Arc;

/// Generic trait used to define data held by views.
pub trait SomeData: Default {
    type DataType;
}

impl<T> SomeData for Vec<T> {
    type DataType = T;
}

impl<'a, T> SomeData for &'a [T] {
    type DataType = T;
}

impl<'a, T> SomeData for &'a mut [T] {
    type DataType = T;
}

impl<T> SomeData for Arc<Vec<T>> {
    type DataType = T;
}
