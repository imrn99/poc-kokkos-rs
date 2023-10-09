use std::sync::Arc;

/// Generic trait used to define data held by views.
pub trait SomeData {
    type DataType;
}

impl<A> SomeData for Vec<A> {
    type DataType = A;
}

impl<'a, A> SomeData for &'a [A] {
    type DataType = A;
}

impl<'a, A> SomeData for &'a mut [A] {
    type DataType = A;
}

impl<A> SomeData for Arc<Vec<A>> {
    type DataType = A;
}
