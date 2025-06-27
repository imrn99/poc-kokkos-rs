use crate::functor::{ExecutionPolicy, ExecutionSpace, Schedule};

#[allow(private_bounds)]
pub trait ForFunctor<P: ExecutionPolicy>: Send + Sync {
    /// Kernel to execute.
    #[allow(private_interfaces)]
    fn execute(&self, args: P::KernelArgType);
}

#[allow(private_interfaces)]
impl<P: ExecutionPolicy, Closure: Fn(P::KernelArgType) + Send + Sync> ForFunctor<P> for Closure {
    fn execute(&self, args: <P as ExecutionPolicy>::KernelArgType) {
        self(args)
    }
}

#[allow(private_bounds)]
pub fn parallel_for<
    const EXECUTION_SPACE: ExecutionSpace,
    const SCHEDULE: Schedule,
    P: ExecutionPolicy,
    F: ForFunctor<P>,
>(
    label: Option<&str>,
    policy: P, // allows simplified sig for numerical ranges
    functor: F,
) {
    <P as ExecutionPolicy>::dispatch::<EXECUTION_SPACE, SCHEDULE, F>(functor);
}
