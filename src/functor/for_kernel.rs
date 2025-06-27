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
    _label: Option<&str>, // TODO: debug-level log?
    policy: P,
    functor: F,
) {
    match EXECUTION_SPACE {
        ExecutionSpace::Serial => {
            policy.dispatch_seq::<SCHEDULE, F>(functor);
        }
        ExecutionSpace::DeviceCPU => {
            policy.dispatch_cpu::<SCHEDULE, F>(functor);
        }
        ExecutionSpace::DeviceGPU => {
            policy.dispatch_gpu::<SCHEDULE, F>(functor);
        }
    }
}
