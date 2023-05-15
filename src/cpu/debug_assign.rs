use std::sync::Arc;

use crate::core::debug_assign::DebugAssign;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::Tensor;

impl CpuOperator for DebugAssign {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let [arg] = tensor.arguments() else { panic!() };
        let output = context.compute(arg);
        println!("{} {:?}", self.info(), output);
        output
    }
}
