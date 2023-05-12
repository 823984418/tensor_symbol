use std::sync::Arc;

use crate::core::reshape::Reshape;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::Tensor;

impl CpuOperator for Reshape {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let [arg] = tensor.arguments() else { panic!() };
        context.compute(arg)
    }
}
