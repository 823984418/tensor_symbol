use std::sync::Arc;

use crate::core::assign::Assign;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::Tensor;

impl CpuOperator for Assign {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let [arg] = tensor.arguments() else { panic!() };
        context.compute(arg)
    }
}

#[test]
fn test() {
    let a = Tensor::scale(1.0).assign();
    assert_eq!(a.compute().unwrap().as_slice(), [1.0]);
}
