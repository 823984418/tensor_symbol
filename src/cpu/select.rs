use std::sync::Arc;

use crate::core::select::Select;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::Tensor;

impl CpuOperator for Select {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let [cond, pos, neg] = tensor.arguments() else { panic!() };
        let cond_data = context.compute(cond)?;

        let &[value] = cond_data.as_slice() else { panic!() };
        context.compute(if value > 0.0 { pos } else { neg })
    }
}
