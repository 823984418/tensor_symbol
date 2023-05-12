use std::sync::Arc;

use crate::core::sum_scale::SumScale;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::Tensor;

impl CpuOperator for SumScale {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let [scale] = tensor.arguments() else { panic!() };
        let value = context.compute(scale)?;
        Ok(Arc::new(vec![value.iter().sum()]))
    }
}
