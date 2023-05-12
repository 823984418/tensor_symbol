use std::sync::Arc;

use crate::core::constant::Constant;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::Tensor;

impl CpuOperator for Constant {
    fn compute(&self, _tensor: &Tensor, _context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        Ok(self.data().clone())
    }
}
