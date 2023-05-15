use std::sync::Arc;

use crate::core::variable::Variable;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::Tensor;

impl CpuOperator for Variable {
    fn compute(&self, _tensor: &Tensor, _context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        eprintln!("${} = None", self.variable_id());
        Err(())
    }
}
