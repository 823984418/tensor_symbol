use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Constant {
    data: Arc<Vec<f32>>,
}

impl Constant {
    pub fn constant(shape: Vec<usize>, data: Arc<Vec<f32>>) -> Tensor {
        Tensor::new(shape, vec![], Box::new(Self { data }))
    }

    pub fn scale(value: f32) -> Tensor {
        let data = Arc::new(vec![value]);
        Tensor::new(vec![], vec![], Box::new(Self { data }))
    }

    pub fn data(&self) -> Arc<Vec<f32>> {
        self.data.clone()
    }
}

impl TensorOperator for Constant {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, _context: &mut ForwardGrad) -> Tensor {
        Tensor::zero(tensor.shape().to_vec())
    }
    fn backward_grad(&self, _tensor: &Tensor, _grad: &Tensor, _context: &mut BackwardGrad) {}
    fn display(&self, tensor: &Tensor, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Constant")?;
        tensor.shape().fmt(f)?;
        f.write_str("{")?;
        if self.data.len() > 0 {
            Debug::fmt(&self.data[0], f)?;
            if self.data.len() > 1 {
                f.write_str(", ... ")?;
            }
        }
        f.write_str("}")?;
        Ok(())
    }
}
