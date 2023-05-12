use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Variable(u32);

impl Variable {
    pub fn variable(shape: Vec<usize>) -> Tensor {
        static INDEX: AtomicU32 = AtomicU32::new(0);
        let index = INDEX.fetch_add(1, Ordering::SeqCst) + 1;

        Tensor::new(shape, vec![], Box::new(Variable(index)))
    }

    pub fn variable_id(&self) -> u32 {
        self.0
    }
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        unimplemented!()
    }
}

impl TensorOperator for Variable {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, _context: &mut ForwardGrad) -> Tensor {
        Tensor::zero(tensor.shape())
    }

    fn backward_grad(&self, _tensor: &Tensor, _grad: &Tensor, _context: &mut BackwardGrad) {}

    fn display(&self, _tensor: &Tensor, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "${}", self.0)
    }
}
