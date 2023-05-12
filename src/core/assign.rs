use std::fmt::{Display, Formatter};

use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::Tensor;

#[derive(Debug, Copy, Clone)]
pub struct Assign;

impl Assign {
    pub fn assign(tensor: Tensor) -> Tensor {
        Tensor::new(tensor.shape().to_vec(), vec![tensor], Box::new(Assign))
    }
}

impl TensorOperator for Assign {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let [arg] = tensor.arguments() else { panic!() };
        context.compute(arg).assign()
    }

    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let [arg] = tensor.arguments() else { panic!() };
        context.append(arg, grad);
    }

    fn display(&self, tensor: &Tensor, f: &mut Formatter<'_>) -> std::fmt::Result {
        let [warp] = tensor.arguments() else { panic!() };
        Display::fmt(warp, f)
    }
}
