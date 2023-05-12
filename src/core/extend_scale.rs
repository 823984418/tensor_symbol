use std::sync::OnceLock;

use crate::core::constant::Constant;
use crate::core::sum_scale::SumScale;
use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::{data_size, Tensor};

#[derive(Debug, Clone)]
pub struct ExtendScale;

impl ExtendScale {
    pub fn zero(shape: Vec<usize>) -> Tensor {
        static ZERO: OnceLock<Tensor> = OnceLock::new();
        Self::extend(ZERO.get_or_init(|| Constant::scale(0.0)).clone(), shape)
    }

    pub fn one(shape: Vec<usize>) -> Tensor {
        static ONE: OnceLock<Tensor> = OnceLock::new();
        Self::extend(ONE.get_or_init(|| Constant::scale(1.0)).clone(), shape)
    }

    pub fn extend(tensor: Tensor, shape: Vec<usize>) -> Tensor {
        assert_eq!(data_size(tensor.shape()), 1);
        Tensor::new(shape, vec![tensor], Box::new(ExtendScale))
    }
}

impl TensorOperator for ExtendScale {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let [arg] = tensor.arguments() else { panic!() };
        ExtendScale::extend(context.compute(arg), tensor.shape().to_vec())
    }

    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let [arg] = tensor.arguments() else { panic!() };
        context.append(arg, SumScale::sum(grad.clone()));
    }
}
