use std::fmt::{Display, Formatter};

use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::{data_size, Tensor};

#[derive(Debug, Clone)]
pub struct Select;

impl Select {
    pub fn tensor(cond: Tensor, pos: Tensor, neg: Tensor) -> Tensor {
        assert_eq!(data_size(cond.shape()), 1);
        assert_eq!(pos.shape(), neg.shape());
        Tensor::new(pos.shape().to_vec(), vec![cond, pos, neg], Box::new(Select))
    }
}

impl TensorOperator for Select {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let [cond, pos, neg] = tensor.arguments() else { panic!() };
        Select::tensor(cond.clone(), context.compute(pos), context.compute(neg))
    }

    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let [cond, pos, neg] = tensor.arguments() else { panic!() };

        let zero = Tensor::zero(tensor.shape());

        context.append(
            pos,
            Select::tensor(cond.clone(), grad.clone(), zero.clone()),
        );
        context.append(neg, Select::tensor(cond.clone(), zero, grad.clone()));
    }

    fn display(&self, tensor: &Tensor, f: &mut Formatter<'_>) -> std::fmt::Result {
        let [cond, pos, neg] = tensor.arguments() else { panic!() };

        f.write_str("if ")?;
        Display::fmt(cond, f)?;
        f.write_str(" { ")?;
        Display::fmt(pos, f)?;
        f.write_str(" } else { ")?;
        Display::fmt(neg, f)?;
        f.write_str(" }")?;
        Ok(())
    }
}
