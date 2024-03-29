use std::fmt::{Display, Formatter};

use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct SubTensor;

impl SubTensor {
    pub fn sub(a: Tensor, b: Tensor) -> Tensor {
        assert_eq!(a.shape(), b.shape());
        Tensor::new(a.shape().to_vec(), vec![a, b], Box::new(SubTensor))
    }
}

impl TensorOperator for SubTensor {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let [a, b] = tensor.arguments() else { panic!() };
        context.compute(a) - context.compute(b)
    }

    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let [a, b] = tensor.arguments() else { panic!() };
        context.append(a, grad.clone());
        context.append(b, -grad);
    }

    fn display(&self, tensor: &Tensor, f: &mut Formatter<'_>) -> std::fmt::Result {
        let [a, b] = tensor.arguments() else { panic!() };
        f.write_str("(")?;
        Display::fmt(a, f)?;
        f.write_str(" - ")?;
        Display::fmt(b, f)?;
        f.write_str(")")?;
        Ok(())
    }
}

#[test]
fn test() {
    let ref a = Tensor::scale(1.0);
    let ref b = Tensor::scale(2.0);
    let ref y = a - b;
    assert_eq!(y.back(a).compute().unwrap().as_slice(), [1.0]);
    assert_eq!(y.back(b).compute().unwrap().as_slice(), [-1.0]);
}
