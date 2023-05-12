use std::fmt::{Display, Formatter};
use std::mem::replace;

use crate::core::add_tensor::AddTensor;
use crate::core::extend_scale::ExtendScale;
use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct MulTensor;

impl MulTensor {
    pub fn tensor(shape: &[usize], arguments: Vec<Tensor>) -> Tensor {
        if arguments.len() == 0 {
            return ExtendScale::one(shape.to_vec());
        }
        for arg in &arguments {
            assert_eq!(shape, arg.shape());
        }

        // if arguments.len() == 1 {
        //     return arguments[0].assign();
        // }

        MulTensor::mul(arguments)
    }

    pub fn mul(arguments: Vec<Tensor>) -> Tensor {
        let shape = arguments[0].shape();
        for arg in arguments.iter() {
            assert_eq!(shape, arg.shape());
        }
        Tensor::new(shape.to_vec(), arguments, Box::new(MulTensor))
    }
}

impl TensorOperator for MulTensor {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let mut g = tensor.arguments().to_vec();
        MulTensor::tensor(
            tensor.shape(),
            tensor
                .arguments()
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    let back = replace(&mut g[i], context.compute(x));
                    let p = AddTensor::tensor(tensor.shape(), g.clone());
                    g[i] = back;
                    p
                })
                .collect(),
        )
    }

    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let mut g = tensor.arguments().to_vec();
        let mut grad = grad.clone();
        for i in 0..g.len() {
            let back = replace(&mut g[i], grad);
            context.append(tensor.arguments()[i].clone(), MulTensor::mul(g.clone()));
            grad = replace(&mut g[i], back);
        }
    }

    fn display(&self, tensor: &Tensor, f: &mut Formatter<'_>) -> std::fmt::Result {
        if tensor.arguments().is_empty() {
            return f.write_str("0");
        }
        f.write_str("(")?;
        for (i, arg) in tensor.arguments().iter().enumerate() {
            if i != 0 {
                f.write_str(" * ")?;
            }
            Display::fmt(arg, f)?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

#[test]
fn test() {
    let ref a = Tensor::scale(1.0);
    let ref b = Tensor::scale(2.0);
    let ref y = a * b;
    assert_eq!(y.back(a).compute().unwrap().as_slice(), [2.0]);
    assert_eq!(y.back(b).compute().unwrap().as_slice(), [1.0]);
}
