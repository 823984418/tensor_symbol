use std::fmt::{Display, Formatter};

use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::Tensor;

#[derive(Debug, Copy, Clone)]
pub struct AddTensor;

impl AddTensor {
    pub fn tensor(shape: &[usize], arguments: Vec<Tensor>) -> Tensor {
        if arguments.len() == 0 {
            return Tensor::zero(shape);
        }
        for arg in &arguments {
            assert_eq!(shape, arg.shape());
        }

        // if arguments.len() == 1 {
        //     return arguments[0].assign();
        // }

        AddTensor::add(arguments)
    }

    pub fn add(arguments: Vec<Tensor>) -> Tensor {
        let shape = arguments[0].shape();
        for arg in arguments.iter() {
            assert_eq!(shape, arg.shape());
        }
        Tensor::new(shape.to_vec(), arguments, Box::new(AddTensor))
    }
}

impl TensorOperator for AddTensor {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        AddTensor::tensor(
            tensor.shape(),
            tensor
                .arguments()
                .iter()
                .map(|x| context.compute(x))
                .collect(),
        )
    }

    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        for i in tensor.arguments() {
            context.append(i.clone(), grad.clone());
        }
    }

    fn display(&self, tensor: &Tensor, f: &mut Formatter<'_>) -> std::fmt::Result {
        if tensor.arguments().is_empty() {
            return f.write_str("0");
        }
        f.write_str("(")?;
        for (i, arg) in tensor.arguments().iter().enumerate() {
            if i != 0 {
                f.write_str(" + ")?;
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
    let ref b = a + a;
    assert_eq!(b.back(a).compute().unwrap().as_slice(), [2.0]);
}
