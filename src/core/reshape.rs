use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::{data_size, Tensor};

#[derive(Debug, Copy, Clone)]
pub struct Reshape;

impl Reshape {
    pub fn reshape(source: Tensor, shape: Vec<usize>) -> Tensor {
        assert_eq!(data_size(source.shape()), data_size(&shape));
        Tensor::new(shape, vec![source], Box::new(Reshape))
    }
}

impl TensorOperator for Reshape {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let [arg] = tensor.arguments() else { panic!() };
        Reshape::reshape(context.compute(arg), tensor.shape().to_vec())
    }

    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let [arg] = tensor.arguments() else { panic!() };
        context.append(arg, Reshape::reshape(grad.clone(), arg.shape().to_vec()));
    }
}
