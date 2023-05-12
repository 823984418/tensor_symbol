use crate::core::merge_tensor::MergeTensor;
use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::{data_size, Tensor};

#[derive(Debug, Copy, Clone)]
pub struct SliceTensor {
    from: usize,
}

impl SliceTensor {
    pub fn index(source: Tensor, index: Vec<usize>) -> Tensor {
        let from = data_size(&index) * data_size(&source.shape()[index.len()..]);
        let new_shape = source.shape()[index.len()..].to_vec();
        Self::tensor(source, from, new_shape)
    }
    pub fn slice(source: Tensor, from: usize, len: usize) -> Tensor {
        let from = from * data_size(&source.shape()[1..]);
        let mut new_shape = source.shape().to_vec();
        new_shape[0] = len;
        Self::tensor(source, from, new_shape)
    }

    pub fn tensor(source: Tensor, from: usize, new_shape: Vec<usize>) -> Tensor {
        assert!(from + data_size(&new_shape) <= data_size(source.shape()));
        Tensor::new(new_shape, vec![source], Box::new(Self { from }))
    }

    pub fn from(&self) -> usize {
        self.from
    }
}

impl TensorOperator for SliceTensor {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }
    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let [arg] = tensor.arguments() else { panic!() };
        SliceTensor::tensor(context.compute(arg), self.from, tensor.shape().to_vec())
    }
    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let [arg] = tensor.arguments() else { panic!() };
        let grad_fill = MergeTensor::tensor(self.from, vec![grad.clone()], arg.shape().to_vec());
        context.append(arg, grad_fill);
    }
}

#[test]
fn test() {
    use std::sync::Arc;

    let ref a = Tensor::constant([2, 3], Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let ref b = a.get([1]);
    assert_eq!(
        b.back(a).compute().unwrap().as_slice(),
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    )
}
