use crate::core::slice_tensor::SliceTensor;
use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::{data_size, Tensor};

#[derive(Debug, Copy, Clone)]
pub struct MergeTensor {
    fill: usize,
}

impl MergeTensor {
    pub fn merge(all: Vec<Tensor>) -> Tensor {
        let mut shape = all[0].shape().to_vec();
        for i in 1..all.len() {
            let s = all[i].shape();
            assert_eq!(shape[1..], s[1..]);
            shape[0] += s[0];
        }
        Self::tensor(0, all, shape)
    }

    pub fn tensor(fill: usize, all: Vec<Tensor>, shape: Vec<usize>) -> Tensor {
        assert!(
            fill + all.iter().map(|t| data_size(t.shape())).sum::<usize>() <= data_size(&shape)
        );
        Tensor::new(shape, all, Box::new(Self { fill }))
    }

    pub fn fill(&self) -> usize {
        self.fill
    }
}

impl TensorOperator for MergeTensor {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }
    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let grads = tensor
            .arguments()
            .iter()
            .map(|x| context.compute(x))
            .collect();
        MergeTensor::tensor(self.fill, grads, tensor.shape().to_vec())
    }
    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let mut from = self.fill;
        for i in tensor.arguments() {
            let shape = i.shape();
            context.append(i, SliceTensor::tensor(grad.clone(), from, shape.to_vec()));
            from += data_size(shape);
        }
    }
}
