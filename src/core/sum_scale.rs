use crate::core::extend_scale::ExtendScale;
use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::{data_size, Tensor};

#[derive(Debug, Copy, Clone)]
pub struct SumScale;

impl SumScale {
    pub fn sum(tensor: Tensor) -> Tensor {
        Tensor::new(vec![], vec![tensor], Box::new(SumScale))
    }

    pub fn sum_to_shape(tensor: Tensor, shape: Vec<usize>) -> Tensor {
        assert_eq!(data_size(&shape), 1);
        Tensor::new(shape, vec![tensor], Box::new(SumScale))
    }
}

impl TensorOperator for SumScale {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let [arg] = tensor.arguments() else { panic!() };
        SumScale::sum(context.compute(arg))
    }

    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let [arg] = tensor.arguments() else { panic!() };
        let shape = arg.shape();
        context.append(arg, ExtendScale::extend(grad.clone(), shape.to_vec()));
    }
}

#[test]
fn test() {
    use std::sync::Arc;

    let ref a = Tensor::constant([2, 3], Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let ref b = SumScale::sum(a.clone());
    assert_eq!(
        b.back(a).compute().unwrap().as_slice(),
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
}
