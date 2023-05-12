use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::Tensor;

#[derive(Debug, Copy, Clone)]
pub enum MatrixMul {
    MulNN,
    MulNT,
    MulTN,
    MulTT,
}

impl MatrixMul {
    fn flip_a(self) -> Self {
        match self {
            MatrixMul::MulNN => MatrixMul::MulTN,
            MatrixMul::MulNT => MatrixMul::MulTT,
            MatrixMul::MulTN => MatrixMul::MulNN,
            MatrixMul::MulTT => MatrixMul::MulNT,
        }
    }
    fn flip_b(self) -> Self {
        match self {
            MatrixMul::MulNN => MatrixMul::MulNT,
            MatrixMul::MulNT => MatrixMul::MulNN,
            MatrixMul::MulTN => MatrixMul::MulTT,
            MatrixMul::MulTT => MatrixMul::MulTN,
        }
    }
    pub fn apply(self, a: Tensor, b: Tensor) -> Tensor {
        let &[a1, a2] = a.shape() else { panic!() };
        let &[b1, b2] = b.shape() else { panic!() };
        let (o1, o2) = match self {
            MatrixMul::MulNN => {
                assert_eq!(a2, b1);
                (a1, b2)
            }
            MatrixMul::MulNT => {
                assert_eq!(a2, b2);
                (a1, b1)
            }
            MatrixMul::MulTN => {
                assert_eq!(a1, b1);
                (a2, b2)
            }
            MatrixMul::MulTT => {
                assert_eq!(a1, b2);
                (a2, b1)
            }
        };
        Tensor::new(vec![o1, o2], vec![a, b], Box::new(self))
    }
}

impl TensorOperator for MatrixMul {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let [a, b] = tensor.arguments() else { panic!() };
        self.apply(a.clone(), context.compute(b)) + self.apply(context.compute(a), b.clone())
    }
    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let [a, b] = tensor.arguments() else { panic!() };
        context.append(a, self.flip_b().apply(grad.clone(), b.clone()));
        context.append(b, self.flip_a().apply(a.clone(), grad.clone()));
    }
}

#[test]
fn test() {
    use std::sync::Arc;
    let ref a = Tensor::constant([2, 3], Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let ref b = Tensor::constant(
        [3, 4],
        Arc::new(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]),
    );
    let ref grad = Tensor::constant([2, 4], Arc::new(vec![1.0; 8]));
    let ref y = a.matrix_mul(b);
    assert_eq!(
        y.back(a).compute().unwrap().as_slice(),
        MatrixMul::MulNT
            .apply(grad.clone(), b.clone())
            .compute()
            .unwrap()
            .as_slice()
    );
    assert_eq!(
        y.back(b).compute().unwrap().as_slice(),
        MatrixMul::MulTN
            .apply(a.clone(), grad.clone())
            .compute()
            .unwrap()
            .as_slice()
    );
}
