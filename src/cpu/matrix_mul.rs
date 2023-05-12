use std::sync::Arc;

use crate::core::matrix_mul::MatrixMul;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::Tensor;

impl CpuOperator for MatrixMul {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let [a, b] = tensor.arguments() else { panic!() };
        let a_input = context.compute(a)?;
        let b_input = context.compute(b)?;

        let a_input = a_input.as_slice();
        let b_input = b_input.as_slice();

        let &[a1, a2] = a.shape() else { panic!() };
        let &[b1, b2] = b.shape() else { panic!() };
        let &[o1, o2] = tensor.shape() else { panic!() };

        let mut output = Vec::with_capacity(o1 * o2);

        assert_eq!(a_input.len(), a1 * a2);
        assert_eq!(b_input.len(), b1 * b2);

        match self {
            MatrixMul::MulNN => {
                let len = a2;
                assert_eq!(a2, b1);
                assert_eq!(a1, o1);
                assert_eq!(o2, b2);

                for i in 0..o1 {
                    for j in 0..o2 {
                        let mut sum = 0.0;
                        for k in 0..len {
                            sum += a_input[i * a2 + k] * b_input[k * b2 + j];
                        }
                        output.push(sum);
                    }
                }
            }
            MatrixMul::MulNT => {
                let len = a2;
                assert_eq!(a2, b2);
                assert_eq!(a1, o1);
                assert_eq!(o2, b1);

                for i in 0..o1 {
                    for j in 0..o2 {
                        let mut sum = 0.0;
                        for k in 0..len {
                            sum += a_input[i * a2 + k] * b_input[k + j * b2];
                        }
                        output.push(sum);
                    }
                }
            }
            MatrixMul::MulTN => {
                let len = a1;
                assert_eq!(a1, b1);
                assert_eq!(a2, o1);
                assert_eq!(o2, b2);

                for i in 0..o1 {
                    for j in 0..o2 {
                        let mut sum = 0.0;
                        for k in 0..len {
                            sum += a_input[i + k * a2] * b_input[k * b2 + j];
                        }
                        output.push(sum);
                    }
                }
            }
            MatrixMul::MulTT => {
                let len = a1;
                assert_eq!(a1, b2);
                assert_eq!(a2, o1);
                assert_eq!(o2, b1);

                for i in 0..o1 {
                    for j in 0..o2 {
                        let mut sum = 0.0;
                        for k in 0..len {
                            sum += a_input[i + k * a2] * b_input[k + j * b2];
                        }
                        output.push(sum);
                    }
                }
            }
        }
        Ok(Arc::new(output))
    }
}

#[test]
fn test() {
    let a = Tensor::constant([2, 3], Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let b = Tensor::constant(
        [3, 4],
        Arc::new(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]),
    );
    assert_eq!(
        a.matrix_mul(b).compute().unwrap().as_slice(),
        [38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
    );
}
