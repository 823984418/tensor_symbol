use std::sync::Arc;

use crate::core::div_tensor::DivTensor;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::{data_size, Tensor};

impl CpuOperator for DivTensor {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let [a, b] = tensor.arguments() else { panic!() };
        let a_input = context.compute(a)?;
        let b_input = context.compute(b)?;
        let a_input = a_input.as_slice();
        let b_input = b_input.as_slice();

        let len = data_size(tensor.shape());
        assert_eq!(a_input.len(), len);
        assert_eq!(b_input.len(), len);

        let mut output = Vec::with_capacity(len);
        for i in 0..len {
            output.push(a_input[i] / b_input[i]);
        }
        Ok(Arc::new(output))
    }
}

#[test]
fn test() {
    let a = Tensor::scale(1.0);
    let b = Tensor::scale(2.0);
    assert_eq!((a / b).compute().unwrap().as_slice(), [0.5]);
}
