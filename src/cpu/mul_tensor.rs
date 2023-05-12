use std::sync::Arc;

use crate::core::mul_tensor::MulTensor;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::{data_size, Tensor};

impl CpuOperator for MulTensor {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let len = data_size(tensor.shape());

        if tensor.arguments().len() == 0 {
            return Ok(Arc::new(vec![1.0; len]));
        }
        if tensor.arguments().len() == 1 {
            return context.compute(&tensor.arguments()[0]);
        }

        let mut data = Vec::with_capacity(len);
        let a = context.compute(&tensor.arguments()[0])?;
        let b = context.compute(&tensor.arguments()[1])?;
        assert_eq!(a.len(), len);
        assert_eq!(b.len(), len);
        for i in 0..len {
            data.push(a[i] * b[i]);
        }

        for item in tensor.arguments().iter().skip(2) {
            let arg = context.compute(item)?;
            assert_eq!(data.len(), arg.len());
            for i in 0..data.len() {
                data[i] *= arg[i];
            }
        }
        Ok(Arc::new(data))
    }
}

#[test]
fn test() {
    let a = Tensor::scale(1.0);
    let b = Tensor::scale(2.0);
    assert_eq!((a * b).compute().unwrap().as_slice(), [2.0]);
}
