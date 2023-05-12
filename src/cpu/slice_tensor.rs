use std::sync::Arc;

use crate::core::slice_tensor::SliceTensor;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::{data_size, Tensor};

impl CpuOperator for SliceTensor {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let [arg] = tensor.arguments() else { panic!() };
        let input = context.compute(arg)?;

        let slice_size = data_size(tensor.shape());
        let skip = self.from();
        Ok(Arc::new(input[skip..(skip + slice_size)].to_vec()))
    }
}

#[test]
fn test() {
    let a = Tensor::constant([2, 3], Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert_eq!(a.get([0]).compute().unwrap().as_slice(), [1.0, 2.0, 3.0]);
    assert_eq!(a.get([1]).compute().unwrap().as_slice(), [4.0, 5.0, 6.0]);
}
