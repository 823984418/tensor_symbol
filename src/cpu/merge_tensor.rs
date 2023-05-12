use std::ops::Deref;
use std::sync::Arc;

use crate::core::merge_tensor::MergeTensor;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::{data_size, Tensor};

impl CpuOperator for MergeTensor {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let all_size = data_size(tensor.shape());

        let mut output = Vec::with_capacity(all_size);

        output.append(&mut vec![0.0; self.fill()]);
        for i in tensor.arguments() {
            let mut value = context.compute(i)?.deref().clone();
            output.append(&mut value);
        }
        output.append(&mut vec![0.0; all_size - output.len()]);

        Ok(Arc::new(output))
    }
}

#[test]
fn test() {
    let a = Tensor::constant([2, 3], Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let b = Tensor::constant(
        [3, 3],
        Arc::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
    );
    assert_eq!(
        Tensor::merge([a, b]).compute().unwrap().as_slice(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    );
}
