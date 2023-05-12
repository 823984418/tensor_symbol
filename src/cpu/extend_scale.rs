use std::sync::Arc;

use crate::core::extend_scale::ExtendScale;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::{data_size, Tensor};

impl CpuOperator for ExtendScale {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let [scale] = tensor.arguments() else { panic!() };
        let &[value] = context.compute(scale)?.as_slice() else { panic!() };
        Ok(Arc::new(vec![value; data_size(tensor.shape())]))
    }
}

#[test]
fn test() {
    let a = Tensor::scale(1.0);
    assert_eq!(
        ExtendScale::extend(a, vec![2, 3])
            .compute()
            .unwrap()
            .as_slice(),
        [1.0; 6]
    );
}
