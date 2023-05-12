use std::sync::Arc;

use crate::core::function::Function;
use crate::cpu::{CpuContext, CpuOperator};
use crate::tensor::{data_size, Tensor};

impl CpuOperator for Function {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()> {
        let [arg] = tensor.arguments() else { panic!() };
        let input_data = context.compute(arg)?;
        let input = input_data.as_slice();
        let len = data_size(tensor.shape());
        assert_eq!(input.len(), len);
        let mut data = Vec::with_capacity(len);
        match *self {
            Function::Sin => {
                for i in 0..len {
                    data.push(f32::sin(input[i]));
                }
            }
            Function::Cos => {
                for i in 0..len {
                    data.push(f32::cos(input[i]));
                }
            }
            Function::ReLU => {
                for i in 0..len {
                    data.push(f32::max(input[i], 0.0));
                }
            }
            Function::Step => {
                for i in 0..len {
                    data.push(if input[i].is_sign_positive() {
                        1.0
                    } else {
                        0.0
                    });
                }
            }
            Function::Abs => {
                for i in 0..len {
                    data.push(f32::abs(input[i]));
                }
            }
            Function::Sig => {
                for i in 0..len {
                    data.push(f32::signum(input[i]));
                }
            }
            Function::Neg => {
                for i in 0..len {
                    data.push(-input[i]);
                }
            }
            Function::Mul(x) => {
                if x == -2.0 {
                    for i in 0..len {
                        data.push(input[i] * -2.0);
                    }
                } else if x == -1.0 {
                    for i in 0..len {
                        data.push(-input[i]);
                    }
                } else if x == 0.0 {
                    for _ in 0..len {
                        data.push(0.0);
                    }
                } else if x == 1.0 {
                    return Ok(input_data.clone());
                } else if x == 2.0 {
                    for i in 0..len {
                        data.push(input[i] * 2.0);
                    }
                } else {
                    for i in 0..len {
                        data.push(input[i] * x);
                    }
                }
            }
            Function::Add(x) => {
                if x == 0.0 {
                    return Ok(input_data.clone());
                } else {
                    for i in 0..len {
                        data.push(input[i] + x);
                    }
                }
            }
            Function::Pow(x) => {
                if x == -1.0 {
                    for i in 0..len {
                        data.push(1.0 / input[i]);
                    }
                } else if x == 0.0 {
                    for _ in 0..len {
                        data.push(0.0);
                    }
                } else if x == 1.0 {
                    return Ok(input_data.clone());
                } else if x == 2.0 {
                    for i in 0..len {
                        data.push(input[i] * input[i]);
                    }
                } else {
                    for i in 0..len {
                        data.push(input[i].powf(x));
                    }
                }
            }
        }
        assert_eq!(data.len(), len);
        Ok(Arc::new(data))
    }
}
