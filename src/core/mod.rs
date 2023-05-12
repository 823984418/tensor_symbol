use std::any::{Any, TypeId};
use std::fmt::{Debug, Display, Formatter};

use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::Tensor;

pub mod add_tensor;
pub mod assign;
pub mod constant;
pub mod div_tensor;
pub mod extend_scale;
pub mod function;
pub mod matrix_mul;
pub mod merge_tensor;
pub mod mul_tensor;
pub mod reshape;
pub mod select;
pub mod slice_tensor;
pub mod sub_tensor;
pub mod sum_scale;
pub mod variable;

pub trait TensorOperator: Any + Debug + Send + Sync {
    fn clone_box(&self) -> Box<dyn TensorOperator>;
    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let _ = tensor;
        let _ = context;
        unimplemented!("{:?}", self)
    }

    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let _ = tensor;
        let _ = grad;
        let _ = context;
        unimplemented!("{:?}", self)
    }

    fn display(&self, tensor: &Tensor, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)?;
        f.write_str("(")?;

        for (i, arg) in tensor.arguments().iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }
            Display::fmt(arg, f)?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

impl dyn TensorOperator {
    pub fn cast_to<O: TensorOperator>(&self) -> Option<&O> {
        if Self::type_id(self) == TypeId::of::<O>() {
            Some(unsafe { &*(self as *const _ as *const O) })
        } else {
            None
        }
    }
}
