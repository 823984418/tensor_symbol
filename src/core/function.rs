use crate::core::TensorOperator;
use crate::grad::{BackwardGrad, ForwardGrad};
use crate::tensor::Tensor;

#[derive(Debug, Copy, Clone)]
pub enum Function {
    Sin,
    Cos,
    /// max(x, 0)
    ReLU,
    /// 阶跃函数
    Step,
    /// 绝对值函数
    Abs,
    /// 符号函数
    Sig,
    /// 取负函数
    Neg,
    /// 乘以标量
    Mul(f32),
    /// 加上标量
    Add(f32),
    /// 幂函数
    Pow(f32),
    Sigmoid,
}

impl Function {
    pub fn apply(self, tensor: Tensor) -> Tensor {
        Tensor::new(tensor.shape().to_vec(), vec![tensor], Box::new(self))
    }
}

impl TensorOperator for Function {
    fn clone_box(&self) -> Box<dyn TensorOperator> {
        Box::new(self.clone())
    }

    fn forward_grad(&self, tensor: &Tensor, context: &mut ForwardGrad) -> Tensor {
        let [arg] = tensor.arguments() else { panic!() };
        let grad = context.compute(arg);
        match *self {
            Function::Sin => grad * arg.apply(Function::Cos),
            Function::Cos => grad * -arg.apply(Function::Sin),
            Function::ReLU => grad * arg.apply(Function::Step),
            Function::Step => Tensor::zero(tensor.shape()),
            Function::Abs => grad * arg.apply(Function::Sig),
            Function::Sig => Tensor::zero(tensor.shape()),
            Function::Neg => -grad,
            Function::Mul(x) => grad * x,
            Function::Add(_) => grad.clone(),
            Function::Pow(x) => {
                if x == 0.0 {
                    Tensor::zero(tensor.shape())
                } else if x == 1.0 {
                    grad.assign()
                } else if x == 2.0 {
                    grad * arg * 2.0
                } else {
                    grad * arg.powf(x - 1.0) * x
                }
            }
            Function::Sigmoid => grad * (tensor * (-tensor + 1.0)),
        }
    }

    fn backward_grad(&self, tensor: &Tensor, grad: &Tensor, context: &mut BackwardGrad) {
        let [arg] = tensor.arguments() else { panic!() };
        let back = match *self {
            Function::Sin => grad * arg.apply(Function::Cos),
            Function::Cos => grad * -arg.apply(Function::Sin),
            Function::ReLU => grad * arg.apply(Function::Step),
            Function::Step => Tensor::zero(tensor.shape()),
            Function::Abs => grad * arg.apply(Function::Sig),
            Function::Sig => Tensor::zero(tensor.shape()),
            Function::Neg => -grad,
            Function::Mul(x) => grad * x,
            Function::Add(_) => grad.clone(),
            Function::Pow(x) => {
                if x == 0.0 {
                    Tensor::zero(tensor.shape())
                } else if x == 1.0 {
                    grad.assign()
                } else if x == 2.0 {
                    grad * arg * 2.0
                } else {
                    grad * arg.powf(x - 1.0) * x
                }
            }
            Function::Sigmoid => grad * (tensor * (-tensor + 1.0)),
        };
        context.append(arg, back);
    }
}
