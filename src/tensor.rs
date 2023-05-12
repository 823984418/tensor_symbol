use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::core::add_tensor::AddTensor;
use crate::core::assign::Assign;
use crate::core::constant::Constant;
use crate::core::div_tensor::DivTensor;
use crate::core::extend_scale::ExtendScale;
use crate::core::function::Function;
use crate::core::matrix_mul::MatrixMul;
use crate::core::merge_tensor::MergeTensor;
use crate::core::mul_tensor::MulTensor;
use crate::core::reshape::Reshape;
use crate::core::select::Select;
use crate::core::slice_tensor::SliceTensor;
use crate::core::sub_tensor::SubTensor;
use crate::core::variable::Variable;
use crate::core::TensorOperator;
use crate::cpu::CpuContext;
use crate::grad::BackwardGrad;

pub fn data_size(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub struct TensorInner {
    level: u32,
    shape: Vec<usize>,
    arguments: Vec<Tensor>,
    operator: Box<dyn TensorOperator>,
}

impl TensorInner {
    pub fn new(
        shape: Vec<usize>,
        arguments: Vec<Tensor>,
        operator: Box<dyn TensorOperator>,
    ) -> Self {
        let level = arguments.iter().map(|x| x.level + 1).max().unwrap_or(0);
        TensorInner {
            level,
            shape,
            arguments,
            operator,
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    pub fn operator(&self) -> &dyn TensorOperator {
        self.operator.deref()
    }

    pub fn same(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }

    pub fn arguments(&self) -> &[Tensor] {
        self.arguments.as_slice()
    }
}

impl Debug for TensorInner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Tensor");
        s.field("level", &self.level);
        s.field("shape", &self.shape);
        s.field("operator", &self.operator);
        s.field("arguments", &self.arguments);
        s.finish()
    }
}

#[derive(Clone)]
pub struct Tensor {
    inner: Arc<TensorInner>,
}

impl Tensor {
    pub fn variable<S: AsRef<[usize]>>(shape: S) -> Self {
        Variable::variable(shape.as_ref().to_vec())
    }

    pub fn scale(value: f32) -> Self {
        Constant::scale(value)
    }

    pub fn constant<S: AsRef<[usize]>>(shape: S, data: Arc<Vec<f32>>) -> Self {
        Constant::constant(shape.as_ref().to_vec(), data)
    }

    pub fn select<P: AsRef<Tensor>, N: AsRef<Tensor>>(&self, pos: P, neg: N) -> Self {
        Select::tensor(self.clone(), pos.as_ref().clone(), neg.as_ref().clone())
    }

    pub fn assign(&self) -> Tensor {
        Assign::assign(self.clone())
    }

    pub fn apply(&self, fun: Function) -> Tensor {
        fun.apply(self.clone())
    }

    pub fn powf(&self, p: f32) -> Tensor {
        self.apply(Function::Pow(p))
    }

    pub fn matrix_mul<O: AsRef<Tensor>>(&self, other: O) -> Tensor {
        MatrixMul::MulNN.apply(self.clone(), other.as_ref().clone())
    }

    pub fn get<I: AsRef<[usize]>>(&self, index: I) -> Tensor {
        SliceTensor::index(self.clone(), index.as_ref().into())
    }

    pub fn reshape<S: AsRef<[usize]>>(&self, shape: S) -> Tensor {
        Reshape::reshape(self.clone(), shape.as_ref().to_vec())
    }

    pub fn merge<I: IntoIterator>(all: I) -> Tensor
    where
        I::Item: AsRef<Tensor>,
    {
        MergeTensor::merge(all.into_iter().map(|x| x.as_ref().clone()).collect())
    }

    pub fn constant_data(&self) -> Option<Arc<Vec<f32>>> {
        self.operator().cast_to::<Constant>().map(|x| x.data())
    }

    pub fn is_variable(&self) -> bool {
        self.operator().cast_to::<Variable>().is_some()
    }

    pub fn zero<S: AsRef<[usize]>>(shape: S) -> Tensor {
        ExtendScale::zero(shape.as_ref().to_vec())
    }

    pub fn one<S: AsRef<[usize]>>(shape: S) -> Tensor {
        ExtendScale::one(shape.as_ref().to_vec())
    }

    pub fn compute(&self) -> Result<Arc<Vec<f32>>, ()> {
        self.compute_with([])
    }
    pub fn compute_with<I: IntoIterator<Item = (Tensor, Arc<Vec<f32>>)>>(
        &self,
        v: I,
    ) -> Result<Arc<Vec<f32>>, ()> {
        let mut context = CpuContext::new();
        for (var, val) in v {
            context.input(&var, val);
        }
        context.compute(self)
    }

    pub fn compute_display(&self) -> Result<(), ()> {
        let r = self.compute()?;
        let r = r.as_slice();
        match self.shape().len() {
            0 => {
                println!("{:10.5?}", r[0]);
            }
            1 => {
                println!("{:10.5?}", r);
            }
            2 => {
                for i in 0..self.shape()[0] {
                    println!(
                        "{:10.5?}",
                        &r[(i * self.shape()[1])..((i + 1) * self.shape()[1])]
                    );
                }
            }
            _ => {
                let mut shape = self.shape().to_vec();
                let x = shape.pop().unwrap();
                let y = shape.pop().unwrap();
                let p = data_size(&shape);
                for s in 0..p {
                    let mut v = vec![0; shape.len()];
                    let mut ss = s;
                    for k in (0..shape.len()).rev() {
                        v[k] = ss % shape[k];
                        ss /= shape[k];
                    }
                    println!("# {:?}:", v);
                    for i in 0..y {
                        let o = s * p + i * x;
                        println!("{:10.5?}", &r[o..(o + x)]);
                    }
                    println!();
                }
            }
        }
        Ok(())
    }

    pub fn back<T: AsRef<Tensor>>(&self, target: T) -> Tensor {
        let mut context = BackwardGrad::new();
        context.append(
            self,
            ExtendScale::extend(Tensor::scale(1.0), self.shape().to_vec()),
        );
        context
            .result()
            .get(target.as_ref().into())
            .unwrap()
            .clone()
    }

    pub fn debug_define(&self) -> impl Debug {
        fn debug_impl<'s, S: FnMut(&HashMap<&TensorHandle, u32>, &Tensor) -> std::fmt::Result>(
            tensor: &'s TensorHandle,
            set: &mut HashMap<&'s TensorHandle, u32>,
            show: &mut S,
        ) -> std::fmt::Result {
            if set.get(&tensor).is_none() {
                for i in tensor.arguments() {
                    debug_impl(i.into(), set, show)?;
                }
                set.insert(tensor.into(), set.len() as u32 + 1);
                show(set, tensor)?;
            }
            Ok(())
        }
        struct DebugDefine(Tensor);
        impl Debug for DebugDefine {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                let mut set: HashMap<&TensorHandle, u32> = HashMap::new();
                let mut d = |set: &HashMap<&TensorHandle, u32>, item: &Tensor| {
                    f.write_str("$")?;
                    let index = *set.get(<&TensorHandle>::from(item)).unwrap();
                    Display::fmt(&index, f)?;
                    Debug::fmt(item.shape(), f)?;
                    f.write_str(" = ")?;
                    item.operator().fmt(f)?;
                    f.write_str("{")?;
                    for (t, i) in item.arguments().iter().enumerate() {
                        if t != 0 {
                            f.write_str(", ")?;
                        }
                        let index = *set.get(<&TensorHandle>::from(i)).unwrap();
                        f.write_str("$")?;
                        Display::fmt(&index, f)?;
                    }
                    f.write_str("}\n")?;
                    Ok(())
                };
                debug_impl((&self.0).into(), &mut set, &mut d)
            }
        }
        DebugDefine(self.clone())
    }

    pub fn new(
        shape: Vec<usize>,
        arguments: Vec<Tensor>,
        operator: Box<dyn TensorOperator>,
    ) -> Self {
        Self::new_inner(TensorInner::new(shape, arguments, operator))
    }

    fn new_inner(inner: TensorInner) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl Deref for Tensor {
    type Target = TensorInner;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        TensorInner::fmt(self, f)
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.operator().display(self, f)
    }
}

impl From<f32> for Tensor {
    fn from(value: f32) -> Self {
        Self::scale(value)
    }
}

impl From<&Tensor> for Tensor {
    fn from(value: &Tensor) -> Self {
        value.clone()
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        match (data_size(self.shape()), data_size(rhs.shape())) {
            (1, 1) => AddTensor::add(vec![self, rhs]),
            (1, _) => AddTensor::add(vec![ExtendScale::extend(self, rhs.shape().to_vec()), rhs]),
            (_, 1) => {
                let shape = self.shape().to_vec();
                AddTensor::add(vec![self, ExtendScale::extend(rhs, shape)])
            }
            (_, _) => AddTensor::add(vec![self, rhs]),
        }
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::add(self.clone(), rhs.clone())
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::add(self.clone(), rhs)
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(self, rhs.clone())
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        Function::Add(rhs).apply(self)
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        self.apply(Function::Add(rhs))
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        match (data_size(self.shape()), data_size(rhs.shape())) {
            (1, 1) => SubTensor::sub(self, rhs),
            (1, _) => SubTensor::sub(ExtendScale::extend(self, rhs.shape().to_vec()), rhs),
            (_, 1) => {
                let shape = self.shape().to_vec();
                SubTensor::sub(self, ExtendScale::extend(rhs, shape))
            }
            (_, _) => SubTensor::sub(self, rhs),
        }
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::sub(self.clone(), rhs)
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::sub(self.clone(), rhs)
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(self, rhs.clone())
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        Function::Add(-rhs).apply(self)
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self.apply(Function::Add(-rhs))
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        match (data_size(self.shape()), data_size(rhs.shape())) {
            (1, 1) => MulTensor::mul(vec![self, rhs]),
            (1, _) => MulTensor::mul(vec![ExtendScale::extend(self, rhs.shape().to_vec()), rhs]),
            (_, 1) => {
                let shape = self.shape().to_vec();
                MulTensor::mul(vec![self, ExtendScale::extend(rhs, shape)])
            }
            (_, _) => MulTensor::mul(vec![self, rhs]),
        }
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::mul(self.clone(), rhs.clone())
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::mul(self.clone(), rhs)
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs.clone())
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        Function::Mul(rhs).apply(self)
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        self.apply(Function::Mul(rhs))
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        match (data_size(self.shape()), data_size(rhs.shape())) {
            (1, 1) => DivTensor::div(self, rhs),
            (1, _) => DivTensor::div(ExtendScale::extend(self, rhs.shape().to_vec()), rhs),
            (_, 1) => {
                let shape = self.shape().to_vec();
                DivTensor::div(self, ExtendScale::extend(rhs, shape))
            }
            (_, _) => DivTensor::div(self, rhs),
        }
    }
}

impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::div(self.clone(), rhs)
    }
}

impl Div<Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::div(self.clone(), rhs)
    }
}

impl Div<&Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Self::Output {
        Tensor::div(self, rhs.clone())
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Self::Output {
        Function::Mul(1.0 / rhs).apply(self)
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Self::Output {
        self.apply(Function::Mul(1.0 / rhs))
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        Function::Neg.apply(self)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self.apply(Function::Neg)
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct TensorHandle(pub Tensor);

impl Debug for TensorHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        <Tensor as Debug>::fmt(self, f)
    }
}

impl Display for TensorHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:p}",
            self.deref() as &TensorInner as *const TensorInner
        )
    }
}

impl<'s> From<&'s Tensor> for &'s TensorHandle {
    fn from(value: &'s Tensor) -> Self {
        unsafe { &*(value as *const _ as *const TensorHandle) }
    }
}

impl From<Tensor> for TensorHandle {
    fn from(value: Tensor) -> Self {
        Self(value)
    }
}

impl Deref for TensorHandle {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Eq for TensorHandle {}

impl PartialEq<Self> for TensorHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl PartialOrd<Self> for TensorHandle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl Ord for TensorHandle {
    fn cmp(&self, other: &Self) -> Ordering {
        let a: &TensorInner = self.deref();
        let b: &TensorInner = other.deref();
        u32::cmp(&a.level, &b.level).then(<*const TensorInner>::cmp(
            &(a as *const _),
            &(b as *const _),
        ))
    }
}

impl Hash for TensorHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let inner: &TensorInner = self.deref();
        std::ptr::hash(inner, state);
    }
}
