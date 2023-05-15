use std::any::TypeId;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::core::add_tensor::AddTensor;
use crate::core::assign::Assign;
use crate::core::constant::Constant;
use crate::core::debug_assign::DebugAssign;
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
use crate::core::sum_scale::SumScale;
use crate::core::variable::Variable;
use crate::core::TensorOperator;
use crate::tensor::{data_size, Tensor, TensorHandle};

pub mod add_tensor;
pub mod assign;
pub mod constant;
pub mod debug_assign;
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

pub trait CpuOperator: TensorOperator {
    fn compute(&self, tensor: &Tensor, context: &mut CpuContext) -> Result<Arc<Vec<f32>>, ()>;
}

pub fn operator_as_cpu(r: &dyn TensorOperator) -> Option<&dyn CpuOperator> {
    static TYPE_MAP: OnceLock<HashMap<TypeId, fn(&dyn TensorOperator) -> &dyn CpuOperator>> =
        OnceLock::new();

    fn init_map() -> HashMap<TypeId, fn(&dyn TensorOperator) -> &dyn CpuOperator> {
        let mut map = HashMap::new();
        let m = &mut map;

        fn insert<T: CpuOperator>(
            map: &mut HashMap<TypeId, fn(&dyn TensorOperator) -> &dyn CpuOperator>,
        ) {
            fn cast<T: CpuOperator>(r: &dyn TensorOperator) -> &dyn CpuOperator {
                r.cast_to::<T>().unwrap()
            }

            map.insert(TypeId::of::<T>(), cast::<T> as _);
        }

        insert::<AddTensor>(m);
        insert::<Assign>(m);
        insert::<Constant>(m);
        insert::<DebugAssign>(m);
        insert::<DivTensor>(m);
        insert::<ExtendScale>(m);
        insert::<Function>(m);
        insert::<MatrixMul>(m);
        insert::<MergeTensor>(m);
        insert::<MulTensor>(m);
        insert::<Reshape>(m);
        insert::<Select>(m);
        insert::<SliceTensor>(m);
        insert::<SubTensor>(m);
        insert::<SumScale>(m);
        insert::<Variable>(m);

        map
    }

    TYPE_MAP
        .get_or_init(init_map)
        .get(&r.type_id())
        .map(|f| f(r))
}

#[derive(Debug, Clone)]
pub struct CpuContext {
    catch: HashMap<TensorHandle, Result<Arc<Vec<f32>>, ()>>,
}

impl CpuContext {
    pub fn new() -> Self {
        Self {
            catch: Default::default(),
        }
    }

    pub fn input(&mut self, tensor: &Tensor, data: Arc<Vec<f32>>) {
        assert!(tensor.is_variable());
        assert_eq!(data.len(), data_size(tensor.shape()));
        match self.catch.entry(tensor.clone().into()) {
            Entry::Vacant(entry) => {
                entry.insert(Ok(data));
            }
            Entry::Occupied(_) => {
                panic!()
            }
        }
    }

    pub fn input_constant_with<'s, I: IntoIterator<Item = (&'s Tensor, Arc<Vec<f32>>)>>(
        &mut self,
        tensor: &Tensor,
        value: &Tensor,
        i: I,
    ) {
        self.input(tensor, value.compute_with(i).unwrap());
    }

    pub fn get(&self, tensor: &Tensor) -> Option<Result<Arc<Vec<f32>>, ()>> {
        self.catch.get(tensor.into()).cloned()
    }

    pub fn compute(&mut self, tensor: &Tensor) -> Result<Arc<Vec<f32>>, ()> {
        let tensor: &TensorHandle = tensor.into();
        match self.catch.get(tensor) {
            Some(x) => x.clone(),
            None => {
                let result = operator_as_cpu(tensor.operator())
                    .expect("the operator no support cpu")
                    .compute(tensor, self);
                self.catch.insert(tensor.clone(), result.clone());
                result
            }
        }
    }

    pub fn compute_as_constant(&mut self, tensor: &Tensor) -> Result<Tensor, ()> {
        let data = self.compute(tensor)?;
        Ok(Tensor::constant(tensor.shape(), data))
    }
}
