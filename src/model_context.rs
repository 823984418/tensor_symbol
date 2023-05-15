use std::sync::Arc;

use rand::distributions::Distribution;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;

use crate::cpu::CpuContext;
use crate::grad::BackwardGrad;
use crate::tensor::{data_size, Tensor};

#[derive(Debug)]
pub struct ModelContext {
    index: usize,
    variables: Vec<(Tensor, Vec<f32>)>,
}

impl ModelContext {
    pub fn new() -> Self {
        Self::new_with(Vec::new())
    }

    pub fn new_with(variables: Vec<(Tensor, Vec<f32>)>) -> Self {
        Self {
            index: 0,
            variables,
        }
    }

    pub fn variable<S: AsRef<[usize]>>(&mut self, shape: S) -> Tensor {
        self.variable_rng(shape, Normal::new(0.0, 0.01).unwrap())
    }

    pub fn variable_rng<S: AsRef<[usize]>, D: Distribution<f32>>(
        &mut self,
        shape: S,
        rng: D,
    ) -> Tensor {
        if self.index < self.variables.len() {
            let var = self.variables[self.index].0.clone();
            assert_eq!(var.shape(), shape.as_ref());
            self.index += 1;
            var
        } else {
            let len = data_size(shape.as_ref());
            let var = Tensor::variable(shape);
            let value = SmallRng::from_entropy()
                .sample_iter(rng)
                .take(len)
                .collect::<Vec<_>>();
            assert_eq!(value.len(), len);
            self.variables.push((var.clone(), value));
            self.index += 1;
            var
        }
    }

    pub fn set_value(&mut self, var: &Tensor, v: Vec<f32>) -> bool {
        assert_eq!(data_size(var.shape()), v.len());
        for (variable, value) in &mut self.variables {
            if variable.same(var) {
                *value = v;
                return true;
            }
        }
        false
    }

    pub fn reset(&mut self) {
        self.index = 0;
    }

    pub fn load_to(&self, context: &mut CpuContext) {
        for (var, val) in &self.variables {
            context.input(var, Arc::new(val.to_vec()));
        }
    }

    pub fn optimization(
        &mut self,
        context: &mut CpuContext,
        target: &Tensor,
        rate: f32,
    ) -> Result<(), ()> {
        let mut back = BackwardGrad::new();
        back.append(target, Tensor::scale(1.0));
        let back = back.result();
        self.load_to(context);
        for (var, val) in &mut self.variables {
            let len = val.len();
            if let Some(b) = back.get(var.as_ref().into()) {
                let g = context.compute(b)?;
                let g = g.as_slice();
                for i in 0..len {
                    val[i] -= g[i] * rate;
                }
            }
        }
        Ok(())
    }
}
