use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::tensor::{Tensor, TensorHandle};

#[derive(Debug)]
pub struct VariableInlineContext {
    catch: HashMap<TensorHandle, Option<Tensor>>,
}

impl VariableInlineContext {
    pub fn new() -> Self {
        Self {
            catch: HashMap::new(),
        }
    }

    pub fn variable(&mut self, v: &Tensor, s: &Tensor) {
        assert!(v.is_variable());
        let v: TensorHandle = v.clone().into();
        if let Entry::Vacant(x) = self.catch.entry(v.clone()) {
            x.insert(Some(s.clone()));
        } else {
            panic!("the {} has define", v);
        }
    }

    pub fn get(&mut self, t: &Tensor) -> Tensor {
        let n: TensorHandle = t.clone().into();

        if let Some(x) = self.catch.get(&n) {
            match x {
                None => t.clone(),
                Some(x) => x.clone(),
            }
        } else {
            let mut arg = t.arguments().to_vec();
            let mut update = false;
            for i in arg.iter_mut() {
                let new = self.get(i);
                if !i.same(&new) {
                    update = true;
                    *i = new;
                }
            }
            if update {
                let r = Tensor::new(n.shape().to_vec(), arg, t.operator().clone_box());
                self.catch.insert(n, Some(r.clone()));
                r
            } else {
                self.catch.insert(n, None);
                t.clone()
            }
        }
    }
}
