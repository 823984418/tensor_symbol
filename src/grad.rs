use std::collections::{BTreeMap, HashMap};

use crate::core::add_tensor::AddTensor;
use crate::tensor::{Tensor, TensorHandle};

pub struct BackwardGrad {
    table: BTreeMap<TensorHandle, Vec<Tensor>>,
}

impl BackwardGrad {
    pub fn new() -> Self {
        Self {
            table: BTreeMap::new(),
        }
    }

    pub fn append<S: Into<Tensor>, G: Into<Tensor>>(&mut self, source: S, grad: G) {
        let source = source.into();
        let grad = grad.into();
        assert_eq!(grad.shape(), source.shape());
        self.table.entry(source.into()).or_default().push(grad);
    }

    pub fn result(self) -> HashMap<TensorHandle, Tensor> {
        let mut context = self;
        let mut output = HashMap::new();
        while let Some((tensor, vec)) = context.table.pop_last() {
            let grad = AddTensor::tensor(tensor.shape(), vec);
            tensor
                .operator()
                .backward_grad(&tensor, &grad, &mut context);
            output.insert(tensor, grad.assign());
        }
        output
    }
}

pub struct ForwardGrad {
    table: HashMap<TensorHandle, Option<Tensor>>,
}

impl ForwardGrad {
    pub fn new(input: &[(Tensor, Tensor)]) -> Self {
        let mut init = BTreeMap::<TensorHandle, Vec<Tensor>>::new();
        for (i, g) in input {
            init.entry(i.clone().into()).or_default().push(g.clone());
        }

        let mut s = Self {
            table: HashMap::new(),
        };

        for (i, mut g) in init {
            g.push(s.compute(&i));
            let n = AddTensor::tensor(i.shape(), g);
            *s.table.get_mut(&i).unwrap() = Some(n);
        }

        s
    }

    pub fn compute(&mut self, tensor: &Tensor) -> Tensor {
        if let Some(r) = self.table.get(tensor.into()) {
            return r.clone().unwrap();
        }
        self.table.insert(tensor.clone().into(), None);
        let r = tensor.operator().forward_grad(tensor, self);
        let node = self.table.get_mut(tensor.into()).unwrap();
        assert!(node.is_none());
        *node = Some(r.clone());
        r
    }
}
