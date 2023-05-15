use std::path::Path;
use std::sync::Arc;

use crate::core::add_tensor::AddTensor;
use crate::core::function::Function;
use crate::core::sum_scale::SumScale;
use crate::cpu::CpuContext;
use crate::model_context::ModelContext;
use crate::tensor::{data_size, Tensor};
use crate::tools::idx::IdxFile;
use crate::variable_inline::VariableInlineContext;

fn mnist_model_build(model: &mut ModelContext, input: &Tensor) -> Tensor {
    const INPUT: usize = 28 * 28;
    assert_eq!(data_size(input.shape()), INPUT);
    let layer = input.reshape([1, INPUT]);

    const M1: usize = 150;
    let fc = model.variable([INPUT, M1]);
    let layer = layer.matrix_mul(fc).reshape([M1]) + model.variable([M1]);
    let layer = (layer.apply(Function::ReLU) + layer * 0.01).reshape([1, M1]);

    const M2: usize = 50;
    let fc = model.variable([M1, M2]);
    let layer = layer.matrix_mul(fc).reshape([M2]) + model.variable([M2]);
    let layer = (layer.apply(Function::ReLU) + layer * 0.01).reshape([1, M2]);

    const M3: usize = 20;
    let fc = model.variable([M2, M3]);
    let layer = layer.matrix_mul(fc).reshape([M3]) + model.variable([M3]);
    let layer = (layer.apply(Function::ReLU) + layer * 0.01).reshape([1, M3]);

    let fc = model.variable([M3, 10]);
    let layer = layer.matrix_mul(fc).reshape([10]) + model.variable([10]);
    let layer = layer.apply(Function::ReLU) + layer * 0.01;

    layer
}

fn read_train_data<P: AsRef<Path>>(path: P) -> Tensor {
    let file = IdxFile::read_file(path).unwrap();
    let data = file.data.get_slice::<u8>().unwrap();
    let data = Arc::new(data.iter().copied().map(|x| x as f32 / 255.0).collect());
    let shape = file
        .dimensions
        .iter()
        .copied()
        .map(|x| x as usize)
        .collect::<Vec<_>>();
    Tensor::constant(shape, data)
}

fn read_train_labels<P: AsRef<Path>>(path: P) -> (Vec<u8>, Tensor) {
    let file = IdxFile::read_file(path).unwrap();
    let data = file.data.get_slice::<u8>().unwrap();
    let mut output = Vec::with_capacity(data.len() * 10);
    for i in 0..data.len() {
        let l = data[i];
        for _ in 0..l {
            output.push(0.0);
        }
        output.push(1.0);
        for _ in (l + 1)..10 {
            output.push(0.0);
        }
    }
    let shape = file
        .dimensions
        .iter()
        .copied()
        .map(|x| x as usize)
        .chain(std::iter::once(10))
        .collect::<Vec<_>>();
    (data.to_vec(), Tensor::constant(shape, Arc::new(output)))
}

fn model_loss_build(input: &Tensor, output: &Tensor, data: &[(Tensor, Tensor)]) -> Tensor {
    let mut all = Vec::new();
    for (i, f) in data {
        let mut c = VariableInlineContext::new();
        c.variable(input, i);
        let out = c.get(output) - f;
        all.push(SumScale::sum(out.powf(2.0)));
    }
    (AddTensor::add(all) / (data.len() as f32)).powf(0.5)
}

pub fn main() {
    let ref mut model = ModelContext::new();
    let ref input = Tensor::variable([28, 28]);
    let ref output = mnist_model_build(model, input);

    println!("{:?}", output.debug_define());

    const TRAIN_SIZE: usize = 10;
    let ref input_and_output = (0..TRAIN_SIZE)
        .map(|_| (Tensor::variable([28, 28]), Tensor::variable([10])))
        .collect::<Vec<_>>();
    let ref loss = model_loss_build(input, output, input_and_output);

    let ref data = read_train_data("data/mnist/train-images.idx3-ubyte");
    assert_eq!(data.shape(), [60000, 28, 28]);

    let (_, labels) = read_train_labels("data/mnist/train-labels.idx1-ubyte");
    assert_eq!(labels.shape(), [60000, 10]);
    for i in 0..5 {
        for t in 0..(60000 / TRAIN_SIZE) {
            let mut context = CpuContext::new();
            for i in 0..TRAIN_SIZE {
                context.input_constant_with(
                    &input_and_output[i].0,
                    &data.get([t * TRAIN_SIZE + i]),
                    [],
                );
                context.input_constant_with(
                    &input_and_output[i].1,
                    &labels.get([t * TRAIN_SIZE + i]),
                    [],
                );
            }
            model.optimization(&mut context, loss, 0.1).unwrap();

            if t % 1000 == 0 {
                // println!("{:?}", model);
                println!("# {:5} {}", i, t);
                println!("{:?}", context.compute(loss).unwrap());

                let mut context = CpuContext::new();
                model.load_to(&mut context);
                context.input_constant_with(input, &data.get([t * TRAIN_SIZE]), []);

                println!("{:10.6?}", labels.get([t * TRAIN_SIZE]).compute().unwrap());
                println!("{:10.6?}", context.compute(output).unwrap());
            }
        }
    }

    let test_data = read_train_data("data/mnist/t10k-images.idx3-ubyte");
    assert_eq!(test_data.shape(), [10000, 28, 28]);
    let (test_result, test_labels) = read_train_labels("data/mnist/t10k-labels.idx1-ubyte");
    assert_eq!(test_labels.shape(), [10000, 10]);

    let mut cross = vec![0; 100];
    let mut correct = 0;
    for i in 0..test_result.len() {
        let mut context = CpuContext::new();
        context.input_constant_with(input, &test_data.get([i]), []);
        model.load_to(&mut context);
        let out = context.compute(output).unwrap();
        let o = out
            .iter()
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        cross[test_result[i] as usize * 10 + o] += 1;
        if test_result[i] as usize == o {
            correct += 1;
        }
    }
    for i in 0..10 {
        println!("{:4?}", &cross[(i * 10)..(i * 10 + 10)]);
    }

    println!("{}", correct as f32 / test_result.len() as f32);
}
