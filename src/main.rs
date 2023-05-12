pub mod core;
pub mod cpu;
pub mod demo;
pub mod grad;
pub mod model_context;
pub mod tensor;
pub mod tools;
pub mod variable_inline;

fn main() {
    demo::mnist::main();
}
