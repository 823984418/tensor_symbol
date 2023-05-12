use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub mod idx;

pub fn rand(len: usize) -> Vec<f32> {
    let mut rng = SmallRng::from_entropy();
    let mut output = Vec::with_capacity(len);
    for _ in 0..len {
        output.push(rng.gen());
    }
    output
}
