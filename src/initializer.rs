use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

//I used this blog post as reference to the initialization methods ->
//https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

/// Implement Initializer for the struct reference as well
macro_rules! impl_ref {
    ($struct:ty) => {
        impl Initializer for &mut $struct {
            fn get(&mut self, in_size: usize, size: usize) -> f32 {
                <$struct as Initializer>::get(self, in_size, size)
            }
        }
    };
}

pub trait Initializer {
    fn get(&mut self, in_size: usize, size: usize) -> f32;
}

///Xavier initialization should be used for layers with symetric activation functions such as sigmoid or tanH
pub struct XavierInit {
    rng: SmallRng,
}
impl XavierInit {
    pub fn new() -> XavierInit {
        XavierInit {
            rng: SmallRng::seed_from_u64(0u64),
        }
    }
}

impl Initializer for XavierInit {
    fn get(&mut self, in_size: usize, _size: usize) -> f32 {
        self.rng.sample::<f32, StandardNormal>(StandardNormal) / (in_size as f32).sqrt()
    }
}
impl_ref!(XavierInit);

///Kaiming initialization should be used for layers with asymetric activation functions such as RELU
pub struct KaimingInit {
    rng: SmallRng,
}
impl KaimingInit {
    pub fn new() -> KaimingInit {
        KaimingInit {
            rng: SmallRng::seed_from_u64(0u64),
        }
    }
}
impl Initializer for KaimingInit {
    fn get(&mut self, in_size: usize, _: usize) -> f32 {
        self.rng.sample::<f32, StandardNormal>(StandardNormal) * (2f32 / (in_size as f32)).sqrt()
    }
}
impl_ref!(KaimingInit);

///Always initializes weights to one
pub struct IdentityInit;
impl Initializer for IdentityInit {
    fn get(&mut self, _: usize, _: usize) -> f32 {
        1f32
    }
}
impl_ref!(IdentityInit);
