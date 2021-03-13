use crate::{f32s, helpers::VectorAdapter};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::StandardNormal;
use std::sync::atomic::{AtomicU64, Ordering};

//I used this blog post as reference to the initialization methods ->
//https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

pub trait Initializer {
    type Iter: Iterator<Item = f32>;

    fn construct(self, in_size: usize, size: usize) -> Self::Iter;
}

pub trait Init {
    fn get(&mut self, in_size: usize, size: usize) -> f32;
}

fn seeder() -> u64 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

pub use misc::InitAdapter;
mod misc {
    use super::*;

    pub struct InitAdapter<T> {
        inner: T,
        in_size: usize,
        size: usize,
    }

    impl<T> InitAdapter<T> {
        pub fn new(inner: T, in_size: usize, size: usize) -> Self {
            Self {
                inner,
                in_size,
                size,
            }
        }
    }

    impl<T> Iterator for InitAdapter<T>
    where
        T: Init,
    {
        type Item = f32;

        fn next(&mut self) -> Option<Self::Item> {
            Some(self.inner.get(self.in_size, self.size))
        }
    }

    // pub struct InitializerBase<T> {
    //     adapter: VectorAdapter<InitAdapter<T>>,
    // }

    // impl<T> InitializerBase<T>
    // where
    //     T: Init,
    // {
    //     pub fn new(init: T, in_size: usize, size: usize) -> Self {
    //         Self {
    //             adapter: VectorAdapter::new(InitAdapter::new(init, in_size, size)),
    //         }
    //     }
    // }

    // impl<T> Iterator for InitializerBase<T>
    // where
    //     T: Init,
    // {
    //     type Item = f32s;

    //     fn next(&mut self) -> Option<Self::Item> {
    //         self.adapter.next()
    //     }
    // }

    impl<T> Initializer for T
    where
        T: Iterator<Item = f32>,
    {
        type Iter = T;

        fn construct(self, _in_size: usize, _size: usize) -> Self::Iter {
            self
        }
    }

    // impl<T> Initializer for T
    // where
    //     T: Iterator<Item = f32s>,
    // {
    //     type Iter = T;

    //     fn construct(self, _in_size: usize, _size: usize) -> Self::Iter {
    //         self
    //     }
    // }
}

pub use xavier::Xavier;
mod xavier {
    use super::*;

    /// Xavier initialization should be used for layers with symetric activation functions such as sigmoid or tanH
    pub struct Xavier {
        rng: SmallRng,
    }

    impl Xavier {
        pub fn new() -> Self {
            Self {
                rng: SmallRng::seed_from_u64(seeder()),
            }
        }

        pub fn seed(seed: u64) -> Self {
            Self {
                rng: SmallRng::seed_from_u64(seed),
            }
        }
    }

    impl Init for Xavier {
        fn get(&mut self, in_size: usize, size: usize) -> f32 {
            self.rng.sample::<f32, StandardNormal>(StandardNormal) / (in_size as f32).sqrt()
        }
    }

    impl Initializer for Xavier {
        type Iter = InitAdapter<Self>;

        fn construct(self, in_size: usize, size: usize) -> Self::Iter {
            InitAdapter::new(self, in_size, size)
        }
    }
}

pub use kaiming::Kaiming;
mod kaiming {
    use super::*;

    /// Kaiming initialization should be used for layers with asymetric activation functions such as RELU
    pub struct Kaiming {
        rng: SmallRng,
    }

    impl Kaiming {
        pub fn new() -> Self {
            Self {
                rng: SmallRng::seed_from_u64(seeder()),
            }
        }

        pub fn seed(seed: u64) -> Self {
            Self {
                rng: SmallRng::seed_from_u64(seed),
            }
        }
    }

    impl Init for Kaiming {
        fn get(&mut self, in_size: usize, size: usize) -> f32 {
            self.rng.sample::<f32, StandardNormal>(StandardNormal)
                * (2f32 / (in_size as f32)).sqrt()
        }
    }

    impl Initializer for Kaiming {
        type Iter = InitAdapter<Self>;

        fn construct(self, in_size: usize, size: usize) -> Self::Iter {
            InitAdapter::new(self, in_size, size)
        }
    }
}

pub use normal::Normal;
mod normal {
    use super::*;

    /// Kaiming initialization should be used for layers with asymetric activation functions such as RELU
    pub struct Normal {
        rng: SmallRng,
    }

    impl Normal {
        pub fn new() -> Self {
            Self {
                rng: SmallRng::seed_from_u64(seeder()),
            }
        }

        pub fn seed(seed: u64) -> Self {
            Self {
                rng: SmallRng::seed_from_u64(seed),
            }
        }
    }

    impl Init for Normal {
        fn get(&mut self, _in_size: usize, _size: usize) -> f32 {
            self.rng.sample::<f32, StandardNormal>(StandardNormal)
        }
    }

    impl Initializer for Normal {
        type Iter = InitAdapter<Self>;

        fn construct(self, in_size: usize, size: usize) -> Self::Iter {
            InitAdapter::new(self, in_size, size)
        }
    }
}

pub use ones::Ones;
mod ones {
    use super::*;
    use std::iter::{repeat, Repeat};

    pub struct Ones;

    impl Initializer for Ones {
        type Iter = Repeat<f32>;

        fn construct(self, _in_size: usize, _size: usize) -> Self::Iter {
            repeat(1.)
        }
    }
}
