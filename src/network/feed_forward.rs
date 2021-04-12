use std::{
    error::Error,
    fmt::{Debug, Display},
    fs::File,
    io::BufReader,
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    f32s,
    layers::{no_value, Aligned, BasicLayer, Layer, Shape},
    network::{construction::LinearConstruction, Network},
    storage::{DualAllocator, GradStorage, WeightStorage},
};

/// This struct represents a neural network and supports the basic functionality of giving predictions based on provided input.
/// Additionally, it can be both saved to and loaded from a file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FeedForward<T = BasicLayer> {
    #[serde(with = "no_value")]
    input: Aligned,
    weights: WeightStorage,
    layers: Vec<T>,
}

impl<T> Network for FeedForward<T>
where
    T: Layer,
{
    fn predict(&mut self, input: &[f32]) -> &Aligned {
        self.input.as_scalar_mut().copy_from_slice(input);

        let mut input = &self.input;
        for l in &mut self.layers {
            input = l.eval(input, &self.weights);
        }
        input
    }

    fn calc_gradients(&mut self, grads: &mut GradStorage, out_grads: &Aligned) {
        let mut iter = self.layers.iter_mut().rev().peekable();

        let mut in_grads = out_grads.to_owned();

        while let Some(layer) = iter.next() {
            let input = iter.peek().map_or(&self.input, |l| l.activations());
            let out_shape = layer.input();
            let mut out_grads = Aligned::zeroed(out_shape.scalar());

            layer.calc_gradients(input, &self.weights, grads, &in_grads, &mut out_grads);

            in_grads = out_grads;
        }
    }

    fn input(&self) -> Shape {
        self.layers.first().unwrap().input()
    }

    fn output(&self) -> Shape {
        self.layers.last().unwrap().output()
    }

    /// Get network weights.
    fn weights(&self) -> &[f32s] {
        self.weights.raw()
    }

    /// Get mutable weights.
    fn weights_mut(&mut self) -> &mut [f32s] {
        self.weights.raw_mut()
    }
}

impl<T> FeedForward<T>
where
    T: for<'de> Deserialize<'de>,
{
    pub fn from_file<P>(path: P) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        let network = serde_json::from_reader(reader)?;
        Ok(network)
    }
}

pub use construction::*;
mod construction {
    use crate::network::construction::Construction;

    use super::*;

    pub enum ConsError {
        Empty,
    }

    impl Debug for ConsError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ConsError::Empty => f.write_str("Network construction failed: No layers."),
            }
        }
    }

    impl Display for ConsError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ConsError::Empty => f.write_str("Network construction failed: No layers."),
            }
        }
    }

    impl Error for ConsError {}

    impl<T> LinearConstruction for FeedForward<T>
    where
        T: Layer,
    {
        type Layer = T;
        type Error = ConsError;
        type Output = Construction<Self>;

        fn construct(
            allocator: DualAllocator,
            layers: Vec<Self::Layer>,
        ) -> Result<Self::Output, Self::Error> {
            if layers.is_empty() {
                Err(ConsError::Empty)
            } else {
                let (weights, g_allocator) = allocator.finish();
                Ok(Construction::new(
                    Self {
                        input: Aligned::zeroed(layers.first().unwrap().input().scalar()),
                        weights,
                        layers,
                    },
                    g_allocator,
                ))
            }
        }
    }
}

// struct SimdVisitor;
// impl<'de> Visitor<'de> for SimdVisitor {
//     type Value = Vec<f32s>;

//     fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
//         formatter.write_str("a sequence of floats")
//     }
//     fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
//     where
//         A: SeqAccess<'de>,
//     {
//         let vec = if let Some(len) = seq.size_hint() {
//             if len % f32s::lanes() != 0 {
//                 return Err(<A::Error as de::Error>::custom(format!(
//                     "Number of elements must be a multiple of {}, received: {}",
//                     f32s::lanes(),
//                     len,
//                 )));
//             }
//             Vec::with_capacity(len / f32s::lanes())
//         } else {
//             Vec::new()
//         };

//         let len = vec.len();
//         let cap = vec.capacity();
//         // we create the float vector this way to ensure alignment
//         let mut vec = unsafe {
//             Vec::from_raw_parts(
//                 vec.leak().as_mut_ptr() as *mut f32,
//                 len * f32s::lanes(),
//                 cap * f32s::lanes(),
//             )
//         };

//         while let Some(e) = seq.next_element()? {
//             vec.push(e);
//         }

//         if vec.len() % f32s::lanes() != 0 {
//             return Err(<A::Error as de::Error>::custom(format!(
//                 "Number of elements must be a multiple of {}, received: {}",
//                 f32s::lanes(),
//                 vec.len(),
//             )));
//         }

//         vec.shrink_to_fit();

//         let len = vec.len();
//         let cap = vec.capacity();
//         unsafe {
//             Ok(Vec::from_raw_parts(
//                 vec.leak().as_mut_ptr() as *mut f32s,
//                 len / f32s::lanes(),
//                 cap / f32s::lanes(),
//             ))
//         }
//     }
// }

// pub fn serialize_simd<S>(vec: &[f32s], s: S) -> Result<S::Ok, S::Error>
// where
//     S: Serializer,
// {
//     let mut seq = s.serialize_seq(Some(vec.len()))?;
//     for e in as_scalar(vec) {
//         seq.serialize_element(e)?;
//     }
//     seq.end()
// }
// pub fn deserialize_simd<'de, D>(d: D) -> Result<Vec<f32s>, D::Error>
// where
//     D: Deserializer<'de>,
// {
//     d.deserialize_seq(SimdVisitor)
// }
