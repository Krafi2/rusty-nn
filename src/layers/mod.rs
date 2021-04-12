pub mod dense_layer;
pub mod map_layer;

pub use dense_layer::DenseBuilder;
pub use map_layer::MapBuilder;

use self::dense_layer::DenseLayer;
use self::map_layer::MapLayer;

use crate::{
    a_funcs::{ActivFunc, Identity, ReLU, SiLU, Sigmoid, TanH},
    f32s,
    misc::simd::to_blocks,
    serde::boxed_simd,
    storage::{DualAllocator, GradStorage, WeightStorage},
};

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use std::ops::{Deref, DerefMut};

/// This enum represents the architecture of a layer. It is primarily used
/// in the BasicLayer enum to fully describe a layer.
#[enum_dispatch(Layer)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum LayerArch<T: ActivFunc> {
    DenseLayer(DenseLayer<T>),
    MapLayer(MapLayer<T>),
}

/// This enum describes the architecture and activation function of a layer
/// so it can be easily serialized and deserialized.
#[enum_dispatch(Layer)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum BasicLayer {
    Sigmoid(LayerArch<Sigmoid>),
    Identity(LayerArch<Identity>),
    TanH(LayerArch<TanH>),
    SiLU(LayerArch<SiLU>),
    ReLU(LayerArch<ReLU>),
}

#[enum_dispatch]
pub trait Layer {
    /// Evaluate the layer's output.
    fn eval(&mut self, input: &Aligned, weights: &WeightStorage) -> &Aligned;

    /// Calculates derivatives of the layer's weights. `in_deriv` are the partial derivatives at the end of the next layer
    /// and `out_deriv` are the derivatives at the layer's input, which will be filled in.
    /// Returns None if the layer isn't readied.
    fn calc_gradients(
        &mut self,
        input: &Aligned,
        weights: &WeightStorage,
        gradients: &mut GradStorage,
        in_grads: &Aligned,
        out_grads: &mut Aligned,
    );

    fn activations(&self) -> &Aligned;

    fn input(&self) -> Shape;

    /// Get layer's output
    fn output(&self) -> Shape;
}

/// Trait all layer constructors must implement in order to be added to a NetworkBuilder via the add function.
pub trait LayerBuilder {
    type Output;
    /// Connect a layer to the previous one. `input` is the shape of the previous layer's output.
    fn connect(self, in_shape: Shape, alloc: &mut DualAllocator) -> Self::Output;
}

pub use shape::Shape;
mod shape {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Shape {
        scalar: usize,
        vector: usize,
    }

    impl Shape {
        pub fn new(len: usize) -> Self {
            Self {
                scalar: len,
                vector: to_blocks(len, f32s::lanes()),
            }
        }

        pub fn scalar(&self) -> usize {
            self.scalar
        }

        pub fn vector(&self) -> usize {
            self.vector
        }
    }
}

pub use aligned::Aligned;
mod aligned {
    use crate::misc::simd::{as_scalar, as_scalar_mut, into_scalar};

    use super::*;
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Aligned {
        len: usize,
        #[serde(with = "boxed_simd")]
        array: Box<[f32s]>,
    }

    impl Aligned {
        pub fn zeroed(len: usize) -> Self {
            let vec_len = to_blocks(len, f32s::lanes());
            Self {
                array: vec![f32s::splat(0.); vec_len].into_boxed_slice(),
                len,
            }
        }

        pub fn from_vector<T>(vec: T, len: usize) -> Self
        where
            T: Into<Box<[f32s]>>,
        {
            let array = vec.into();
            assert!(len <= array.len() * f32s::lanes());
            Self { len, array }
        }

        pub fn from_scalar<T>(vec: &T) -> Self
        where
            T: AsRef<[f32]>,
        {
            let array = vec.as_ref();
            let mut new = Self::zeroed(array.len());
            new.as_scalar_mut().copy_from_slice(array);
            new
        }

        pub fn as_scalar(&self) -> &[f32] {
            &as_scalar(&self.array)[..self.len]
        }

        pub fn as_vector(&self) -> &[f32s] {
            &self.array
        }

        pub fn as_scalar_mut(&mut self) -> &mut [f32] {
            &mut as_scalar_mut(&mut self.array)[..self.len]
        }

        pub fn as_vector_mut(&mut self) -> &mut [f32s] {
            &mut self.array
        }

        pub fn into_vector(self) -> Box<[f32s]> {
            self.array
        }

        pub fn into_scalar(self) -> Box<[f32]> {
            into_scalar(self.array)
        }

        pub fn shape(&self) -> Shape {
            Shape::new(self.len)
        }

        pub fn eq_shape(&self, other: Aligned) -> bool {
            self.shape() == other.shape()
        }
    }

    impl AsRef<[f32]> for Aligned {
        fn as_ref(&self) -> &[f32] {
            self.as_scalar()
        }
    }

    impl AsMut<[f32]> for Aligned {
        fn as_mut(&mut self) -> &mut [f32] {
            self.as_scalar_mut()
        }
    }

    impl AsRef<[f32s]> for Aligned {
        fn as_ref(&self) -> &[f32s] {
            self.as_vector()
        }
    }

    impl AsMut<[f32s]> for Aligned {
        fn as_mut(&mut self) -> &mut [f32s] {
            self.as_vector_mut()
        }
    }
}

pub use no_value::{deserialize, serialize};
pub mod no_value {
    use super::*;
    use serde::{
        de::{self, MapAccess, Visitor},
        ser::SerializeStruct,
        Deserializer, Serializer,
    };

    pub fn serialize<S>(aligned: &Aligned, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = ser.serialize_struct("aligned_no_value", 1)?;
        state.serialize_field("len", &aligned.shape().scalar())?;
        state.end()
    }

    struct NoValueVisitor;

    impl<'de> Visitor<'de> for NoValueVisitor {
        type Value = Aligned;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a map containing len")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            match map.next_entry()? {
                Some(("len", len)) => Ok(Aligned::zeroed(len)),
                Some((field, _)) => Err(<A::Error as de::Error>::unknown_field(field, &["len"])),
                None => Err(<A::Error as de::Error>::missing_field("len")),
            }
        }
    }

    pub fn deserialize<'de, D>(de: D) -> Result<Aligned, D::Error>
    where
        D: Deserializer<'de>,
    {
        de.deserialize_struct("aligned_no_value", &["len"], NoValueVisitor)
    }
}

macro_rules! impl_layer_from_deref {
    () => {
        fn eval(&mut self, input: &Aligned, weights: &WeightStorage) -> &Aligned {
            self.deref_mut().eval(input, weights)
        }

        fn calc_gradients(
            &mut self,
            input: &Aligned,
            weights: &WeightStorage,
            gradients: &mut GradStorage,
            in_grads: &Aligned,
            out_grads: &mut Aligned,
        ) {
            self.deref_mut()
                .calc_gradients(input, weights, gradients, in_grads, out_grads)
        }

        fn activations(&self) -> &Aligned {
            self.deref().activations()
        }

        fn input(&self) -> Shape {
            self.deref().input()
        }

        fn output(&self) -> Shape {
            self.deref().output()
        }
    };
}

impl<T: Layer + ?Sized> Layer for Box<T> {
    impl_layer_from_deref!();
}

#[cfg(test)]
mod tests {
    /// Compares two arrays with the given error tolerance. Returns None if either of the arrays contains NaN.
    pub(crate) fn is_equal_ish(left: &[f32], right: &[f32], tolerance: f32) -> Option<bool> {
        assert_eq!(left.len(), right.len());
        let err = left
            .iter()
            .zip(right)
            .map(|(l, r)| f32::abs(l - r))
            .try_fold(0., |a, b| {
                a.partial_cmp(&b).map(|ord| match ord {
                    std::cmp::Ordering::Less => b,
                    std::cmp::Ordering::Equal => a,
                    std::cmp::Ordering::Greater => a,
                })
            });
        err.map(|e| e < tolerance)
    }

    pub(crate) fn check(expected: &[f32], output: &[f32], tolerance: f32, id: &str) {
        let diag = || format!("expected: {:?}\nreceived: {:?}", expected, output);

        match is_equal_ish(expected, output, tolerance) {
            Some(false) => panic!("Evaluation produced incorrect {}.\n{}", id, diag()),
            None => panic!("Evaluation produced a NaN\n{}", diag()),
            _ => {}
        }
    }
}
