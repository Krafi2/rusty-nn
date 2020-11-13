use std::convert::TryFrom;
use std::error;
use std::fmt;
use std::fs;

use serde::{
    de::{self, SeqAccess, Visitor},
    ser::SerializeSeq,
    Deserialize, Deserializer, Serialize, Serializer,
};

use super::{construction::LinearConstruction, CalcGradients, Network};
use crate::allocator::Mediator;
use crate::f32s;
use crate::helpers::{as_scalar, as_scalar_mut, zero_simd};
use crate::layers::{GradError, Layer, LayerGradients};

/// This struct represents a neural network and supports the basic functionality of giving predictions based on provided input.
/// Additionally, it can be both saved to and loaded from a file.
#[derive(Serialize, Deserialize, Debug)]
#[serde(into = "NetworkUnvalidated<T>", try_from = "NetworkUnvalidated<T>")]
pub struct FeedForward<T: Layer + Clone> {
    weights: Vec<f32s>,
    layers: Vec<T>,
    #[serde(skip)]
    grads: Option<Grads>,
}

/// This struct behaves exactly the same as [FeedForward](self::FeedForward) with the exception
/// that the layers it contains don't need to implement Serialize and Deserialize.
/// This means that it can't be saved but can use layers which cannot be serialized.
#[derive(Debug)]
pub struct FeedForwardNoSer<T: Layer> {
    weights: Vec<f32s>,
    layers: Vec<T>,
    grads: Option<Grads>,
}

#[derive(Clone, Debug)]
pub struct Grads {
    gradients: Vec<f32s>,
    buffer1: Vec<f32s>,
    buffer2: Vec<f32s>,
}

impl Grads {
    fn new<T: Layer>(weights: &[f32s], layers: &[T]) -> Self {
        let wc = weights.len();
        let max = layers.iter().map(|l| l.out_size()).max().unwrap();

        Self {
            gradients: vec![f32s::splat(0.); wc],
            buffer1: vec![f32s::splat(0.); max],
            buffer2: vec![f32s::splat(0.); max],
        }
    }
}

macro_rules! feed_forward_impl {
    () => {
        fn predict(&mut self, input: &[f32]) -> &[f32s] {
            self.layers.first_mut().unwrap().set_activations(input);

            let (l, layers) = self.layers.split_first_mut().unwrap();
            let mut input = l.output();
            for l in layers {
                l.eval(input, Mediator::new(&self.weights));
                input = l.output();
            }
            self.output()
        }

        /// Get network output.
        fn output(&self) -> &[f32s] {
            self.layers.last().unwrap().output()
        }

        /// Get network weights.
        fn weights(&self) -> &[f32s] {
            &self.weights
        }

        /// Get mutable weights.
        fn weights_mut(&mut self) -> &mut [f32s] {
            &mut self.weights
        }

        /// Returns input size of the network
        fn in_size(&self) -> usize {
            self.layers.first().unwrap().out_size()
        }

        /// Return output size of the network
        fn out_size(&self) -> usize {
            self.layers.last().unwrap().out_size()
        }
    };
}

macro_rules! calc_grads_impl {
    () => {
        fn ready(&mut self) {
            if self.grads.is_none() {
                self.grads = Some(Grads::new(&self.weights, &self.layers));
                for l in &mut self.layers {
                    l.ready();
                }
            }
        }

        fn unready(&mut self) {
            if self.grads.is_some() {
                self.grads.take();
                for l in &mut self.layers {
                    l.unready();
                }
            }
        }

        fn calc_gradients(&mut self, output_gradients: &[f32]) -> Result<(), GradError> {
            if let Some(grads) = &mut self.grads {
                let size = self.layers.last().unwrap().out_size();
                as_scalar_mut(&mut grads.buffer1)[..size].copy_from_slice(output_gradients);

                let mut buffer1 = &mut grads.buffer1;
                let mut buffer2 = &mut grads.buffer2;

                let mut iter = self.layers.iter_mut().rev().peekable();

                while let Some(layer) = iter.next() {
                    if let Some(prev_layer) = iter.peek() {
                        let in_size = prev_layer.actual_out();
                        let out_size = layer.actual_out();
                        let out_deriv = &mut buffer2[..in_size];
                        let in_deriv = &mut buffer1[..out_size];
                        zero_simd(out_deriv); //zero out space needed in the buffer

                        let inputs = prev_layer.output();

                        layer
                            .calc_gradients(
                                Mediator::new(&self.weights),
                                Mediator::new(&mut grads.gradients),
                                inputs,
                                in_deriv,
                                out_deriv,
                            )
                            .expect("Layer wasn't readied for gradient calculation");

                        // swap the buffers without copying the contents as std::mem::swap does
                        let temp = buffer1;
                        buffer1 = buffer2;
                        buffer2 = temp;
                    }
                }
                Ok(())
            } else {
                Err(GradError::new())
            }
        }

        fn gradients(&self) -> Result<&[f32s], GradError> {
            self.grads
                .as_ref()
                .map(|g| g.gradients.as_ref())
                .ok_or_else(GradError::new)
        }

        fn gradients_mut(&mut self) -> Result<&mut [f32s], GradError> {
            self.grads
                .as_mut()
                .map(|g| g.gradients.as_mut())
                .ok_or_else(GradError::new)
        }

        fn reset_gradients(&mut self) -> Result<(), GradError> {
            if let Some(grads) = &mut self.grads {
                zero_simd(&mut grads.gradients);
                Ok(())
            } else {
                Err(GradError::new())
            }
        }

        fn weight_grads_mut(&mut self) -> Result<(&mut [f32s], &mut [f32s]), GradError> {
            if let Some(grads) = &mut self.grads {
                Ok((&mut self.weights, &mut grads.gradients))
            } else {
                Err(GradError::new())
            }
        }
    };
}

impl<T: Layer + Clone> Network for FeedForward<T> {
    feed_forward_impl!();
}

impl<T: Layer + Clone + LayerGradients> CalcGradients for FeedForward<T> {
    calc_grads_impl!();
}

impl<T: Layer> Network for FeedForwardNoSer<T> {
    feed_forward_impl!();
}

impl<T: Layer + LayerGradients> CalcGradients for FeedForwardNoSer<T> {
    calc_grads_impl!();
}

impl<T: Layer + Clone> FeedForward<T>
where
    for<'de> T: Deserialize<'de>,
{
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let s = fs::read_to_string(path)?;
        let network: Self = serde_json::from_str(&s)?;
        Ok(network)
    }
}

impl<T: Layer + Clone> LinearConstruction<T> for FeedForward<T> {
    fn construct(weights: Vec<f32s>, layers: Vec<T>) -> Self {
        Self {
            weights,
            layers,
            grads: None,
        }
    }
}

impl<T: Layer> LinearConstruction<T> for FeedForwardNoSer<T> {
    fn construct(weights: Vec<f32s>, layers: Vec<T>) -> Self {
        Self {
            weights,
            layers,
            grads: None,
        }
    }
}

impl<T: Layer + Clone> Into<NetworkUnvalidated<T>> for FeedForward<T> {
    fn into(self) -> NetworkUnvalidated<T> {
        NetworkUnvalidated {
            weights: self.weights,
            layers: self.layers,
        }
    }
}

impl<T: Layer + Clone> Clone for FeedForward<T> {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            layers: self.layers.clone(),
            grads: self.grads.clone(),
        }
    }
}

impl<T: Layer + Clone> Clone for FeedForwardNoSer<T> {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            layers: self.layers.clone(),
            grads: self.grads.clone(),
        }
    }
}

/// When deserializing, we first construct this object, validate that it's structure is correct and convert to Network
#[derive(Serialize, Deserialize)]
struct NetworkUnvalidated<T: Layer> {
    #[serde(
        serialize_with = "serialize_simd",
        deserialize_with = "deserialize_simd"
    )]
    pub weights: Vec<f32s>,
    pub layers: Vec<T>,
}

/// An Error during the construction of a network.
#[derive(Debug)]
pub enum ConsError {
    /// A layer is incompatible with the previous one
    Incompatible {
        index: usize,
        received_input: usize,
        expected_input: usize,
    },
    /// Not enough weights had been given.
    NotEnoughWeights {
        weights: usize,
        expected: usize,
    },
    Empty,
}
impl error::Error for ConsError {}
impl fmt::Display for ConsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
		    ConsError::Incompatible { index, received_input, expected_input } => {
				f.write_fmt(format_args!("Layer {} is incompatible with layer {}:\n\tExpected input length of {} but received {}.",
					index,
					index - 1,
					expected_input,
					received_input,
			))
			}
		    ConsError::NotEnoughWeights { weights, expected } => {
				f.write_fmt(format_args!("Expected {} weights but only {} were provided.",
					expected,
					weights
				))
			}
		    ConsError::Empty => {
				f.write_str("The network must have at least a single layer, but it was empty.")
			}
        }?;
        f.write_str(" Error occured while attempting to construct a network from file.")?;
        Ok(())
    }
}

impl<T: Layer + Clone> TryFrom<NetworkUnvalidated<T>> for FeedForward<T> {
    type Error = ConsError;
    fn try_from(mut value: NetworkUnvalidated<T>) -> Result<Self, Self::Error> {
        let mut weights: usize = 0;
        let mut iter = value.layers.iter().peekable();
        for i in 0.. {
            if let Some(prev_l) = iter.next() {
                weights += prev_l.weight_count();
                if let Some(l) = iter.peek() {
                    if prev_l.out_size() != l.in_size() {
                        return Err(ConsError::Incompatible {
                            index: i,
                            received_input: prev_l.out_size(),
                            expected_input: l.in_size(),
                        });
                    }
                } else {
                    break;
                }
            } else {
                return Err(ConsError::Empty);
            }
        }
        if weights != value.weights.len() {
            return Err(ConsError::NotEnoughWeights {
                weights: value.weights.len() * f32s::lanes(),
                expected: weights * f32s::lanes(),
            });
        }
        for l in &mut value.layers {
            l.rebuild();
        }

        Ok(FeedForward::construct(value.weights, value.layers))
    }
}

struct SimdVisitor;
impl<'de> Visitor<'de> for SimdVisitor {
    type Value = Vec<f32s>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a sequence of floats")
    }
    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let vec = if let Some(len) = seq.size_hint() {
            if len % f32s::lanes() != 0 {
                return Err(<A::Error as de::Error>::custom(format!(
                    "Number of elements must be a multiple of {}, received: {}",
                    f32s::lanes(),
                    len,
                )));
            }
            Vec::with_capacity(len / f32s::lanes())
        } else {
            Vec::new()
        };

        let len = vec.len();
        let cap = vec.capacity();
        // we create the float vector this way to ensure alignment
        let mut vec = unsafe {
            Vec::from_raw_parts(
                vec.leak().as_mut_ptr() as *mut f32,
                len * f32s::lanes(),
                cap * f32s::lanes(),
            )
        };

        while let Some(e) = seq.next_element()? {
            vec.push(e);
        }

        if vec.len() % f32s::lanes() != 0 {
            return Err(<A::Error as de::Error>::custom(format!(
                "Number of elements must be a multiple of {}, received: {}",
                f32s::lanes(),
                vec.len(),
            )));
        }

        vec.shrink_to_fit();

        let len = vec.len();
        let cap = vec.capacity();
        unsafe {
            Ok(Vec::from_raw_parts(
                vec.leak().as_mut_ptr() as *mut f32s,
                len / f32s::lanes(),
                cap / f32s::lanes(),
            ))
        }
    }
}

pub fn serialize_simd<S>(vec: &[f32s], s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = s.serialize_seq(Some(vec.len()))?;
    for e in as_scalar(vec) {
        seq.serialize_element(e)?;
    }
    seq.end()
}
pub fn deserialize_simd<'de, D>(d: D) -> Result<Vec<f32s>, D::Error>
where
    D: Deserializer<'de>,
{
    d.deserialize_seq(SimdVisitor)
}
