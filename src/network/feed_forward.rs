use std::convert::TryFrom;
use std::error;
use std::fmt;
use std::fs;

use serde::{
    de::{self, SeqAccess, Visitor},
    ser::SerializeSeq,
    Deserialize, Deserializer, Serialize, Serializer,
};

use super::{construction::LinearConstruction, Network};
use crate::allocator::Mediator;
use crate::f32s;
use crate::helpers::{as_scalar, as_scalar_mut, zero_simd};
use crate::layers::Layer;

mod private_ {
    use super::*;
    pub trait FeedForwardMarker_ {
        type Layer: Layer;
        fn decompose(&self) -> (&[Self::Layer], &[f32s], &Option<Grads>);
        fn decompose_mut(&mut self) -> (&mut [Self::Layer], &mut [f32s], &mut Option<Grads>);
        fn try_save(&self, path: &std::path::Path) -> anyhow::Result<bool>;
    }
}
use private_::FeedForwardMarker_;

/// This struct represents a neural network and supports the basic functionality of giving predictions based on provided input.
/// Additionally, it can be both saved to and loaded from a file.
#[derive(Clone, Serialize, Deserialize)]
#[serde(into = "NetworkUnvalidated<T>", try_from = "NetworkUnvalidated<T>")]
pub struct FeedForward<T: Layer + Clone> {
    weights: Vec<f32s>,
    layers: Vec<T>,
    #[serde(skip)]
    grads: Option<Grads>,
}

impl<T: Layer + Clone + Serialize> FeedForwardMarker_ for FeedForward<T> {
    type Layer = T;

    fn try_save(&self, path: &std::path::Path) -> anyhow::Result<bool> {
        fs::write(path, serde_json::to_string(&self)?)?;
        Ok(true)
    }

    fn decompose(&self) -> (&[Self::Layer], &[f32s], &Option<Grads>) {
        (&self.layers, &self.weights, &self.grads)
    }

    fn decompose_mut(&mut self) -> (&mut [Self::Layer], &mut [f32s], &mut Option<Grads>) {
        (&mut self.layers, &mut self.weights, &mut self.grads)
    }
}

/// This struct behaves exactly the same as [FeedForward](self::FeedForward) with the exception
/// that the layers it contains don't need to implement Serialize and Deserialize.
/// This means that it can't be saved but can use layers which cannot be serialized.
#[derive(Clone)]
pub struct FeedForwardNoSer<T: Layer> {
    weights: Vec<f32s>,
    layers: Vec<T>,
    grads: Option<Grads>,
}
impl<T: Layer> FeedForwardMarker_ for FeedForwardNoSer<T> {
    type Layer = T;
    fn try_save(&self, _path: &std::path::Path) -> anyhow::Result<bool> {
        Ok(false) // We cannot save
    }

    fn decompose(&self) -> (&[Self::Layer], &[f32s], &Option<Grads>) {
        (&self.layers, &self.weights, &self.grads)
    }

    fn decompose_mut(&mut self) -> (&mut [Self::Layer], &mut [f32s], &mut Option<Grads>) {
        (&mut self.layers, &mut self.weights, &mut self.grads)
    }
}

impl<T: FeedForwardMarker_> Network for T {
    fn predict(&mut self, input: &[f32]) {
        let (layers, weights, _) = self.decompose_mut();
        layers.first_mut().unwrap().set_activations(input);

        let (l, layers) = layers.split_first_mut().unwrap();
        let mut input = l.output();
        for l in layers {
            l.eval(input, Mediator::new(weights));
            input = l.output();
        }
    }

    fn output(&self) -> &[f32s] {
        self.decompose().0.last().unwrap().output()
    }

    fn output_scalar(&self) -> &[f32] {
        let last = self.decompose().0.last().unwrap();
        &as_scalar(last.output())[..last.out_size()]
    }

    fn weights(&self) -> &[f32s] {
        self.decompose().1
    }

    fn weights_mut(&mut self) -> &mut [f32s] {
        self.decompose_mut().1
    }

    fn try_save(&self, path: &std::path::Path) -> anyhow::Result<bool> {
        self.try_save(path)
    }

    fn debug(&self) {
        println!("Diagnostic information for network:");
        for (i, l) in self.decompose().0.iter().enumerate() {
            println!("Layer {}:", i);
            println!("{}", l.debug(Mediator::new(self.weights())));
        }
    }

    fn in_size(&self) -> usize {
        self.decompose().0.first().unwrap().out_size()
    }

    fn out_size(&self) -> usize {
        self.decompose().0.last().unwrap().out_size()
    }

    fn ready(&mut self) {
        if self.decompose_mut().2.is_none() {
            let grads = Grads::new(self);
            self.decompose_mut().2.replace(grads);

            for l in self.decompose_mut().0 {
                l.ready();
            }
        }
    }

    fn unready(&mut self) {
        if self.decompose_mut().2.is_some() {
            self.decompose_mut().2.take();

            for l in self.decompose_mut().0 {
                l.unready();
            }
        }
    }

    fn calc_gradients(&mut self, output_gradients: &[f32]) -> Result<(), ()> {
        let (layers, weights, grads) = self.decompose_mut();
        if let Some(grads) = grads {
            let size = layers.last().unwrap().out_size();
            as_scalar_mut(&mut grads.buffer1)[..size].copy_from_slice(output_gradients);

            let mut buffer1 = &mut grads.buffer1;
            let mut buffer2 = &mut grads.buffer2;

            let mut iter = layers.iter_mut().rev().peekable();

            while let Some(layer) = iter.next() {
                if let Some(prev_layer) = iter.peek() {
                    let in_size = prev_layer.actual_out();
                    let out_size = layer.actual_out();
                    let out_deriv = &mut buffer2[..in_size];
                    let in_deriv = &mut buffer1[..out_size];
                    zero_simd(out_deriv); //zero out space needed in the buffer

                    let inputs = prev_layer.output();

                    layer
                        .calculate_derivatives(
                            Mediator::new(weights),
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
            Err(())
        }
    }

    fn gradients(&mut self) -> Result<&mut [f32s], ()> {
        self.decompose_mut().2.as_mut().map(|g| g.gradients.as_mut_slice()).ok_or(())
    }

    fn reset_gradients(&mut self) -> Result<(), ()> {
        self.decompose_mut().2.as_mut().map(|g| zero_simd(&mut g.gradients)).ok_or(())
    }

    fn weight_grads_mut(&mut self) -> Option<(&mut [f32s], &mut [f32s])> {
        let (_, weights, grads) = self.decompose_mut();
        grads
            .as_mut()
            .map(|g| (weights, g.gradients.as_mut_slice()))
    }
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

#[derive(Clone)]
pub struct Grads {
    gradients: Vec<f32s>,
    buffer1: Vec<f32s>,
    buffer2: Vec<f32s>,
}

impl Grads {
    fn new<T: FeedForwardMarker_>(net: &T) -> Self {
        let wc = net.weights().len();
        let max = net
            .decompose()
            .0
            .iter()
            .map(|l| l.out_size())
            .max()
            .unwrap();

        Self {
            gradients: vec![f32s::splat(0.); wc],
            buffer1: vec![f32s::splat(0.); max],
            buffer2: vec![f32s::splat(0.); max],
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
        let vec = 
        if let Some(len) = seq.size_hint() {
            if len % f32s::lanes() != 0 {
                return Err(<A::Error as de::Error>::custom(format!(
                    "Number of elements must be a multiple of {}, received: {}",
                    f32s::lanes(),
                    len,
                )))
            }
            Vec::with_capacity(len / f32s::lanes())
        }
        else {
            Vec::new()
        };

        let len = vec.len();
        let cap = vec.capacity();
        // we create the float vector this way to ensure alignment
        let mut vec =
        unsafe {
            Vec::from_raw_parts(vec.leak().as_mut_ptr() as *mut f32, len * f32s::lanes(), cap * f32s::lanes())
        };

        while let Some(e) = seq.next_element()? {
            vec.push(e);
        }

        if vec.len() % f32s::lanes() != 0 {
            return Err(<A::Error as de::Error>::custom(format!(
                "Number of elements must be a multiple of {}, received: {}",
                f32s::lanes(),
                vec.len(),
            )))
        }

        vec.shrink_to_fit();
        
        let len = vec.len();
        let cap = vec.capacity();
        unsafe {
            Ok(Vec::from_raw_parts(vec.leak().as_mut_ptr() as *mut f32s, len / f32s::lanes(), cap / f32s::lanes()))
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
