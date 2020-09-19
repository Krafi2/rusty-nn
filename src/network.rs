use std::convert::TryFrom;
use std::error;
use std::fmt;
use std::fs;

use serde::{
    de::{self, SeqAccess, Visitor},
    ser::SerializeSeq,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::allocator::{Allocator, Mediator, WeightHndl};
use crate::f32s;
use crate::helpers::{least_size, to_scalar};
use crate::layers::{Layer, LayerBuilder, LayerType};

pub struct NetworkBuilder {
    weights: Vec<f32s>,
    layers: Vec<LayerType>,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        NetworkBuilder {
            weights: Vec::new(),
            layers: Vec::new(),
        }
    }

    /// Adds a single layer to the network.
    pub fn add<T>(mut self, layer_builder: T) -> Self
    where
        T: LayerBuilder,
        T::Output: Into<LayerType>,
    {
        let shape = self.layers.last().map(|l| l.out_shape());
        self.layers.push(
            layer_builder
                .connect(shape, Allocator::new(&mut self.weights))
                .into(),
        );
        self
    }
    /// Adds all of the layers provided by the `builders` argument.
    pub fn add_layers<T>(mut self, builders: T) -> Self
    where
        T: IntoIterator,
        T::Item: LayerBuilder,
        <T::Item as LayerBuilder>::Output: Into<LayerType>,
    {
        for builder in builders {
            self = self.add(builder);
        }
        self
    }

    /// Builds the network. Returns None if no layers had been provided.
    pub fn build(self) -> Option<Network> {
        if self.layers.is_empty() {
            None
        } else {
            Some(Network {
                weights: self.weights,
                layers: self.layers,
            })
        }
    }
}

/// This struct represents a neural network and supports the basic functionality of giving predictions based on provided input.
/// Additionally, it can be both saved to and loaded from a file.
#[derive(Clone, Serialize, Deserialize)]
#[serde(into = "NetworkUnvalidated", try_from = "NetworkUnvalidated")]
pub struct Network {
    pub(crate) weights: Vec<f32s>,
    pub(crate) layers: Vec<LayerType>,
}

impl Network {
    pub fn from_file(path: &str) -> anyhow::Result<Network> {
        let s = fs::read_to_string(path)?;
        let network: Network = serde_json::from_str(&s)?;
        Ok(network)
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        fs::write(path, serde_json::to_string(&self)?)?;
        Ok(())
    }

    /// Propagate input through the network. `input` has to be the same size as the network's ipnut layer.
    pub fn eval<'a>(&'a mut self, input: &'a [f32]) {
        // This should align the data
        self.set_input(input);

        let (l, layers) = self.layers.split_first_mut().unwrap();
        let mut input = l.get_output();

        for l in layers {
            l.eval(input, Mediator::new(&self.weights));
            input = l.get_output();
        }
    }

    pub fn debug(&self) {
        println!("Diagnostic information for network:");
        for (i, l) in self.layers.iter().enumerate() {
            println!("Layer {}:", i);
            println!("{}", l.debug(self.get_weights()));
        }
    }

    //getters
    pub fn size(&self) -> usize {
        self.layers.len()
    }
    pub fn in_size(&self) -> usize {
        self.layers.first().unwrap().get_size()
    }
    pub fn out_size(&self) -> usize {
        self.layers.last().unwrap().get_size()
    }
    pub fn output(&self) -> &[f32] {
        &to_scalar(self.layers.last().unwrap().get_output())[..self.out_size()]
    }
    pub fn layers(&self) -> &[LayerType] {
        &self.layers
    }
    pub fn layers_mut(&mut self) -> &mut Vec<LayerType> {
        &mut self.layers
    }
    pub fn get_weights(&self) -> Mediator<&[f32s], WeightHndl> {
        Mediator::new(&self.weights)
    }
    pub fn get_weights_mut(&mut self) -> Mediator<&mut [f32s], WeightHndl> {
        Mediator::new(&mut self.weights)
    }
    pub fn weights_mut(&mut self) -> &mut [f32s] {
        &mut self.weights
    }
    pub fn weight_count(&self) -> usize {
        self.weights.len() * f32s::lanes()
    }

    //setters
    /// Private method to set the network's input
    fn set_input(&mut self, input: &[f32]) {
        // this call will panic if the first layer doesn't support activation assignment
        self.layers
            .first_mut()
            .expect("Network is empty")
            .set_activations(input);
    }

    //TODO actually panic if the layers arent the same
    /// Copies weights from another network. Panics if the networks don't have the same structure.
    pub fn copy_weights(&mut self, other: &Network) {
        self.weights.copy_from_slice(&other.weights);
    }
}

/// When deserializing, we first construct this object, validate that it's structure is correct and convert to Network
#[derive(Serialize, Deserialize)]
struct NetworkUnvalidated {
    #[serde(
        serialize_with = "serialize_simd",
        deserialize_with = "deserialize_simd"
    )]
    pub weights: Vec<f32s>,
    pub layers: Vec<LayerType>,
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
				f.write_fmt(format_args!("Expected at least {} weights but there are only {}.",
					expected,
					weights
				))
			}
		    ConsError::Empty => {
				f.write_str("The network must have at least a single layer, but it was empty.")
			}
		}
    }
}

impl Into<NetworkUnvalidated> for Network {
    fn into(self) -> NetworkUnvalidated {
        NetworkUnvalidated {
            weights: self.weights,
            layers: self.layers,
        }
    }
}

impl TryFrom<NetworkUnvalidated> for Network {
    type Error = ConsError;
    fn try_from(mut value: NetworkUnvalidated) -> Result<Self, Self::Error> {
        let mut weights: usize = 0;
        let mut iter = value.layers.iter().peekable();
        for i in 0.. {
            if let Some(prev_l) = iter.next() {
                weights += prev_l.get_weight_count();
                if let Some(l) = iter.peek() {
                    if prev_l.get_size() != l.get_in_size() {
                        return Err(ConsError::Incompatible {
                            index: i,
                            received_input: prev_l.get_size(),
                            expected_input: l.get_in_size(),
                        });
                    }
                } else {
                    break;
                }
            } else {
                return Err(ConsError::Empty);
            }
        }
        if weights > value.weights.len() * f32s::lanes() {
            return Err(ConsError::NotEnoughWeights {
                weights,
                expected: value.weights.len() * f32s::lanes(),
            });
        }
        for l in &mut value.layers {
            l.rebuild();
        }

        Ok(Network {
            weights: value.weights,
            layers: value.layers,
        })
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
        if let Some(len) = seq.size_hint() {
            let mut vec: Vec<f32s> = Vec::with_capacity(least_size(len, f32s::lanes()));

            let mut len = 0;
            let cap = vec.capacity() * f32s::lanes();

            unsafe {
                let vec_: *mut f32 = vec.as_mut_ptr() as *mut f32;
                while let Some(e) = seq.next_element()? {
                    vec_.add(len).write(e);

                    len += 1;
                    if len > cap {
                        panic!("size hint lied");
                    }
                }
                assert!(len % f32s::lanes() == 0);
                vec.set_len(len / f32s::lanes());
                return Ok(vec);
            }
        } else {
            //if the length isnt known, switch to this slower variant
            let mut vec = Vec::new();

            for len in 0.. {
                vec.reserve(1);

                unsafe {
                    let vec_: *mut f32 = vec.as_mut_ptr() as *mut f32;
                    let mut i = 0;
                    while let Some(e) = seq.next_element()? {
                        vec_.add(len * f32s::lanes() + i).write(e);
                        i += 1;
                        if i == f32s::lanes() {
                            break;
                        }
                    }

                    vec.set_len(len + 1);

                    match i {
                        //all lanes are initialized so we can safely return
                        0 => return Ok(vec),
                        //all lanes are initialized so we can continue
                        x if x == f32s::lanes() => (),
                        //some lanes were not initialized so we return an error
                        _ => {
                            return Err(<A::Error as de::Error>::custom(format!(
                                "Number of elements must be a multiple of {}, received: {}",
                                f32s::lanes(),
                                len * f32s::lanes() + i
                            )))
                        }
                    }
                }
            }
        }
        unreachable!()
    }
}

pub fn serialize_simd<S>(vec: &[f32s], s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = s.serialize_seq(Some(vec.len()))?;
    for e in to_scalar(vec) {
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
