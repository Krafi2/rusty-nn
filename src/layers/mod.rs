pub mod dense_layer;
pub mod input_layer;
pub mod norm_layer;

pub use dense_layer::DenseBuilder;
pub use input_layer::InputBuilder;
pub use norm_layer::NormBuilder;

use self::dense_layer::DenseLayer;
use self::input_layer::InputLayer;
use self::norm_layer::NormLayer;

use crate::activation_functions::*;
use crate::allocator::{Allocator, GradHdnl, Mediator, WeightHndl};
use crate::f32s;

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use std::error::Error;
use std::fmt::Display;
use std::ops::{Deref, DerefMut};

//TODO maybe remove the Clone bound
#[enum_dispatch]
pub trait Layer {
    /// Reallocate memory needed for evaluation.
    /// Used after deserialization as this memory doesnt need to be serialized.
    fn rebuild(&mut self);

    /// Evaluate the layer's output.
    fn eval(&mut self, inputs: &[f32s], med: Mediator<&[f32s], WeightHndl>);

    /// Ready the network for optimization, if already readied do nothing.
    fn ready(&mut self);

    /// Undo the changes made by ready; if the network isn't readied do nothing.
    fn unready(&mut self);

    /// Calculates derivatives of the layer's weights. `in_deriv` are the partial derivatives at the end of the next layer
    /// and `out_deriv` are the derivatives at the layer's input, which will be filled in.
    /// Returns None if the layer isn't readied.
    fn calculate_derivatives(
        &mut self,
        weights: Mediator<&[f32s], WeightHndl>,
        self_deriv: Mediator<&mut [f32s], GradHdnl>,
        inputs: &[f32s],
        in_deriv: &[f32s],
        out_deriv: &mut [f32s],
    ) -> Result<(), GradError>;

    /// Return string containing formated debug information
    fn debug(&self, med: Mediator<&[f32s], WeightHndl>) -> String;

    /// Get layer's output
    fn output(&self) -> &[f32s];
    /// Get layer's output size
    fn out_size(&self) -> usize;
    /// Get the actual size of the output in terms of f32s
    fn actual_out(&self) -> usize;
    /// Get layer's input size
    fn in_size(&self) -> usize;
    /// Get the shape of the output
    fn out_shape(&self) -> OutShape;
    /// Get number of weights in the layer in terms of f32s
    fn weight_count(&self) -> usize;

    /// This function should panic for all non-input layer types
    fn set_activations(&mut self, _activations: &[f32]) {
        unimplemented!(
            "set_activations not implemented for {}",
            std::any::type_name::<Self>()
        )
    }
}

/// Trait all layer builders must implement in order to be added to a NetworkBuilder via the add function.
pub trait LayerBuilder {
    type Output: Layer;
    /// Connect a layer to the previous one. `shape` will be None if there are no layers before.
    fn connect(self, previous: Option<&dyn Layer>, alloc: Allocator) -> Self::Output;
}

/// Error encountered when calculating the weight gradients.
/// Currently this error simply means that the queried structure wasn't
/// properly initialized.
#[derive(Clone, Debug)]
pub struct GradError;

impl GradError {
    pub fn new() -> Self {
        Self
    }
}

impl Error for GradError {}

impl Display for GradError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Wasn't ready to calculate gradients")
    }
}

#[derive(Clone)]
pub struct OutShape {
    pub dims: Vec<usize>,
}

#[enum_dispatch(Layer)]
#[derive(Serialize, Deserialize, Clone)]
/// This enum represents the architecture of a layer. It is primarily used
/// in the BasicLayer enum to fully describe a layer.
pub enum LayerArch<T: ActivFunc> {
    DenseLayer(DenseLayer<T>),
    InputLayer(InputLayer),
    NormLayer(NormLayer<T>),
}

#[enum_dispatch(Layer)]
#[derive(Serialize, Deserialize, Clone)]
/// This enum describes the architecture and activation function of a layer
/// so it can be easily serialized and deserialized.
pub enum BasicLayer {
    Sigmoid(LayerArch<Sigmoid>),
    Identity(LayerArch<Identity>),
    TanH(LayerArch<TanH>),
    SiLU(LayerArch<SiLU>),
    ReLU(LayerArch<ReLU>),
}

// The conversion is implemented like this instead of a trait, because the from trait is incompatible
// with some trait definitions made by the enum_dispatch macro and this seems like the path of least resistance
// as I don't think anyone will ever need to pass LayerArch as a generic argument needing the trait
pub trait FromArch<T: ActivFunc> {
    fn from(arch: LayerArch<T>) -> BasicLayer;
}

macro_rules! impl_from_arch {
    ($t:tt) => {
        impl FromArch<$t> for BasicLayer {
            fn from(arch: LayerArch<$t>) -> Self {
                Self::$t(arch)
            }
        }
    };
}

impl_from_arch!(Sigmoid);
impl_from_arch!(Identity);
impl_from_arch!(TanH);
impl_from_arch!(SiLU);
impl_from_arch!(ReLU);

impl<T: Layer + ?Sized> Layer for Box<T> {
    fn rebuild(&mut self) {
        <Self as DerefMut>::deref_mut(self).rebuild()
    }

    fn eval(&mut self, inputs: &[f32s], med: Mediator<&[f32s], WeightHndl>) {
        <Self as DerefMut>::deref_mut(self).eval(inputs, med)
    }

    fn ready(&mut self) {
        <Self as DerefMut>::deref_mut(self).ready()
    }

    fn unready(&mut self) {
        <Self as DerefMut>::deref_mut(self).unready()
    }

    fn calculate_derivatives(
        &mut self,
        weights: Mediator<&[f32s], WeightHndl>,
        self_deriv: Mediator<&mut [f32s], GradHdnl>,
        inputs: &[f32s],
        in_deriv: &[f32s],
        out_deriv: &mut [f32s],
    ) -> Result<(), GradError> {
        <Self as DerefMut>::deref_mut(self)
            .calculate_derivatives(weights, self_deriv, inputs, in_deriv, out_deriv)
    }

    fn debug(&self, med: Mediator<&[f32s], WeightHndl>) -> String {
        <Self as Deref>::deref(self).debug(med)
    }

    fn output(&self) -> &[f32s] {
        <Self as Deref>::deref(self).output()
    }

    fn out_size(&self) -> usize {
        <Self as Deref>::deref(self).out_size()
    }

    fn actual_out(&self) -> usize {
        <Self as Deref>::deref(self).actual_out()
    }

    fn in_size(&self) -> usize {
        <Self as Deref>::deref(self).in_size()
    }

    fn out_shape(&self) -> OutShape {
        <Self as Deref>::deref(self).out_shape()
    }

    fn weight_count(&self) -> usize {
        <Self as Deref>::deref(self).weight_count()
    }

    fn set_activations(&mut self, activations: &[f32]) {
        <Self as DerefMut>::deref_mut(self).set_activations(activations)
    }
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
                if let Some(ord) = a.partial_cmp(&b) {
                    Some(match ord {
                        std::cmp::Ordering::Less => b,
                        std::cmp::Ordering::Equal => a,
                        std::cmp::Ordering::Greater => a,
                    })
                } else {
                    None
                }
            });
        err.map(|e| e < tolerance)
    }

    pub(crate) fn check(expected: &[f32], output: &[f32], tolerance: f32, id: &str) {
        let diag = || format!("expected: {:?}\nreceived: {:?}", expected, output);

        if let Some(eq) = is_equal_ish(expected, output, tolerance) {
            if eq {
                return;
            } else {
                panic!("Evaluation produced incorrect {}.\n{}", id, diag())
            }
        } else {
            panic!("Evaluation produced a NaN\n{}", diag())
        }
    }
}
