mod dense_layer;
mod input_layer;
mod norm_layer;

pub use dense_layer::DenseLayer;
pub use input_layer::InputLayer;
pub use norm_layer::NormLayer;

use crate::allocator::{Allocator, GradHdnl, Mediator, WeightHndl};
use crate::f32s;
use crate::initializer::Initializer;

use crate::activation_functions::*;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
pub trait Layer: Clone {
    ///Connect to a layer
    fn connect(&mut self, shape: OutShape, init: &mut dyn Initializer, aloc: Allocator);
    ///Reallocate memory needed for evaluation.
    ///Used after deserialization as this memory doesnt need to be serialized.
    fn rebuild(&mut self);
    /// Evaluate the layer's output.
    fn eval(&mut self, inputs: &[f32s], med: Mediator<&[f32s], WeightHndl>);
    /// Calculates derivatives of the layer's weights. `in_deriv` are the partial derivatives at the end of the next layer
    /// and `out_deriv` are the derivatives at the layer's input.
    fn calculate_derivatives(
        &mut self,
        weights: Mediator<&[f32s], WeightHndl>,
        self_deriv: Mediator<&mut [f32s], GradHdnl>,
        inputs: &[f32s],
        in_deriv: &[f32s],
        out_deriv: &mut [f32s],
    );
    /// Return string containing formated debug information
    fn debug(&self, med: Mediator<&[f32s], WeightHndl>) -> String;

    //getters and setters
    /// Get layer's output
    fn get_output(&self) -> &[f32s];
    /// Get layer's size
    fn get_size(&self) -> usize;
    /// Get layer's input size
    fn get_in_size(&self) -> usize;

    fn get_weight_count(&self) -> usize;

    fn out_shape(&self) -> OutShape;

    /// This function should panic for all non-input layer types
    fn set_activations(&mut self, _activations: &[f32]) {
        panic!("set_activations not implemented for this layer type")
    }
}

#[derive(Clone)]
pub struct OutShape {
    pub dims: Vec<usize>,
}

#[enum_dispatch(Layer)]
#[derive(Serialize, Deserialize, Clone)]
pub enum LayerArch<T: ActivFunc> {
    DenseLayer(DenseLayer<T>),
    InputLayer(InputLayer),
    NormLayer(NormLayer<T>),
}

#[enum_dispatch(Layer)]
#[derive(Serialize, Deserialize, Clone)]
pub enum LayerType {
    Sigmoid(LayerArch<Sigmoid>),
    Identity(LayerArch<Identity>),
    TanH(LayerArch<TanH>),
    SiLU(LayerArch<SiLU>),
    ReLU(LayerArch<ReLU>),
}

// The conversion is implemented like this instead of a trait, because the from trait is incompatible
// with some trait definitions made by the enum_dispatch macro and this seems like the path of least resistance
// as I don't think anyone will ever need to pass LayerArch as a generic argument needing the trait
impl LayerType {
    /// Convert from a generic LayerArch<T> into a LayerType
    fn from<T: ActivFunc>(l_arch: LayerArch<T>) -> Self {
        // Sound as long as each implementation of ActivFunc has the correct Kind
        // Could be potentially made safe by implementing TryInto for LayerArch<T>
        // but that would likely require some macro magic to be sustainable and it shouldn't
        // be hard to uphold the soundness requirements
        unsafe {
            match T::KIND {
                AFunc::Sigmoid => Self::Sigmoid(std::mem::transmute(l_arch)),
                AFunc::Identity => Self::Identity(std::mem::transmute(l_arch)),
                AFunc::TanH => Self::TanH(std::mem::transmute(l_arch)),
                AFunc::SiLU => Self::SiLU(std::mem::transmute(l_arch)),
                AFunc::ReLU => Self::ReLU(std::mem::transmute(l_arch)),
            }
        }
    }
}