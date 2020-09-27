mod dense_layer;
mod input_layer;
mod norm_layer;

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
    ) -> Result<(), ()>;

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
        unimplemented!("set_activations not implemented for this layer type")
    }
}

/// Trait all layer builders must implement in order to be added to a NetworkBuilder via the add function.
pub trait LayerBuilder {
    type Output: Layer;
    /// Connect a layer to the previous one. `shape` will be None if there are no layers before.
    fn connect(self, previous: Option<&dyn Layer>, alloc: Allocator) -> Self::Output;
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
impl BasicLayer {
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
