pub mod construction;
pub mod feed_forward;

pub use self::construction::LinearBuilder;
pub use self::feed_forward::FeedForward;

use crate::f32s;
use crate::helpers::as_scalar_mut;
use crate::layers::GradError;

/// Trait all neural network architectures must implement
pub trait Network {
    /// Predict the value corresponding to `input`.
    fn predict(&mut self, input: &[f32]) -> &[f32s];

    /// Get network output.
    fn output(&self) -> &[f32s];

    /// Get network weights.
    fn weights(&self) -> &[f32s];

    /// Get mutable weights.
    fn weights_mut(&mut self) -> &mut [f32s];

    /// Returns input size of the network
    fn in_size(&self) -> usize;

    /// Return output size of the network
    fn out_size(&self) -> usize;

    /// Copies the weights of another instance.
    /// Panics if the instance has a diffferent number of weights.
    fn copy_weights(&mut self, other: &Self) {
        self.weights_mut().copy_from_slice(other.weights());
    }

    /// Copies the provided weights.
    /// Panics if the number of weights provided isn't the same as the instance's weights.
    fn set_weights(&mut self, weights: &[f32]) {
        as_scalar_mut(self.weights_mut()).copy_from_slice(weights);
    }
}

pub trait CalcGradients: Network {
    /// Ready the network for optimization, if already readied do nothing.
    fn ready(&mut self);

    /// Undo the changes made by ready; if the network isn't readied do nothing.
    fn unready(&mut self);

    /// Calculates the weight gradients based on the gradients of the output.
    /// Returns Err if the network isn't readied.
    fn calc_gradients(&mut self, output_gradients: &[f32]) -> Result<(), GradError>;

    // Returns the gradients acumulated by the network so far or Err if the network isn't readied.
    fn gradients(&self) -> Result<&[f32s], GradError>;

    fn gradients_mut(&mut self) -> Result<&mut [f32s], GradError>;

    /// Resets the gradients to zero or returns Err if the network isn't readied.
    fn reset_gradients(&mut self) -> Result<(), GradError>;

    /// Accesor for weights and gradients at the same time so borrow checker doesn't yell at us.
    /// Returns None if the network isn't readied.
    fn weight_grads_mut(&mut self) -> Result<(&mut [f32s], &mut [f32s]), GradError>;
}
