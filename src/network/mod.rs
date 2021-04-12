pub mod construction;
pub mod feed_forward;

pub use self::{construction::LinearBuilder, feed_forward::FeedForward};

use crate::{
    f32s,
    layers::{Aligned, Shape},
    misc::simd::as_scalar_mut,
    storage::GradStorage,
};

/// Trait all neural network architectures must implement
pub trait Network {
    fn predict(&mut self, input: &[f32]) -> &Aligned;

    fn calc_gradients(&mut self, grads: &mut GradStorage, out_grads: &Aligned);

    fn input(&self) -> Shape;

    fn output(&self) -> Shape;

    /// Get network weights.
    fn weights(&self) -> &[f32s];

    /// Get mutable weights.
    fn weights_mut(&mut self) -> &mut [f32s];

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
