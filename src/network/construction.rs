use super::Network;
use crate::allocator::Allocator;
use crate::f32s;
use crate::layers::{Layer, LayerBuilder};

/// This trait allows network architectures to be build using the LinearBuilder
pub trait LinearConstruction<T> {
    fn construct(weights: Vec<f32s>, layers: Vec<T>) -> Self;
}

/// Builder for networks where all layers have only a single input and output.
pub struct LinearBuilder<L: Layer, O: Network + LinearConstruction<L>> {
    weights: Vec<f32s>,
    layers: Vec<L>,
    marker_: std::marker::PhantomData<*const (L, O)>,
}

impl<L: Layer, O: Network + LinearConstruction<L>> LinearBuilder<L, O> {
    pub fn new() -> Self {
        LinearBuilder {
            weights: Vec::new(),
            layers: Vec::new(),
            marker_: std::marker::PhantomData,
        }
    }

    /// Adds a single layer to the network.
    pub fn add<T>(mut self, layer_builder: T) -> Self
    where
        T: LayerBuilder,
        T::Output: Into<L>,
    {
        let previous = self.layers.last().map(|l| l as &dyn Layer);
        let layer = layer_builder
            .connect(previous, Allocator::new(&mut self.weights))
            .into();
        self.layers.push(layer);
        self
    }
    /// Adds all of the layers provided by the `builders` argument.
    pub fn add_layers<T>(mut self, builders: T) -> Self
    where
        T: IntoIterator,
        T::Item: LayerBuilder,
        <T::Item as LayerBuilder>::Output: Into<L>,
    {
        for builder in builders {
            self = self.add(builder);
        }
        self
    }

    /// Builds the network. Returns None if no layers had been provided.
    pub fn build(self) -> Option<O> {
        if self.layers.is_empty() {
            None
        } else {
            Some(O::construct(self.weights, self.layers))
        }
    }
}
