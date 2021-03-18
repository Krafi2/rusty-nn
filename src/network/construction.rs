use crate::{
    storage::{DualAllocator, GradAllocator},
    layers::{BasicLayer, Layer, LayerBuilder, Shape},
};

/// This trait allows network architectures to be build using the LinearBuilder
pub trait LinearConstruction {
    type Layer;
    type Error;
    type Output;

    fn construct(
        allocator: DualAllocator,
        layers: Vec<Self::Layer>,
    ) -> Result<Self::Output, Self::Error>;
}

/// Builder for networks where all layers have only a single input and output.
pub struct LinearBuilder<L = BasicLayer> {
    allocator: DualAllocator,
    layers: Vec<L>,
    in_shape: Shape,
}

impl<L> LinearBuilder<L>
where
    L: Layer,
{
    pub fn new(in_size: usize) -> Self {
        LinearBuilder {
            allocator: DualAllocator::new(),
            layers: Vec::new(),
            in_shape: Shape::new(in_size),
        }
    }

    fn last_out_shape(&self) -> Shape {
        self.layers.last().map_or(self.in_shape, |l| l.output())
    }

    /// Adds a single layer to the network.
    pub fn layer<T>(mut self, layer: T) -> Self
    where
        T: LayerBuilder,
        T::Output: Into<L>,
    {
        let in_shape = self.last_out_shape();
        let layer = layer.connect(in_shape, &mut self.allocator).into();
        self.layers.push(layer);
        self
    }

    /// Adds the result of calling `func` on the layer produced by the provided layer builder.
    pub fn layer_with<T, F, U>(mut self, layer: T, func: F) -> Self
    where
        T: LayerBuilder,
        F: FnOnce(T::Output) -> U,
        U: Into<L>,
    {
        let in_shape = self.last_out_shape();
        let layer = layer.connect(in_shape, &mut self.allocator);
        self.layers.push(func(layer).into());
        self
    }

    /// Adds all of the layers provided by the `builders` argument.
    pub fn layers<T>(mut self, builders: T) -> Self
    where
        T: IntoIterator,
        T::Item: LayerBuilder,
        <T::Item as LayerBuilder>::Output: Into<L>,
    {
        for builder in builders {
            self = self.layer(builder);
        }
        self
    }

    /// Builds the network. Returns None if no layers had been provided.
    pub fn build<T>(self) -> Result<T::Output, T::Error>
    where
        T: LinearConstruction<Layer = L>,
    {
        if let Some(l) = self.layers.first() {
            assert_eq!(
                l.input(),
                self.in_shape,
                "Layer size mismatch. First layer reports a different size than configured"
            )
        }
        T::construct(self.allocator, self.layers)
    }
}

impl LinearBuilder<Box<dyn Layer>> {
    pub fn boxed_layer<T>(mut self, layer: T) -> Self
    where
        T: LayerBuilder,
        T::Output: Layer + 'static,
    {
        let in_shape = self.last_out_shape();
        let layer = layer.connect(in_shape, &mut self.allocator);
        self.layers.push(Box::new(layer) as Box<dyn Layer>);
        self
    }
}

#[derive(Debug)]
pub struct Construction<N> {
    network: N,
    grads: GradAllocator,
}

impl<N> Construction<N> {
    pub fn new(network: N, grads: GradAllocator) -> Self {
        Self { network, grads }
    }

    pub fn unwrap(self) -> N {
        self.network
    }

    pub fn decompose(self) -> (N, GradAllocator) {
        (self.network, self.grads)
    }
}
