use crate::{
    a_funcs::{ActivFunc, Identity},
    storage::{DualAllocator, GradStorage, Handle, WeightStorage},
    f32s,
    helpers::{simd_to_iter, simd_with, IterMask, VectorAdapter},
    initializer::{Initializer, Ones},
    layers::{no_value, Aligned, BasicLayer, Layer, LayerArch, LayerBuilder, Shape},
};
use serde::{Deserialize, Serialize};
use std::{convert::TryInto, marker::PhantomData};

/// Layer type where every neuron operates only on a single output of the layer below.
/// Useful when you want to normalize some values
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MapLayer<F> {
    weights: Handle,
    biases: Handle,

    #[serde(with = "no_value")]
    weighted_inputs: Aligned,
    #[serde(with = "no_value")]
    activations: Aligned,

    phantom: PhantomData<*const F>,
}

impl<T> MapLayer<T> {
    pub fn new<I>(init: I, alloc: &mut DualAllocator, input: Shape) -> Self
    where
        I: Initializer,
    {
        let init = init.construct(input.scalar(), input.scalar());
        let init = IterMask::new(init, input.scalar(), input.vector() * f32s::lanes());
        let weights = alloc.allocate(input.vector(), VectorAdapter::new(init));
        let biases = alloc.allocate_zeroed(input.vector());

        MapLayer {
            weights,
            biases,
            weighted_inputs: Aligned::zeroed(input.scalar()),
            activations: Aligned::zeroed(input.scalar()),
            phantom: PhantomData,
        }
    }
}

impl<F> Layer for MapLayer<F>
where
    F: ActivFunc,
{
    fn eval(&mut self, input: &Aligned, weights: &WeightStorage) -> &Aligned {
        for (((inp, w), b), wi) in input
            .as_vector()
            .iter()
            .zip(weights.get(self.weights).as_vector())
            .zip(weights.get(self.biases).as_vector())
            .zip(self.weighted_inputs.as_vector_mut())
        {
            *wi = inp.mul_add(*w, *b);
        }

        for (wi, o) in self
            .weighted_inputs
            .as_scalar()
            .iter()
            .zip(self.activations.as_scalar_mut())
        {
            *o = F::evaluate(*wi);
        }

        &self.activations
    }

    fn calc_gradients(
        &mut self,
        input: &Aligned,
        weights: &WeightStorage,
        gradients: &mut GradStorage,
        in_grads: &Aligned,
        out_grads: &mut Aligned,
    ) {
        let handles = gradients
            .get_multiple_mut(&[self.weights, self.biases])
            .expect("Failed to dereference handles");
        let [mut w_grad, mut b_grad] = TryInto::<[_; 2]>::try_into(handles).unwrap();
        // let w_grad = gradients.get_mut(self.weights);
        // let b_grad = gradients.get_mut(self.biases);
        let weights = weights.get(self.weights);

        let shape = self.activations.shape();

        assert_eq!(w_grad.as_vector().len(), shape.vector());
        assert_eq!(b_grad.as_vector().len(), shape.vector());
        assert_eq!(input.as_vector().len(), shape.vector());
        assert_eq!(self.weighted_inputs.as_vector().len(), shape.vector());
        assert_eq!(self.activations.as_vector().len(), shape.vector());
        assert!(in_grads.as_vector().len() >= shape.vector());
        assert!(out_grads.as_vector().len() >= shape.vector());

        for (((((((w, wd), bd), inp), wi), a), id), od) in weights
            .as_vector()
            .iter()
            .zip(w_grad.as_vector_mut())
            .zip(b_grad.as_vector_mut())
            .zip(input.as_vector())
            .zip(self.weighted_inputs.as_vector())
            .zip(self.activations.as_vector())
            .zip(in_grads.as_vector())
            .zip(out_grads.as_vector_mut())
        {
            let af_deriv = *id
                * simd_with(
                    simd_to_iter(*wi)
                        .zip(simd_to_iter(*a))
                        .map(|(i, o)| F::derivative(*i, *o)),
                );

            *bd += af_deriv; //bias derivative
            *wd += af_deriv * *inp;
            *od = af_deriv * *w;
        }
    }

    fn activations(&self) -> &Aligned {
        &self.activations
    }

    fn input(&self) -> Shape {
        self.activations.shape()
    }

    fn output(&self) -> Shape {
        self.input()
    }
}

impl<T> From<MapLayer<T>> for BasicLayer
where 
    T: ActivFunc,
    LayerArch<T>: From<MapLayer<T>>,
    BasicLayer: From<LayerArch<T>>,
{
    fn from(val: MapLayer<T>) -> Self {
        <BasicLayer as From<_>>::from(LayerArch::from(val))
    }
}

pub struct MapBuilder<T = Identity, I = Ones> {
    init: I,
    phantom: PhantomData<*const T>,
}

impl<T, I> MapBuilder<T, I> {
    pub fn new(init: I) -> Self {
        MapBuilder {
            init,
            phantom: PhantomData,
        }
    }
}

impl<T, I> LayerBuilder for MapBuilder<T, I>
where
    I: Initializer,
    T: ActivFunc,
{
    type Output = MapLayer<T>;

    fn connect(self, input: Shape, alloc: &mut DualAllocator) -> Self::Output {
        MapLayer::new(self.init, alloc, input)
    }
}

#[cfg(test)]
mod tests {
    use std::iter::repeat;

    use super::*;
    use crate::{a_funcs::Test, storage::GradAllocator, layers::tests::check};

    fn create_layer() -> (MapLayer<Test>, WeightStorage, GradAllocator) {
        let mut alloc = DualAllocator::new();
        let init = [1., 2., 3.].iter().copied().chain(repeat(0.));
        let layer = MapLayer::<Test>::new(init, &mut alloc, Shape::new(3));

        let (weights, grads) = alloc.finish();
        (layer, weights, grads)
    }

    const INPUTS: [f32; 3] = [1., 2., 3.];
    const TOLERANCE: f32 = 0.0001;

    #[test]
    fn map_eval() {
        let (mut layer, weights, _grads) = create_layer();

        let input = Aligned::from_scalar(&INPUTS);
        let output = layer.eval(&input, &weights);

        let expected = &[2.0, 8.0, 18.0];
        check(expected, output.as_scalar(), TOLERANCE, "output");
    }

    /// Computes the various derivatives to be tested.
    /// The derivatives are returned in this order: [weight_deriv, bias_deriv, out_deriv].
    fn derivs() -> [Vec<f32>; 3] {
        let (mut layer, weights, grads) = create_layer();
        let mut grads = grads.finish();

        let input = Aligned::from_scalar(&INPUTS);
        let in_grads = Aligned::from_scalar(&[0.1, 0.2, 0.3]);
        let mut out_grads = Aligned::zeroed(input.shape().scalar());

        layer.eval(&input, &weights);
        layer.calc_gradients(&input, &weights, &mut grads, &in_grads, &mut out_grads);

        [
            grads.get(layer.weights).as_scalar()[0..3].to_owned(),
            grads.get(layer.biases).as_scalar()[0..3].to_owned(),
            out_grads.into_scalar()[0..3].to_vec(),
        ]
    }

    #[test]
    fn map_backprop_weights() {
        let output = &derivs()[0];
        let expected = [0.2, 0.8, 1.8];
        check(&expected, output, TOLERANCE, "weight derivatives");
    }

    #[test]
    fn map_backprop_bias() {
        let output = &derivs()[1];
        let expected = [0.2, 0.4, 0.6];
        check(&expected, output, TOLERANCE, "bias derivatives");
    }

    #[test]
    fn map_backprop_output() {
        let output = &derivs()[2];
        let expected = [0.2, 0.8, 1.8];
        check(&expected, output, TOLERANCE, "output derivatives");
    }
}
