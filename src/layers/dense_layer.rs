use crate::{
    a_funcs::ActivFunc,
    allocator::{
        DualAllocator, GradHndl, GradStorage, Handle, WeightAllocator, WeightHndl, WeightStorage,
    },
    f32s,
    helpers::{as_scalar, as_scalar_mut, simd_to_iter, simd_with, sum, IterMask, VectorAdapter},
    initializer::{Initializer, Xavier},
    layers::{no_value, Aligned, BasicLayer, FromArch, Layer, LayerArch, LayerBuilder, Shape}
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Your run of the mill fully connected (dense) layer
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DenseLayer<F> {
    in_size: usize,
    size: usize,

    weights: Handle,
    biases: Handle,

    update_weights: bool,
    update_biases: bool,

    #[serde(with = "no_value")]
    weighted_inputs: Aligned,
    #[serde(with = "no_value")]
    activations: Aligned,
    #[serde(with = "no_value")]
    temp: Aligned,

    phantom: PhantomData<*const F>,
}

impl<F> Layer for DenseLayer<F>
where
    F: ActivFunc,
{
    fn eval(&mut self, input: &Aligned, weights: &WeightStorage) -> &Aligned {
        let biases = weights.get(self.biases);
        let weights = weights.get(self.weights);

        let in_shape = self.input();
        let out_shape = self.output();

        // assert dominance
        assert_eq!(
            weights.as_vector().len(),
            in_shape.vector() * out_shape.scalar()
        );
        assert_eq!(biases.as_vector().len(), out_shape.vector());
        assert_eq!(self.temp.as_vector().len(), in_shape.vector());
        assert_eq!(self.weighted_inputs.as_vector().len(), out_shape.vector());
        assert_eq!(self.activations.as_vector().len(), out_shape.vector());
        assert_eq!(input.as_vector().len(), in_shape.vector());

        for (weights, weighted_input) in weights
            .as_vector()
            .chunks_exact(in_shape.vector())
            .zip(self.weighted_inputs.as_scalar_mut())
        {
            //TODO test if mul_add is faster
            for ((inp, w), temp) in input
                .as_vector()
                .iter()
                .zip(weights)
                .zip(self.temp.as_vector_mut())
            {
                *temp = *inp * *w;
            }
            *weighted_input = sum(self.temp.as_vector(), in_shape.scalar());
        }

        for (wi, b) in self
            .weighted_inputs
            .as_vector_mut()
            .iter_mut()
            .zip(biases.as_vector())
        {
            *wi += *b;
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
        let mut b_grads = gradients.get_mut(self.biases);
        let weights = weights.get(self.weights);

        let in_shape = self.input();
        let out_shape = self.output();

        // assert dominance
        assert_eq!(weights.as_vector().len(), in_shape.vector() * self.size);
        assert_eq!(self.weighted_inputs.as_vector().len(), out_shape.vector());
        assert_eq!(self.temp.as_vector().len(), in_shape.vector());
        assert_eq!(self.activations.as_vector().len(), out_shape.vector());
        assert_eq!(b_grads.as_vector().len(), out_shape.vector());
        assert_eq!(input.shape().vector(), in_shape.vector());
        assert!(out_shape.vector() <= in_grads.shape().vector());
        assert!(in_shape.vector() <= out_grads.shape().vector());

        // compute activation function derivatives
        for ((temp, inp), out) in self
            .temp
            .as_scalar_mut()
            .iter_mut()
            .zip(self.weighted_inputs.as_scalar())
            .zip(self.activations.as_scalar())
        {
            *temp = F::derivative(*inp, *out);
        }
        for (temp, id) in self
            .temp
            .as_vector_mut()
            .iter_mut()
            .zip(in_grads.as_vector())
        {
            *temp *= *id;
        }

        // compute bias derivatives
        if self.update_biases {
            for (bd, temp) in b_grads
                .as_vector_mut()
                .iter_mut()
                .zip(self.temp.as_vector())
            {
                *bd += *temp;
            }
        }

        let mut w_grads = gradients.get_mut(self.weights);
        assert_eq!(w_grads.as_vector().len(), in_shape.vector() * self.size);

        // compute weight derivative
        if self.update_weights {
            for (wds, temp) in w_grads
                .as_vector_mut()
                .chunks_exact_mut(in_shape.vector())
                .zip(self.temp.as_scalar())
            {
                let af_deriv = f32s::splat(*temp);
                for (wd, inp) in wds.iter_mut().zip(input.as_vector()) {
                    *wd += *inp * af_deriv;
                }
            }
        }

        //compute output derivatives
        for (weights, temp) in weights
            .as_vector()
            .chunks_exact(in_shape.vector())
            .zip(self.temp.as_scalar())
        {
            let af_deriv = f32s::splat(*temp);
            for (od, w) in out_grads.as_vector_mut().iter_mut().zip(weights) {
                *od += *w * af_deriv;
            }
        }
    }

    fn activations(&self) -> &Aligned {
        &self.activations
    }

    fn input(&self) -> Shape {
        Shape::new(self.in_size)
    }

    fn output(&self) -> Shape {
        Shape::new(self.size)
    }
}

impl<F> DenseLayer<F> {
    pub fn new<I>(
        init: I,
        alloc: &mut DualAllocator,
        in_size: usize,
        size: usize,
        update_w: bool,
        update_b: bool,
    ) -> Self
    where
        I: Initializer,
    {
        let in_shape = Shape::new(in_size);
        let out_shape = Shape::new(size);
        let weight_count = in_shape.vector() * out_shape.scalar();

        let init = init.construct(in_size, size);
        let init = IterMask::new(init, in_shape.scalar(), in_shape.vector() * f32s::lanes());

        let weights = alloc.allocate(weight_count, VectorAdapter::new(init));
        let biases = alloc.allocate_zeroed(out_shape.vector());

        Self {
            in_size,
            size,
            weights,
            biases,
            update_weights: update_w,
            update_biases: update_b,
            weighted_inputs: Aligned::zeroed(out_shape.scalar()),
            activations: Aligned::zeroed(out_shape.scalar()),
            temp: Aligned::zeroed(in_shape.scalar()),
            phantom: PhantomData,
        }
    }
}

pub struct DenseBuilder<F, I = Xavier> {
    init: I,
    size: usize,
    update_w: bool,
    update_b: bool,
    phantom: PhantomData<*const F>,
}

impl<F, I> DenseBuilder<F, I> {
    pub fn new(init: I, size: usize, update_w: bool, update_b: bool) -> Self {
        DenseBuilder {
            init,
            size,
            update_w,
            update_b,
            phantom: PhantomData,
        }
    }
}

impl<F, I> LayerBuilder for DenseBuilder<F, I>
where
    I: Initializer,
{
    type Output = DenseLayer<F>;

    fn connect(self, input: Shape, alloc: &mut DualAllocator) -> Self::Output {
        DenseLayer::new(
            self.init,
            alloc,
            input.scalar(),
            self.size,
            self.update_w,
            self.update_b,
        )
    }
}

impl<T: ActivFunc> Into<BasicLayer> for DenseLayer<T>
where
    BasicLayer: FromArch<T>,
{
    fn into(self) -> BasicLayer {
        <BasicLayer as FromArch<T>>::from(LayerArch::DenseLayer(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        a_funcs::Test,
        allocator::{Allocator, DualAllocator, GradAllocator},
        helpers::{as_scalar, VectorAdapter},
        layers::tests::check,
    };

    fn create_layer() -> (DenseLayer<Test>, WeightStorage, GradAllocator) {
        let mut alloc = DualAllocator::new();
        let init = (1..=12).map(|x| x as f32);
        let layer = DenseLayer::<Test>::new(init, &mut alloc, 4, 3, true, true);

        let (weights, grads) = alloc.finish();
        (layer, weights, grads)
    }

    const INPUTS: [f32; 4] = [1., 2., 3., 4.];
    const TOLERANCE: f32 = 0.0001;

    #[test]
    fn dense_eval() {
        let (mut layer, weights, _) = create_layer();

        let input = Aligned::from_scalar(&INPUTS);
        let output = layer.eval(&input, &weights);
        let expected = &[60., 140., 220.];

        // dbg!(&layer);
        // dbg!(&weights);

        check(expected, output.as_scalar(), TOLERANCE, "output");
    }

    /// Computes the various derivatives to be tested.
    /// The derivatives are returned in this order [weight_deriv, bias_deriv, out_deriv]
    fn derivs() -> [Vec<f32>; 3] {
        let (mut layer, weights, grads) = create_layer();
        let mut grads = grads.finish();

        let input = Aligned::from_scalar(&INPUTS);
        let in_grads = Aligned::from_scalar(&[0.1, 0.2, 0.3, 0.4]);
        let mut out_grads = Aligned::zeroed(4);

        layer.eval(&input, &weights);
        layer.calc_gradients(&input, &weights, &mut grads, &in_grads, &mut out_grads);

        [
            grads.get(layer.weights).as_scalar()[..12].to_owned(),
            grads.get(layer.biases).as_scalar()[..3].to_owned(),
            out_grads.into_scalar().to_vec(),
        ]
    }

    #[test]
    fn dense_backprop_weights() {
        let output = &derivs()[0];
        let expected = [0.2, 0.4, 0.6, 0.8, 0.4, 0.8, 1.2, 1.6, 0.6, 1.2, 1.8, 2.4];
        check(&expected, output, TOLERANCE, "weight derivatives");
    }

    #[test]
    fn dense_backprop_bias() {
        let output = &derivs()[1];
        let expected = [0.2, 0.4, 0.6];
        check(&expected, output, TOLERANCE, "bias derivatives");
    }

    #[test]
    fn dense_backprop_output() {
        let output = &derivs()[2];
        let expected = [7.6, 8.8, 10., 11.2];
        check(&expected, output, TOLERANCE, "output derivatives");
    }
}
