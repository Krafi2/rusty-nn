use super::{BasicLayer, FromArch, GradError, Layer, LayerArch, LayerBuilder, OutShape};
use crate::activation_functions::ActivFunc;
use crate::allocator::{Allocator, GradHdnl, Mediator, WeightHndl};
use crate::f32s;
use crate::helpers::{
    as_scalar, as_scalar_mut, empty_vec_simd, least_size, simd_to_iter, simd_with, splat_n,
};
use crate::initializer::Initializer;

use serde::{Deserialize, Serialize};
use small_table::Table;

/// Layer type where every neuron operates only on a single output of the layer below.
/// Useful when you want to normalize some values
#[derive(Serialize, Deserialize)]
pub struct NormLayer<T: ActivFunc> {
    size: usize,
    actual_size: usize,

    weights: WeightHndl,
    biases: WeightHndl,

    w_gradients: GradHdnl,
    b_gradients: GradHdnl,

    #[serde(skip, default = "empty_vec_simd")]
    weighted_inputs: Vec<f32s>,
    #[serde(skip, default = "empty_vec_simd")]
    activations: Vec<f32s>,
    marker_: std::marker::PhantomData<*const T>,
}

impl<T: ActivFunc> Layer for NormLayer<T> {
    fn rebuild(&mut self) {
        self.weighted_inputs = splat_n(self.actual_size, 0.);
        self.activations = splat_n(self.actual_size, 0.);
    }

    fn eval(&mut self, inputs: &[f32s], med: Mediator<&[f32s], WeightHndl>) {
        for (((inp, w), b), wi) in inputs
            .iter()
            .zip(med.get(&self.weights))
            .zip(med.get(&self.biases))
            .zip(&mut self.weighted_inputs)
        {
            *wi = inp.mul_add(*w, *b);
        }

        for (wi, o) in as_scalar(&self.weighted_inputs)
            .iter()
            .zip(as_scalar_mut(&mut self.activations))
        {
            *o = T::evaluate(*wi);
        }
    }

    fn calculate_derivatives(
        &mut self,
        weights: Mediator<&[f32s], WeightHndl>,
        mut self_deriv: Mediator<&mut [f32s], GradHdnl>,
        inputs: &[f32s],
        in_deriv: &[f32s],
        out_deriv: &mut [f32s],
    ) -> Result<(), GradError> {
        let w_grad = self_deriv.get_mut(&mut self.w_gradients);
        let b_grad = self_deriv.get_mut(&mut self.b_gradients);
        let weights = weights.get(&self.weights);

        assert_eq!(w_grad.len(), self.actual_size);
        assert_eq!(b_grad.len(), self.actual_size);
        assert_eq!(inputs.len(), self.actual_size);
        assert_eq!(self.weighted_inputs.len(), self.actual_size);
        assert_eq!(self.activations.len(), self.actual_size);
        assert!(in_deriv.len() >= self.actual_size);
        assert!(out_deriv.len() >= self.actual_size);

        for (((((((w, wd), bd), inp), wi), a), id), od) in weights
            .iter()
            .zip(w_grad)
            .zip(b_grad)
            .zip(inputs)
            .zip(&self.weighted_inputs)
            .zip(&self.activations)
            .zip(in_deriv)
            .zip(out_deriv)
        {
            let af_deriv = *id
                * simd_with(
                    simd_to_iter(*wi)
                        .zip(simd_to_iter(*a))
                        .map(|(i, o)| T::derivative(*i, *o)),
                );

            *bd += af_deriv; //bias derivative
            *wd += af_deriv * *inp;
            *od = af_deriv * *w;
        }
        Ok(())
    }

    fn debug(&self, med: Mediator<&[f32s], WeightHndl>) -> String {
        Table::new(self.size, 6, 2)
            .line()
            .with_caption("w", as_scalar(&med.get(&self.weights)))
            .with_caption("b", as_scalar(&med.get(&self.biases)))
            .line()
            .with_caption("a", as_scalar(&self.activations))
            .line()
            .build()
    }

    fn output(&self) -> &[f32s] {
        &self.activations
    }
    fn out_size(&self) -> usize {
        self.size
    }
    fn actual_out(&self) -> usize {
        self.actual_size
    }
    fn in_size(&self) -> usize {
        self.size
    }
    fn out_shape(&self) -> OutShape {
        OutShape {
            dims: vec![self.out_size()],
        }
    }
    fn weight_count(&self) -> usize {
        self.actual_size * 2
    }

    fn ready(&mut self) {}

    fn unready(&mut self) {}
}

impl<T: ActivFunc> NormLayer<T> {
    pub fn new<I: Initializer>(mut init: I, mut alloc: Allocator, size: usize) -> Self {
        let actual_size = least_size(size, f32s::lanes());

        // make sure that any extra weights get initialized with zeroes
        let mut n = 0;
        let w_handles = alloc.allocate_with(actual_size, || {
            let w = if n < size { init.get(size, size) } else { 0. };
            n += 1;
            w
        });
        let b_handles = alloc.allocate(actual_size);

        let mut layer = NormLayer {
            size,
            actual_size,

            weights: w_handles.0,
            biases: b_handles.0,

            w_gradients: w_handles.1,
            b_gradients: b_handles.1,

            weighted_inputs: vec![],
            activations: vec![],
            marker_: std::marker::PhantomData,
        };
        layer.rebuild();
        layer
    }
}

impl<T: ActivFunc> Clone for NormLayer<T> {
    fn clone(&self) -> Self {
        unsafe {
            Self {
                size: self.size.clone(),
                actual_size: self.actual_size.clone(),
                weights: self.weights.clone(),
                biases: self.biases.clone(),
                w_gradients: self.w_gradients.clone(),
                b_gradients: self.b_gradients.clone(),
                weighted_inputs: self.weighted_inputs.clone(),
                activations: self.activations.clone(),
                marker_: self.marker_.clone(),
            }
        }
    }
}

impl<T: ActivFunc> Into<BasicLayer> for NormLayer<T>
where
    BasicLayer: FromArch<T>,
{
    fn into(self) -> BasicLayer {
        <BasicLayer as FromArch<T>>::from(LayerArch::NormLayer(self))
    }
}

pub struct NormBuilder<T: ActivFunc, I: Initializer> {
    init: I,
    marker_: std::marker::PhantomData<*const T>,
}

impl<T: ActivFunc, I: Initializer> NormBuilder<T, I> {
    pub fn new(init: I) -> Self {
        NormBuilder {
            init,
            marker_: std::marker::PhantomData,
        }
    }
}

impl<T: ActivFunc, I: Initializer> LayerBuilder for NormBuilder<T, I> {
    type Output = NormLayer<T>;
    fn connect(self, previous: Option<&dyn Layer>, alloc: Allocator) -> Self::Output {
        let previous = previous
            .expect("A NormLayer cannot be the input layer of a network, use a specialized layer");
        let in_size = previous.out_size();
        NormLayer::new(self.init, alloc, in_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation_functions::Test;
    use crate::helpers::as_scalar;
    use crate::initializer::WeightInit;
    use crate::layers::tests::*;

    fn create_layer() -> (NormLayer<Test>, Vec<f32s>) {
        let mut weights = Vec::new();
        let alloc = Allocator::new(&mut weights);
        let init = WeightInit::new((1..=3).map(|x| x as f32));
        let layer = NormLayer::<Test>::new(init, alloc, 3);

        (layer, weights)
    }

    const INPUTS: [f32s; 1] = [f32s::new(1., 2., 3., 4.)];
    const TOLERANCE: f32 = 0.0001;

    #[test]
    fn norm_eval() {
        let (mut layer, weights) = create_layer();

        layer.eval(&INPUTS, Mediator::new(&weights));
        let output = &as_scalar(&layer.output())[0..3];
        let expected = &[2.0, 8.0, 18.0];

        check(expected, output, TOLERANCE, "output");
    }

    /// Computes the various derivatives to be tested.
    /// [weight_deriv, bias_deriv, out_deriv]
    fn derivs() -> [Vec<f32>; 3] {
        let (mut layer, weights) = create_layer();

        let input = [f32s::new(1., 2., 3., 4.)];
        layer.eval(&input, Mediator::new(&weights));

        let mut deriv = vec![f32s::splat(0.); weights.len()];
        let mut out_deriv = vec![f32s::splat(0.); layer.actual_size];

        layer.ready();
        layer
            .calculate_derivatives(
                Mediator::new(&weights),
                Mediator::new(deriv.as_mut()),
                &INPUTS,
                &[f32s::new(0.1, 0.2, 0.3, 0.4)],
                &mut out_deriv,
            )
            .unwrap();

        let mediator = Mediator::<&mut [_], _>::new(deriv.as_mut());
        let weight_deriv = &as_scalar(&mediator.get(&layer.w_gradients))[0..3];
        let bias_deriv = &as_scalar(&mediator.get(&layer.b_gradients))[0..3];
        let out_deriv = &as_scalar(&out_deriv)[0..3];
        [
            weight_deriv.to_owned(),
            bias_deriv.to_owned(),
            out_deriv.to_owned(),
        ]
    }

    #[test]
    fn norm_backprop_weights() {
        let output = &derivs()[0];
        let expected = [0.2, 0.8, 1.8];
        check(&expected, output, TOLERANCE, "weight derivatives");
    }

    #[test]
    fn norm_backprop_bias() {
        let output = &derivs()[1];
        let expected = [0.2, 0.4, 0.6];
        check(&expected, output, TOLERANCE, "bias derivatives");
    }

    #[test]
    fn norm_backprop_output() {
        let output = &derivs()[2];
        let expected = [0.2, 0.8, 1.8];
        check(&expected, output, TOLERANCE, "output derivatives");
    }
}
