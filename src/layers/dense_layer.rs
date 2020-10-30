use super::{BasicLayer, FromArch, Layer, LayerArch, LayerBuilder, OutShape};
use crate::activation_functions::ActivFunc;
use crate::allocator::{Allocator, GradHdnl, Mediator, WeightHndl};
use crate::f32s;
use crate::helpers::{as_scalar, as_scalar_mut, empty_vec_simd, least_size, splat_n, sum};
use crate::initializer::Initializer;

use serde::{Deserialize, Serialize};
use small_table::Table;

#[derive(Serialize, Deserialize, Clone)]
///Your run of the mill fully connected (or dense) layer
pub struct DenseLayer<T: ActivFunc> {
    in_size: usize,
    actual_in: usize, //input size rounded up to the nearest simd type

    size: usize,
    actual_size: usize,

    weights: WeightHndl,
    biases: WeightHndl, //size of actual_size

    w_gradients: GradHdnl, //handle to weight gradients
    b_gradients: GradHdnl, //handle to bias gradients

    update_weights: bool,
    update_biases: bool,

    #[serde(skip, default = "empty_vec_simd")]
    weighted_inputs: Vec<f32s>, //size of actual_size
    #[serde(skip, default = "empty_vec_simd")]
    activations: Vec<f32s>, //size of actual_size
    #[serde(skip, default = "empty_vec_simd")]
    temp: Vec<f32s>, //size of actual_in
    marker_: std::marker::PhantomData<*const T>,
}

impl<T: ActivFunc> Layer for DenseLayer<T> {
    fn rebuild(&mut self) {
        self.weighted_inputs = splat_n(self.actual_size, 0.);
        self.activations = splat_n(self.actual_size, 0.);
        self.temp = splat_n(self.actual_in, 0.);
    }

    fn eval(&mut self, inputs: &[f32s], med: Mediator<&[f32s], WeightHndl>) {
        let weights = med.get(&self.weights);
        let biases = med.get(&self.biases);

        // assert dominance
        assert_eq!(weights.len(), self.actual_in * self.size);
        assert_eq!(biases.len(), self.actual_size);
        assert_eq!(self.temp.len(), self.actual_in);
        assert_eq!(self.weighted_inputs.len(), self.actual_size);
        assert_eq!(self.activations.len(), self.actual_size);
        assert_eq!(inputs.len(), self.actual_in);

        for (weights, weighted_input) in weights
            .chunks_exact(self.actual_in)
            .zip(as_scalar_mut(&mut self.weighted_inputs))
        {
            //TODO test if mul_add is faster
            for ((inp, w), temp) in inputs.iter().zip(weights).zip(&mut self.temp) {
                *temp = *inp * *w;
            }
            *weighted_input = sum(&self.temp, self.in_size);
        }

        for (wi, b) in self.weighted_inputs.iter_mut().zip(biases) {
            *wi += *b;
        }

        for (wi, o) in as_scalar(&self.weighted_inputs)
            .iter()
            .zip(as_scalar_mut(&mut self.activations))
        {
            *o = T::evaluate(*wi)
        }
    }

    fn calculate_derivatives(
        &mut self,
        weights: Mediator<&[f32s], WeightHndl>,
        mut self_deriv: Mediator<&mut [f32s], GradHdnl>,
        inputs: &[f32s],
        in_deriv: &[f32s],
        out_deriv: &mut [f32s],
    ) -> Result<(), ()> {
        let mut b_grad = self_deriv.get_mut(&mut self.b_gradients);
        let mut w_grad = self_deriv.get_mut(&mut self.w_gradients);
        let weights = weights.get(&self.weights);

        // assert dominance
        assert_eq!(weights.len(), self.actual_in * self.size);
        assert_eq!(self.weighted_inputs.len(), self.actual_size);
        assert_eq!(self.temp.len(), self.actual_in);
        assert_eq!(self.activations.len(), self.actual_size);
        assert_eq!(b_grad.len(), self.actual_size);
        assert_eq!(w_grad.len(), self.actual_in * self.size);
        assert_eq!(inputs.len(), self.actual_in);
        assert!(self.actual_size <= in_deriv.len());
        assert!(self.actual_in <= out_deriv.len());

        // compute activation function derivatives
        for ((temp, inp), out) in as_scalar_mut(&mut self.temp)
            .iter_mut()
            .zip(as_scalar(&self.weighted_inputs))
            .zip(as_scalar(&self.activations))
        {
            *temp = T::derivative(*inp, *out);
        }
        for (temp, id) in self.temp.iter_mut().zip(in_deriv) {
            *temp *= *id;
        }

        // compute bias derivatives
        if self.update_biases {
            for (bd, temp) in b_grad.iter_mut().zip(&self.temp) {
                *bd += *temp;
            }
        }

        // compute weight derivative
        if self.update_weights {
            for (wds, temp) in w_grad
                .chunks_exact_mut(self.actual_in)
                .zip(as_scalar(&self.temp))
            {
                let af_deriv = f32s::splat(*temp);
                for (wd, inp) in wds.iter_mut().zip(inputs) {
                    *wd += *inp * af_deriv;
                }
            }
        }

        //compute output derivatives
        for (weights, temp) in weights
            .chunks_exact(self.actual_in)
            .zip(as_scalar(&self.temp))
        {
            let af_deriv = f32s::splat(*temp);
            for (od, w) in out_deriv.iter_mut().zip(weights) {
                *od += *w * af_deriv;
            }
        }
        Ok(())
    }

    fn debug(&self, med: Mediator<&[f32s], WeightHndl>) -> String {
        let s = Table::new(self.in_size + 1, 6, 2)
            .line()
            .row(
                (0..self.in_size)
                    .map(|i| usize::to_string(&i))
                    .chain(std::iter::once("b".to_string())),
            )
            .rows(
                med.get(&self.weights)
                    .chunks(self.actual_in)
                    .zip(as_scalar(&med.get(&self.biases)))
                    .map(|(chunk, bias)| {
                        as_scalar(chunk)
                            .iter()
                            .take(self.in_size)
                            .chain(std::iter::once(bias))
                    }),
            )
            .line()
            .with_caption("a", as_scalar(&self.activations))
            .line()
            .build();
        s
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
        self.in_size
    }
    fn out_shape(&self) -> OutShape {
        OutShape {
            dims: vec![self.out_size()],
        }
    }
    fn weight_count(&self) -> usize {
        self.actual_in * self.size + self.actual_size
    }

    fn ready(&mut self) {}

    fn unready(&mut self) {}
}

impl<T: ActivFunc> DenseLayer<T> {
    pub fn new<I: Initializer>(
        mut init: I,
        mut alloc: Allocator,
        in_size: usize,
        size: usize,
        update_w: bool,
        update_b: bool,
    ) -> DenseLayer<T> {
        let actual_in = least_size(in_size, f32s::lanes());
        let actual_size = least_size(size, f32s::lanes());
        // weight count
        let wc = actual_in * size;

        // make sure that any extra weights get initialized with zeroes
        let mut n = 0;
        let w_handles = alloc.allocate_with(wc, || {
            let w = if n < in_size {
                init.get(in_size, size)
            } else {
                0.
            };
            n += 1;
            if n == actual_in * f32s::lanes() {
                n = 0;
            }
            w
        });

        let b_handles = alloc.allocate(actual_size);

        let mut layer = DenseLayer {
            in_size,
            actual_in,
            size,
            actual_size,
            weights: w_handles.0,
            biases: b_handles.0,
            w_gradients: w_handles.1,
            b_gradients: b_handles.1,
            update_weights: update_w,
            update_biases: update_b,
            weighted_inputs: vec![],
            activations: vec![],
            temp: vec![],
            marker_: std::marker::PhantomData,
        };
        layer.rebuild();
        layer
    }
}

pub struct DenseBuilder<T: ActivFunc, I: Initializer> {
    init: I,
    size: usize,
    update_w: bool,
    update_b: bool,
    marker_: std::marker::PhantomData<*const T>,
}

impl<T: ActivFunc, I: Initializer> DenseBuilder<T, I> {
    pub fn new(init: I, size: usize, update_w: bool, update_b: bool) -> Self {
        DenseBuilder {
            init,
            size,
            update_w,
            update_b,
            marker_: std::marker::PhantomData,
        }
    }
}

impl<T: ActivFunc, I: Initializer> LayerBuilder for DenseBuilder<T, I> {
    type Output = DenseLayer<T>;

    fn connect(self, previous: Option<&dyn Layer>, alloc: Allocator) -> Self::Output {
        let previous = previous
            .expect("A DenseLayer cannot be the input layer of a network, use a specialized layer");
        let in_size = previous.out_size();
        DenseLayer::new(
            self.init,
            alloc,
            in_size,
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
    use crate::activation_functions::Test;
    use crate::helpers::as_scalar;
    use crate::initializer::WeightInit;
    use crate::layers::tests::*;

    fn create_layer() -> (DenseLayer<Test>, Vec<f32s>) {
        let mut weights = Vec::new();
        let alloc = Allocator::new(&mut weights);
        let init = WeightInit::new((1..=12).map(|x| x as f32));
        let layer = DenseLayer::<Test>::new(init, alloc, 4, 3, true, true);

        (layer, weights)
    }

    const INPUTS: [f32s; 1] = [f32s::new(1., 2., 3., 4.)];
    const TOLERANCE: f32 = 0.0001;

    #[test]
    fn dense_eval() {
        let (mut layer, weights) = create_layer();

        layer.eval(&INPUTS, Mediator::new(&weights));
        let output = &as_scalar(&layer.output())[0..3];
        let expected = &[60., 140., 220.];

        check(expected, output, TOLERANCE, "output");
    }

    /// Computes the various derivatives to be tested.
    /// [weight_deriv, bias_deriv, out_deriv]
    fn derivs() -> [Vec<f32>; 3] {
        let (mut layer, weights) = create_layer();

        let input = [f32s::new(1., 2., 3., 4.)];
        layer.eval(&input, Mediator::new(&weights));

        let mut deriv = vec![f32s::splat(0.); weights.len()];
        let mut out_deriv = vec![f32s::splat(0.); layer.actual_in];

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
        let weight_deriv = as_scalar(&mediator.get(&layer.w_gradients));
        let bias_deriv = &as_scalar(&mediator.get(&layer.b_gradients))[0..3];
        let out_deriv = as_scalar(&out_deriv);
        [
            weight_deriv.to_owned(),
            bias_deriv.to_owned(),
            out_deriv.to_owned(),
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
