use super::{Layer, LayerArch, LayerType, OutShape};
use crate::activation_functions::ActivFunc;
use crate::allocator::{invalid_handle, Allocator, GradHdnl, Mediator, WeightHndl};
use crate::f32s;
use crate::helpers::{empty_vec_simd, least_size, splat_n, sum, to_scalar, to_scalar_mut};
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

    #[serde(skip, default = "empty_vec_simd")]
    weighted_inputs: Vec<f32s>, //size of actual_size
    #[serde(skip, default = "empty_vec_simd")]
    activations: Vec<f32s>, //size of actual_size
    #[serde(skip, default = "empty_vec_simd")]
    temp: Vec<f32s>, //size of actual_in
    marker_: std::marker::PhantomData<*const T>,
}

impl<T: ActivFunc> Layer for DenseLayer<T> {
    fn connect(&mut self, shape: OutShape, init: &mut dyn Initializer, mut aloc: Allocator) {
        self.in_size = shape.dims.iter().product();
        self.actual_in = least_size(self.in_size, f32s::lanes());

        let n_w = self.actual_in * self.size;
        let res = aloc.allocate_with(n_w, || init.get(self.in_size, self.size));
        self.weights = res.0;
        self.w_gradients = res.1;

        let res = aloc.allocate(self.actual_size);
        self.biases = res.0;
        self.b_gradients = res.1;

        self.rebuild();
    }

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
            .zip(to_scalar_mut(&mut self.weighted_inputs))
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

        for (wi, o) in to_scalar(&self.weighted_inputs)
            .iter()
            .zip(to_scalar_mut(&mut self.activations))
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
    ) {
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
        for ((temp, inp), out) in to_scalar_mut(&mut self.temp)
            .iter_mut()
            .zip(to_scalar(&self.weighted_inputs))
            .zip(to_scalar(&self.activations))
        {
            *temp = T::derivative(*inp, *out);
        }
        for (temp, id) in self.temp.iter_mut().zip(in_deriv) {
            *temp *= *id;
        }

        // compute bias derivatives
        for (bd, temp) in b_grad.iter_mut().zip(&self.temp) {
            *bd += *temp;
        }

        // compute weight derivative
        for (wds, temp) in w_grad
            .chunks_exact_mut(self.actual_in)
            .zip(to_scalar(&self.temp))
        {
            let af_deriv = f32s::splat(*temp);
            for (wd, inp) in wds.iter_mut().zip(inputs) {
                *wd += *inp * af_deriv;
            }
        }

        //compute output derivatives
        for (weights, temp) in weights
            .chunks_exact(self.actual_in)
            .zip(to_scalar(&self.temp))
        {
            let af_deriv = f32s::splat(*temp);
            for (od, w) in out_deriv.iter_mut().zip(weights) {
                *od += *w * af_deriv;
            }
        }
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
                    .zip(to_scalar(&med.get(&self.biases)))
                    .map(|(chunk, bias)| {
                        to_scalar(chunk)
                            .iter()
                            .take(self.in_size)
                            .chain(std::iter::once(bias))
                    }),
            )
            .line()
            .with_caption("a", to_scalar(&self.activations))
            .line()
            .build();
        s
    }

    fn get_output(&self) -> &[f32s] {
        &self.activations
    }
    fn get_size(&self) -> usize {
        self.size
    }
    fn get_in_size(&self) -> usize {
        self.in_size
    }
    fn out_shape(&self) -> OutShape {
        OutShape {
            dims: vec![self.get_size()],
        }
    }
    fn get_weight_count(&self) -> usize {
        (self.in_size + 1) * self.size
    }
}

impl<T: ActivFunc> DenseLayer<T> {
    pub fn new(size: usize) -> DenseLayer<T> {
        DenseLayer {
            in_size: 0,
            actual_in: 0,
            size,
            actual_size: least_size(size, f32s::lanes()),

            weights: invalid_handle(),
            biases: invalid_handle(),

            w_gradients: invalid_handle(),
            b_gradients: invalid_handle(),

            weighted_inputs: empty_vec_simd(),
            activations: empty_vec_simd(),
            temp: empty_vec_simd(),
            marker_: std::marker::PhantomData,
        }
    }
}

impl<T: ActivFunc> Into<LayerType> for DenseLayer<T> {
    fn into(self) -> LayerType {
        LayerType::from(LayerArch::DenseLayer(self))
    }
}
