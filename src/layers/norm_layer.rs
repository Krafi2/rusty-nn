use super::{Layer, LayerArch, LayerType, OutShape};
use crate::activation_functions::ActivFunc;
use crate::allocator::{invalid_handle, Allocator, GradHdnl, Mediator, WeightHndl};
use crate::f32s;
use crate::helpers::{
    empty_vec_simd, least_size, simd_to_iter, simd_with, splat_n, to_scalar, to_scalar_mut,
};
use crate::initializer::Initializer;

use serde::{Deserialize, Serialize};
use small_table::Table;

/// Layer type where every neuron operates only on a single output of the layer below.
/// Useful when you want to normalize some values
#[derive(Serialize, Deserialize, Clone)]
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
    fn connect(&mut self, shape: OutShape, init: &mut dyn Initializer, mut aloc: Allocator) {
        self.size = shape.dims.iter().product();
        self.actual_size = least_size(self.size, f32s::lanes());

        let res = aloc.allocate_with(self.actual_size, || init.get(self.size, self.size));
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

        for (wi, o) in to_scalar(&self.weighted_inputs)
            .iter()
            .zip(to_scalar_mut(&mut self.activations))
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
    ) {
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
    }

    fn debug(&self, med: Mediator<&[f32s], WeightHndl>) -> String {
        Table::new(self.size, 6, 2)
            .line()
            .with_caption("w", to_scalar(&med.get(&self.weights)))
            .with_caption("b", to_scalar(&med.get(&self.biases)))
            .line()
            .with_caption("a", to_scalar(&self.activations))
            .line()
            .build()
    }

    fn get_output(&self) -> &[f32s] {
        &self.activations
    }
    fn get_size(&self) -> usize {
        self.size
    }
    fn get_in_size(&self) -> usize {
        self.size
    }
    fn out_shape(&self) -> OutShape {
        OutShape {
            dims: vec![self.get_size()],
        }
    }
    fn get_weight_count(&self) -> usize {
        self.size * 2
    }
}
impl<T: ActivFunc> NormLayer<T> {
    pub fn new() -> Self {
        NormLayer {
            size: 0,
            actual_size: 0,

            weights: invalid_handle(),
            biases: invalid_handle(),

            w_gradients: invalid_handle(),
            b_gradients: invalid_handle(),

            weighted_inputs: empty_vec_simd(),
            activations: empty_vec_simd(),
            marker_: std::marker::PhantomData,
        }
    }
}

impl<T: ActivFunc> Into<LayerType> for NormLayer<T> {
    fn into(self) -> LayerType {
        LayerType::from(LayerArch::NormLayer(self))
    }
}
