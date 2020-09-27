use super::{Layer, LayerArch, LayerBuilder, LayerType, OutShape};
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
    ) -> Result<(), ()> {
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
        self.actual_size * 2 * f32s::lanes()
    }

    fn ready(&mut self) {}

    fn unready(&mut self) {}
}
impl<T: ActivFunc> NormLayer<T> {
    pub fn new<I: Initializer>(mut init: I, mut alloc: Allocator, size: usize) -> Self {
        let actual_size = least_size(size, f32s::lanes());

        let w_handles = alloc.allocate_with(actual_size, || init.get(size, size));
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

impl<T: ActivFunc> Into<LayerType> for NormLayer<T> {
    fn into(self) -> LayerType {
        LayerType::from(LayerArch::NormLayer(self))
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
        let shape = previous
            .expect("A NormLayer cannot be the input layer of a network, use a specialized layer")
            .out_shape();
        let in_size = shape.dims.iter().product();
        NormLayer::new(self.init, alloc, in_size)
    }
}
