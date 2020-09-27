use super::{Layer, LayerArch, LayerBuilder, BasicLayer, OutShape};
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
        for (bd, temp) in b_grad.iter_mut().zip(&self.temp) {
            *bd += *temp;
        }

        // compute weight derivative
        for (wds, temp) in w_grad
            .chunks_exact_mut(self.actual_in)
            .zip(as_scalar(&self.temp))
        {
            let af_deriv = f32s::splat(*temp);
            for (wd, inp) in wds.iter_mut().zip(inputs) {
                *wd += *inp * af_deriv;
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
    ) -> DenseLayer<T> {
        let actual_in = least_size(in_size, f32s::lanes());
        let actual_size = least_size(size, f32s::lanes());
        // weight count
        let wc = actual_in * size;

        let w_handles = alloc.allocate_with(wc, || init.get(in_size, size));
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
    marker_: std::marker::PhantomData<*const T>,
}

impl<T: ActivFunc, I: Initializer> DenseBuilder<T, I> {
    pub fn new(init: I, size: usize) -> Self {
        DenseBuilder {
            init,
            size,
            marker_: std::marker::PhantomData,
        }
    }
}

impl<T: ActivFunc, I: Initializer> LayerBuilder for DenseBuilder<T, I> {
    type Output = DenseLayer<T>;

    fn connect(self, previous: Option<&dyn Layer>, alloc: Allocator) -> Self::Output {
        let previous = previous
            .expect("A DenseLayer cannot be the input layer of a network, use a specialized layer");
        let in_size = previous.out_shape().dims.iter().product();
        DenseLayer::new(self.init, alloc, in_size, self.size)
    }
}

impl<T: ActivFunc> Into<BasicLayer> for DenseLayer<T> {
    fn into(self) -> BasicLayer {
        BasicLayer::from(LayerArch::DenseLayer(self))
    }
}
