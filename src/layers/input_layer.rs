use super::{Layer, LayerArch, LayerBuilder, LayerType, OutShape};
use crate::activation_functions::Identity;
use crate::allocator::{Allocator, GradHdnl, Mediator, WeightHndl};
use crate::f32s;
use crate::helpers::{empty_vec_simd, least_size, to_scalar, to_scalar_mut};

use serde::{Deserialize, Serialize};
use small_table::Table;

#[derive(Serialize, Deserialize, Clone)]
pub struct InputLayer {
    size: usize,
    actual_size: usize,
    #[serde(skip, default = "empty_vec_simd")]
    activations: Vec<f32s>,
}

impl Layer for InputLayer {
    fn rebuild(&mut self) {
        self.activations = vec![f32s::splat(0.); self.actual_size]
    }

    fn eval(&mut self, _inputs: &[f32s], _med: Mediator<&[f32s], WeightHndl>) {
        panic!("Input layers cannot be evaluated")
    }

    fn calculate_derivatives(
        &mut self,
        _weights: Mediator<&[f32s], WeightHndl>,
        _self_deriv: Mediator<&mut [f32s], GradHdnl>,
        _inputs: &[f32s],
        _in_deriv: &[f32s],
        _out_deriv: &mut [f32s],
    ) {
        panic!("Input layers cannot calculate derivatives")
    }

    fn debug(&self, _med: Mediator<&[f32s], WeightHndl>) -> String {
        Table::new(self.size, 6, 2)
            .line()
            .row(to_scalar(&self.activations))
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
        0
    }
    fn out_shape(&self) -> OutShape {
        OutShape {
            dims: vec![self.get_size()],
        }
    }
    fn get_weight_count(&self) -> usize {
        0
    }

    /// Force set the layer's activations
    fn set_activations(&mut self, activations: &[f32]) {
        //copy the slice, leaving any extra elements as they are so copy_from_slice doesn't panic
        to_scalar_mut(&mut self.activations)[..self.size].copy_from_slice(activations);
    }
}

impl InputLayer {
    pub fn new(size: usize) -> InputLayer {
        let actual_size = least_size(size, f32s::lanes());
        InputLayer {
            size,
            actual_size,
            activations: vec![f32s::splat(0.); actual_size],
        }
    }
}

impl Into<LayerType> for InputLayer {
    fn into(self) -> LayerType {
        LayerType::from(LayerArch::InputLayer::<Identity>(self))
    }
}

pub struct InputBuilder {
    size: usize,
}
impl InputBuilder {
    pub fn new(size: usize) -> Self {
        InputBuilder { size }
    }
}

impl LayerBuilder for InputBuilder {
    type Output = InputLayer;
    fn connect(self, shape: Option<OutShape>, _alloc: Allocator) -> Self::Output {
        if shape.is_some() {
            panic!("There can't be any layers before InputLayer")
        }
        InputLayer::new(self.size)
    }
}
