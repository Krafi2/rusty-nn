use crate::{
    a_funcs::Identity,
    allocator::{GradStorage, WeightAllocator, WeightStorage},
    layers::{
        no_value, Aligned, BasicLayer, Layer, LayerArch, LayerBuilder, LayerGradients, Shape,
    },
};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct InputLayer {
    #[serde(with = "no_value")]
    input: Aligned,
}

impl InputLayer {
    pub fn new(size: usize) -> InputLayer {
        Self {
            input: Aligned::from_scalar(size),
        }
    }
}

impl Layer for InputLayer {
    fn eval(&mut self, input: &Aligned, _weights: &WeightStorage) -> &Aligned {
        assert!(self.input.eq_shape(input));
        self.input.clone_from(input);
        &self.input
    }

    fn input(&self) -> Shape {
        self.input.shape()
    }

    fn output(&self) -> Shape {
        self.input()
    }
}

impl LayerGradients for InputLayer {
    fn calc_gradients(
        &mut self,
        _inputs: &Aligned,
        _weights: &WeightStorage,
        _gradients: &mut GradStorage,
        _in_grads: &Aligned,
        _out_grads: &mut Aligned,
    ) {
    }
}

impl Into<BasicLayer> for InputLayer {
    fn into(self) -> BasicLayer {
        BasicLayer::from(LayerArch::InputLayer::<Identity>(self))
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
    fn connect(self, _previous: Shape, _alloc: WeightAllocator) -> Self::Output {
        InputLayer::new(self.size)
    }
}
