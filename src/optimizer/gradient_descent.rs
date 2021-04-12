use super::*;

/// Gradient descent simply steps the weights based on their derivatives.
#[derive(Clone, Debug)]
pub struct GradientDescent {
    l_rate: f32,
}

impl OptimizerAlg for GradientDescent {
    fn update_weights(&mut self, weights: &mut [f32s], gradients: &[f32s]) {
        assert_eq!(weights.len(), gradients.len());
        let k = f32s::splat(-self.l_rate);
        for (w, d) in weights.iter_mut().zip(gradients) {
            *w += k * *d;
        }
    }
}

impl GradientDescent {
    pub fn builder() -> GradDescBuilder {
        Default::default()
    }

    pub fn new(l_rate: f32) -> Self {
        Self { l_rate }
    }
}

/// Constructor for [GradientDescent](self::GradientDescent)
#[derive(Clone, Debug)]
pub struct GradDescBuilder {
    l_rate: f32,
}

impl Default for GradDescBuilder {
    fn default() -> Self {
        Self { l_rate: 0.01 }
    }
}

impl GradDescBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn l_rate(mut self, l_rate: f32) -> Self {
        self.l_rate = l_rate;
        self
    }

    pub fn build(self) -> GradDescConstructor {
        GradDescConstructor { builder: self }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GradDescConstructor {
    builder: GradDescBuilder,
}

impl AlgBuilder for GradDescConstructor {
    type Output = GradientDescent;

    fn build(self, _len: usize) -> Self::Output {
        GradientDescent::new(self.builder.l_rate)
    }
}
