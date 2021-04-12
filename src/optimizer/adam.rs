use super::*;

/// The adam optimizer algorithm as shown in the research paper <https://arxiv.org/abs/1412.6980>
#[derive(Clone, Debug)]
pub struct Adam {
    momentum: Vec<f32s>,
    velocity: Vec<f32s>,

    beta1: f32,
    beta2: f32,
    epsilon: f32,

    beta1_pow: f32,
    beta2_pow: f32,

    l_rate: f32,
}

impl OptimizerAlg for Adam {
    fn update_weights(&mut self, weights: &mut [f32s], gradients: &[f32s]) {
        assert_eq!(gradients.len(), weights.len());
        assert_eq!(gradients.len(), self.momentum.len());
        assert_eq!(gradients.len(), self.velocity.len());

        for (m, g) in self.momentum.iter_mut().zip(gradients.iter()) {
            *m = self.beta1 * *m + (1. - self.beta1) * *g;
        }

        for (v, g) in self.velocity.iter_mut().zip(gradients.iter()) {
            *v = self.beta2 * *v + (1. - self.beta2) * *g * *g;
        }

        let alpha =
            f32s::splat(-self.l_rate * f32::sqrt(1. - self.beta2_pow) / (1. - self.beta1_pow));

        let epsilon = f32s::splat(self.epsilon);

        for ((w, m), v) in weights.iter_mut().zip(&self.momentum).zip(&self.velocity) {
            *w += alpha * *m / (v.sqrt() + epsilon);
        }

        self.beta1_pow *= self.beta1;
        self.beta2_pow *= self.beta2;
    }
}

impl Adam {
    pub fn builder() -> AdamBuilder {
        Default::default()
    }

    pub fn new(beta1: f32, beta2: f32, epsilon: f32, l_rate: f32, len: usize) -> Self {
        Self {
            momentum: vec![f32s::splat(0.); len],
            velocity: vec![f32s::splat(0.); len],
            beta1,
            beta2,
            epsilon,
            beta1_pow: beta1,
            beta2_pow: beta2,
            l_rate,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdamBuilder {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    l_rate: f32,
}

impl Default for AdamBuilder {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 0.001,
            l_rate: 0.01,
        }
    }
}

impl AdamBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn l_rate(mut self, l_rate: f32) -> Self {
        self.l_rate = l_rate;
        self
    }

    pub fn build(self) -> AdamConstructor {
        AdamConstructor { builder: self }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AdamConstructor {
    builder: AdamBuilder,
}

impl AlgBuilder for AdamConstructor {
    type Output = Adam;

    fn build(self, len: usize) -> Self::Output {
        let builder = self.builder;
        Adam::new(
            builder.beta1,
            builder.beta2,
            builder.epsilon,
            builder.l_rate,
            len,
        )
    }
}
