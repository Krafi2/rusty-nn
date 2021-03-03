use crate::f32s;
use crate::helpers::as_scalar;
use crate::helpers::AsScalarExt;
use crate::loss_funcs::LossFunc;
use crate::network::CalcGradients;

use std::{marker::PhantomData, ops::{Deref, DerefMut}};

pub trait Optimizer {
    /// Process the provided data and label
    fn process(&mut self, input: &[f32], target: &[f32]) -> f32;

    /// Calculate gradient based on only a single target value. `index` is the index of the value to target.
    fn process_partial(&mut self, input: &[f32], index: usize, target: f32) -> f32;

    /// Update model weights based on collected data.
    fn update_model(&mut self);

    /// Get input size of the network.
    fn in_size(&self) -> usize;

    /// Get output size of the network.
    fn out_size(&self) -> usize;
}

#[derive(Debug)]
pub struct OptimizerBase<F: LossFunc, O: OptimizerAlg, N: CalcGradients> {
    optimizer: O,
    network: N,
    marker_: std::marker::PhantomData<*const F>,
    n: usize,
}

impl<O: OptimizerAlg, N: CalcGradients, F: LossFunc> OptimizerBase<F, O, N> {
    pub fn new<B: OptimizerBuilder<Output = O>>(mut network: N, optimizer: B) -> Self {
        network.ready();
        Self {
            optimizer: optimizer.build(network.weights().len()),
            network,
            marker_: std::marker::PhantomData,
            n: 0,
        }
    }
}

impl<O: OptimizerAlg, N: CalcGradients, F: LossFunc> Optimizer for OptimizerBase<F, O, N> {
    /// Process pairs of input and output values laid out as [inputs, outputs]
    fn process(&mut self, input: &[f32], target: &[f32]) -> f32 {
        self.network.predict(input);
        let mut grads = vec![0f32; self.network.out_size()];
        F::gradients(
            &as_scalar(self.network.output())[..self.network.out_size()],
            target,
            &mut grads,
        );

        self.network.calc_gradients(&grads).unwrap();
        self.n += 1;
        let size = self.network.out_size();
        F::loss(&self.network.output().as_scalar()[..size], target)
    }

    /// Calculate gradient based on only a single target value. `index` is the index of the value to target.
    fn process_partial(&mut self, input: &[f32], index: usize, target: f32) -> f32 {
        self.network.predict(input);
        let out = as_scalar(self.network.output())[index];
        let deriv = F::deriv(out, target);

        let mut grads = vec![0f32; self.network.out_size()];
        grads[index] = deriv;

        self.network.calc_gradients(&grads).unwrap();
        self.n += 1;
        F::eval(target, out)
    }

    /// Update model weights based on collected data.
    fn update_model(&mut self) {
        if self.n == 0 {
            eprintln!("Attempted to update the model without processing any gradients.")
        }
        else {
            let (weights, gradients) = self.network.weight_grads_mut().unwrap();

            // We have to average the gradients
            let avg = f32s::splat(1. / self.n as f32);
            for i in gradients.iter_mut() {
                *i *= avg;
            }
            
            self.optimizer.update_weights(weights, gradients);
            self.network.reset_gradients().unwrap();
            
            self.n = 0;
        }
    }

    /// Get input size of the network.
    fn in_size(&self) -> usize {
        self.network.in_size()
    }

    /// Get output size of the network.
    fn out_size(&self) -> usize {
        self.network.out_size()
    }
}

impl<O: OptimizerAlg, N: CalcGradients, F: LossFunc> Deref for OptimizerBase<F, O, N> {
    type Target = N;

    fn deref(&self) -> &Self::Target {
        &self.network
    }
}

impl<O: OptimizerAlg, N: CalcGradients, F: LossFunc> DerefMut for OptimizerBase<F, O, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.network
    }
}

impl<O: OptimizerAlg, N: CalcGradients, F: LossFunc> Clone for OptimizerBase<F, O, N>
where
    O: Clone,
    N: Clone,
{
    fn clone(&self) -> Self {
        Self {
            optimizer: self.optimizer.clone(),
            network: self.network.clone(),
            marker_: PhantomData,
            n: self.n,
            
        }
    }
}

/// Structs implementing this trait can be constructed into an instance of OptimizerAlg
/// by being provided the length of the data they work on.
pub trait OptimizerBuilder {
    type Output: OptimizerAlg;
    /// Receives data length and constructs Self::Output
    fn build(self, len: usize) -> Self::Output;
}

/// This trait provides interface which must be implemented by optimization
/// algorithms so that they can be used by the OptimizerBase.
pub trait OptimizerAlg {
    /// Modifies the weights based on the gradients such that a minimum can be reached.
    fn update_weights(&mut self, weights: &mut [f32s], gradients: &mut [f32s]);
}

/// Gradient descent simply steps the weights based on their derivatives.
#[derive(Clone, Debug)]
pub struct GradientDescent {
    l_rate: f32,
}

impl OptimizerAlg for GradientDescent {
    fn update_weights(&mut self, weights: &mut [f32s], gradients: &mut [f32s]) {
        assert_eq!(weights.len(), gradients.len());
        let k = f32s::splat(-self.l_rate);
        for (w, d) in weights.iter_mut().zip(gradients) {
            *w += k * *d;
        }
    }
}

impl GradientDescent {
    pub fn builder(learning_rate: f32) -> GradDescBuilder {
        GradDescBuilder {
            l_rate: learning_rate,
        }
    }
}

/// Constructor for [GradientDescent](self::GradientDescent)
#[derive(Clone, Debug)]
pub struct GradDescBuilder {
    l_rate: f32,
}

impl OptimizerBuilder for GradDescBuilder {
    type Output = GradientDescent;

    fn build(self, _len: usize) -> Self::Output {
        GradientDescent {
            l_rate: self.l_rate,
        }
    }
}

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
    fn update_weights(&mut self, weights: &mut [f32s], gradients: &mut [f32s]) {
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
    pub fn builder(beta1: f32, beta2: f32, epsilon: f32, l_rate: f32) -> AdamBuilder {
        AdamBuilder {
            beta1,
            beta2,
            epsilon,
            l_rate,
        }
    }
}

/// Constructor for [Adam](self::Adam)
#[derive(Clone, Debug)]
pub struct AdamBuilder {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    l_rate: f32,
}

impl OptimizerBuilder for AdamBuilder {
    type Output = Adam;

    fn build(self, len: usize) -> Self::Output {
        Adam {
            momentum: vec![f32s::splat(0.); len],
            velocity: vec![f32s::splat(0.); len],
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            beta1_pow: self.beta1,
            beta2_pow: self.beta2,
            l_rate: self.l_rate,
        }
    }
}
