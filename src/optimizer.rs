use crate::{f32s, helpers::as_scalar, loss_funcs::LossFunc, network::Network};

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

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

/// This trait provides interface which must be implemented by optimization
/// algorithms so that they can be used by the OptimizerBase.
pub trait OptimizerAlg {
    /// Modifies the weights based on the gradients such that a minimum can be reached.
    fn update_weights(&mut self, weights: &mut [f32s], gradients: &[f32s]);
}

pub use base::*;
mod base {
    use super::*;
    use crate::{
        allocator::{Allocator, GradAllocator, GradStorage},
        layers::Aligned,
        loss_funcs::SquaredError,
        network::{construction::Construction, FeedForward},
    };

    #[derive(Debug)]
    pub struct OptimizerBase<F = SquaredError, N = FeedForward, O = Adam> {
        optimizer: O,
        network: N,
        phantom: PhantomData<*const F>,
        grads: GradStorage,
        n: usize,
    }

    impl<F, N, O> OptimizerBase<F, N, O> {
        pub fn new<B>(construction: Construction<N>, optimizer: B) -> Self
        where
            N: Network,
            B: OptimizerBuilder<Output = O>,
        {
            let (network, g_allocator) = construction.decompose();
            Self {
                optimizer: optimizer.build(network.weights().len()),
                network,
                phantom: PhantomData,
                grads: g_allocator.finish(),
                n: 0,
            }
        }
    }

    impl<F, N, O> Optimizer for OptimizerBase<F, N, O>
    where
        O: OptimizerAlg,
        N: Network,
        F: LossFunc,
    {
        /// Process pairs of input and output values laid out as [inputs, outputs]
        fn process(&mut self, input: &[f32], target: &[f32]) -> f32 {
            let mut out_grads = Aligned::zeroed(self.network.output().scalar());
            let output = self.network.predict(input);
            let loss = F::loss(output.as_scalar(), target);

            F::gradients(output.as_scalar(), target, out_grads.as_scalar_mut());

            self.network.calc_gradients(&mut self.grads, &out_grads);
            self.n += 1;
            loss
        }

        /// Calculate gradient based on only a single target value. `index` is the index of the value to target.
        fn process_partial(&mut self, input: &[f32], index: usize, target: f32) -> f32 {
            let output = self.network.predict(input).as_scalar()[index];
            let deriv = F::deriv(output, target);

            let mut out_grads = Aligned::zeroed(self.network.output().scalar());
            out_grads.as_scalar_mut()[index] = deriv;

            self.network.calc_gradients(&mut self.grads, &out_grads);
            self.n += 1;
            F::eval(output, target)
        }

        /// Update model weights based on collected data.
        fn update_model(&mut self) {
            if self.n == 0 {
                eprintln!("Attempted to update the model without processing any gradients.")
            } else {
                let gradients = self.grads.raw_mut();
                let weights = self.network.weights_mut();

                // We have to average the gradients
                let avg = f32s::splat(1. / self.n as f32);
                for i in gradients.iter_mut() {
                    *i *= avg;
                }

                self.optimizer.update_weights(weights, gradients);
                weights.fill(f32s::splat(0.));
                self.n = 0;
            }
        }

        /// Get input size of the network.
        fn in_size(&self) -> usize {
            self.network.input().scalar()
        }

        /// Get output size of the network.
        fn out_size(&self) -> usize {
            self.network.output().scalar()
        }
    }

    impl<F, N, O> Deref for OptimizerBase<F, N, O> {
        type Target = N;

        fn deref(&self) -> &Self::Target {
            &self.network
        }
    }

    impl<F, N, O> DerefMut for OptimizerBase<F, N, O> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.network
        }
    }

    impl<F, N, O> Clone for OptimizerBase<F, N, O>
    where
        O: Clone,
        N: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                optimizer: self.optimizer.clone(),
                network: self.network.clone(),
                phantom: PhantomData,
                grads: self.grads.clone(),
                n: self.n,
            }
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
