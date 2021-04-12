pub use adam::{Adam, AdamBuilder};
pub mod adam;

pub use gradient_descent::{GradDescBuilder, GradientDescent};
pub mod gradient_descent;

use crate::{f32s, loss::LossFunc, network::Network, trainer::Data};
use std::ops::{Deref, DerefMut};

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

pub trait OptimizerExt {
    fn test<const IN: usize, const OUT: usize>(&mut self, data: &[Data<IN, OUT>]) -> f32;
}

/// This trait provides interface which must be implemented by optimization
/// algorithms so that they can be used by the OptimizerBase.
pub trait OptimizerAlg {
    /// Modifies the weights based on the gradients such that a minimum can be reached.
    fn update_weights(&mut self, weights: &mut [f32s], gradients: &[f32s]);
}

/// Structs implementing this trait can be constructed into an instance of OptimizerAlg
/// by being provided the length of the data they work on.
pub trait AlgBuilder {
    type Output: OptimizerAlg;
    /// Receives data length and constructs Self::Output
    fn build(self, len: usize) -> Self::Output;
}

pub use base::*;
mod base {
    use super::{adam::AdamConstructor, *};
    use crate::{
        layers::Aligned,
        loss::{Loss, MeanSquared},
        misc::error::BuilderError,
        network::{construction::Construction, FeedForward},
        storage::GradStorage,
    };

    #[derive(Debug)]
    pub struct DefaultOptimizer<F = MeanSquared, N = FeedForward, O = Adam> {
        optimizer: O,
        network: N,
        loss: F,
        grads: GradStorage,
        n: usize,
    }

    impl<F, N, O> DefaultOptimizer<F, N, O>
    where
        N: Network,
    {
        pub fn new<T>(network: Construction<N>, optimizer: T, loss: F) -> Self
        where
            T: AlgBuilder<Output = O>,
        {
            let (network, g_allocator) = network.decompose();
            Self {
                optimizer: optimizer.build(network.weights().len()),
                network,
                loss,
                grads: g_allocator.finish(),
                n: 0,
            }
        }
    }

    impl<F, N, O> Optimizer for DefaultOptimizer<F, N, O>
    where
        O: OptimizerAlg,
        N: Network,
        F: LossFunc,
    {
        /// Process pairs of input and output values
        fn process(&mut self, input: &[f32], target: &[f32]) -> f32 {
            let output = self.network.predict(input);
            let Loss { loss, grads } = self.loss.eval(output.as_scalar(), target);
            let grads = Aligned::from_scalar(&grads);
            self.network.calc_gradients(&mut self.grads, &grads);
            self.n += 1;
            loss
        }

        /// Calculate gradient based on only a single target value. `index` is the index of the value to target.
        fn process_partial(&mut self, input: &[f32], index: usize, target: f32) -> f32 {
            let output = self.network.predict(input).as_scalar()[index];
            let Loss { loss, grads } = self.loss.eval(&[output], &[target]);
            let grad = grads[0];
            let mut grads = Aligned::zeroed(self.out_size());
            grads.as_scalar_mut()[index] = grad;
            self.network.calc_gradients(&mut self.grads, &grads);
            self.n += 1;
            loss
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
                gradients.fill(f32s::splat(0.));
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

    impl<F, N, O> OptimizerExt for DefaultOptimizer<F, N, O>
    where
        O: OptimizerAlg,
        N: Network,
        F: LossFunc,
    {
        fn test<const IN: usize, const OUT: usize>(&mut self, data: &[Data<IN, OUT>]) -> f32 {
            assert_eq!(
                IN,
                self.in_size(),
                "Input size mismatch. Optimizer input size is {}, data is {}",
                self.in_size(),
                IN
            );
            assert_eq!(
                OUT,
                self.out_size(),
                "Output size mismatch. Optimizer output size is {}, data is {}",
                self.in_size(),
                OUT
            );

            let mut loss = 0.;
            for Data { input, target } in data {
                let output = self.network.predict(input);
                loss += self.loss.eval(output.as_scalar(), target).loss;
            }
            loss / data.len() as f32
        }
    }

    impl<F, N, O> Deref for DefaultOptimizer<F, N, O> {
        type Target = N;

        fn deref(&self) -> &Self::Target {
            &self.network
        }
    }

    impl<F, N, O> DerefMut for DefaultOptimizer<F, N, O> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.network
        }
    }

    impl<F, N, O> Clone for DefaultOptimizer<F, N, O>
    where
        F: Clone,
        N: Clone,
        O: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                optimizer: self.optimizer.clone(),
                network: self.network.clone(),
                loss: self.loss.clone(),
                grads: self.grads.clone(),
                n: self.n,
            }
        }
    }

    #[derive(Debug)]
    pub struct OptimizerBuilder<F = MeanSquared, N = FeedForward, O = AdamConstructor> {
        optimizer: Option<O>,
        loss: Option<F>,
        network: Option<Construction<N>>,
    }

    impl<F, N, O> Default for OptimizerBuilder<F, N, O> {
        fn default() -> Self {
            Self {
                optimizer: None,
                loss: None,
                network: None,
            }
        }
    }

    impl<F, N, O> OptimizerBuilder<F, N, O> {
        pub fn new() -> Self {
            Default::default()
        }
    }

    impl<F, N, O> OptimizerBuilder<F, N, O>
    where
        F: Default,
        N: Network,
        O: AlgBuilder + Default,
    {
        pub fn optimizer(mut self, optimizer: O) -> Self {
            self.optimizer = Some(optimizer);
            self
        }
        pub fn network(mut self, network: Construction<N>) -> Self {
            self.network = Some(network);
            self
        }
        pub fn loss(mut self, loss: F) -> Self {
            self.loss = Some(loss);
            self
        }

        pub fn build(self) -> Result<DefaultOptimizer<F, N, O::Output>, BuilderError> {
            let network = self.network.ok_or(BuilderError::new("network"))?;
            let optimizer = self.optimizer.unwrap_or_default();
            let loss = self.loss.unwrap_or_default();
            Ok(DefaultOptimizer::new(network, optimizer, loss))
        }
    }
}
