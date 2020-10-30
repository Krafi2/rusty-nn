use crate::f32s;
use crate::helpers::as_scalar;
use crate::loss_functions::LossFunc;
use crate::network::Network;
use crate::trainer::TrainingConfig;

pub trait OptimizerManager {
    type Network: Network;
    /// Process pairs of input and output values laid out as [inputs, outputs]
    fn process(&mut self, data: &[f32]) -> f32;

    /// Calculate gradient based on only a single target value. `index` is the index of the value to target.
    fn process_partial(&mut self, data: &[f32], index: usize) -> f32;

    /// Update model weights based on collected data.
    fn update_model(&mut self, config: &TrainingConfig);

    /// Get input size of the network.
    fn in_size(&self) -> usize;

    /// Get output size of the network.
    fn out_size(&self) -> usize;

    /// Get an immutable reference to the network.
    fn net(&self) -> &Self::Network;

    /// Get a mutable reference to the network.
    fn net_mut(&mut self) -> &mut Self::Network;
}

pub struct OptimizerBase<F: LossFunc, O: Optimizer, N: Network> {
    optimizer: O,
    network: N,
    marker_: std::marker::PhantomData<*const F>,
}

impl<O: Optimizer, N: Network, F: LossFunc> OptimizerBase<F, O, N> {
    pub fn new<B: OptimizerBuilder<Output = O>>(mut network: N, optimizer: B) -> Self {
        network.ready();
        Self {
            optimizer: optimizer.build(network.weights().len()),
            network,
            marker_: std::marker::PhantomData,
        }
    }
}

impl<O: Optimizer, N: Network, F: LossFunc> OptimizerManager for OptimizerBase<F, O, N> {
    type Network = N;

    /// Process pairs of input and output values laid out as [inputs, outputs]
    fn process(&mut self, data: &[f32]) -> f32 {
        let input_size = self.network.in_size();
        let input = &data[..input_size]; //extract the part of data that contains the inputs
        self.network.predict(input);

        let target = &data[input_size..];
        let mut grads = vec![0f32; self.network.out_size()];
        F::gradients(
            &as_scalar(self.network.output())[..self.network.out_size()],
            target,
            &mut grads,
        );

        self.network.calc_gradients(&grads).unwrap();

        F::loss(self.network.output_scalar(), target)
    }

    /// Calculate gradient based on only a single target value. `index` is the index of the value to target.
    fn process_partial(&mut self, data: &[f32], index: usize) -> f32 {
        let input_size = self.network.in_size();
        let input = &data[..input_size]; //extract the part of data that contains the inputs
        self.network.predict(input);

        let target = data[input_size];
        let out = as_scalar(self.network.output())[index];
        let deriv = F::deriv(out, target);

        let mut grads = vec![0f32; self.network.out_size()];
        grads[index] = deriv;

        self.network.calc_gradients(&grads).unwrap();

        F::eval(target, out)
    }

    /// Update model weights based on collected data.
    fn update_model(&mut self, config: &TrainingConfig) {
        let (weights, gradients) = self.network.weight_grads_mut().unwrap();
        self.optimizer.update_weights(weights, gradients, config);
        self.network.reset_gradients().unwrap();
    }

    /// Get input size of the network.
    fn in_size(&self) -> usize {
        self.network.in_size()
    }

    /// Get output size of the network.
    fn out_size(&self) -> usize {
        self.network.out_size()
    }

    /// Get an immutable reference to the network.
    fn net(&self) -> &N {
        &self.network
    }

    /// Get a mutable reference to the network.
    fn net_mut(&mut self) -> &mut N {
        &mut self.network
    }
}

pub trait OptimizerBuilder {
    type Output: Optimizer;
    fn build(self, len: usize) -> Self::Output;
}

pub trait Optimizer {
    fn update_weights(
        &mut self,
        weights: &mut [f32s],
        gradients: &mut [f32s],
        config: &TrainingConfig,
    );
}

pub struct GradientDescent;
impl Optimizer for GradientDescent {
    fn update_weights(
        &mut self,
        weights: &mut [f32s],
        gradients: &mut [f32s],
        config: &TrainingConfig,
    ) {
        //multiplying by k gives us the average of the gradients times the learning rate
        let k = f32s::splat(-config.learning_rate / (config.batch_size as f32));
        let decay = f32s::splat(1. - config.weight_decay);

        assert_eq!(weights.len(), gradients.len());

        for (w, d) in weights.iter_mut().zip(gradients) {
            *w = w.mul_add(decay, k * *d);
        }
    }
}

impl GradientDescent {
    pub fn builder() -> GradDescBuilder {
        GradDescBuilder
    }
}

pub struct GradDescBuilder;
impl OptimizerBuilder for GradDescBuilder {
    type Output = GradientDescent;

    fn build(self, _len: usize) -> Self::Output {
        GradientDescent
    }
}

pub struct Adam {
    momentum: Vec<f32s>,
    velocity: Vec<f32s>,

    beta1: f32,
    beta2: f32,
    epsilon: f32,

    beta1_pow: f32,
    beta2_pow: f32,
}

impl Optimizer for Adam {
    fn update_weights(
        &mut self,
        weights: &mut [f32s],
        gradients: &mut [f32s],
        config: &TrainingConfig,
    ) {
        assert_eq!(gradients.len(), weights.len());
        assert_eq!(gradients.len(), self.momentum.len());
        assert_eq!(gradients.len(), self.velocity.len());

        // compute the average
        for g in gradients.iter_mut() {
            *g /= config.batch_size as f32
        }

        for (m, g) in self.momentum.iter_mut().zip(gradients.iter()) {
            *m = self.beta1 * *m + (1. - self.beta1) * *g;
        }

        for (v, g) in self.momentum.iter_mut().zip(gradients.iter()) {
            *v = self.beta2 * *v + (1. - self.beta2) * *g * *g;
        }

        let alpha = f32s::splat(
            -config.learning_rate * f32::sqrt(1. - self.beta2_pow) / (1. - self.beta1_pow),
        );

        let decay = f32s::splat(1. - config.weight_decay);
        let epsilon = f32s::splat(self.epsilon);

        for ((w, m), v) in weights.iter_mut().zip(&self.momentum).zip(&self.velocity) {
            *w = w.mul_add(decay, alpha * *m / (v.sqrt() + epsilon));
        }

        self.beta1_pow *= self.beta1;
        self.beta2_pow *= self.beta2;
    }
}

impl Adam {
    pub fn builder(beta1: f32, beta2: f32, epsilon: f32) -> AdamBuilder {
        AdamBuilder {
            beta1,
            beta2,
            epsilon,
        }
    }
}

pub struct AdamBuilder {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
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
        }
    }
}
