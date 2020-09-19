use crate::allocator::Mediator;
use crate::f32s;
use crate::helpers::{least_size, splat_n, to_scalar_mut, zero_simd};
use crate::layers::{Layer, LayerType};
use crate::loss_functions::LossFunc;
use crate::network::Network;
use crate::trainer::TrainingConfig;

pub trait Optimizer {
    /// Process pairs of input and output values laid out as [inputs, outputs]
    fn process(&mut self, data: &[f32]) -> f32;
    /// Calculate gradient based on only a single target value. `index` is the index of the value to target.
    fn process_partial(&mut self, data: &[f32], index: usize) -> f32;
    /// Call `func` with the network output to calculate the output gradients and loss.
    fn process_with<F: FnMut(&[f32], &mut [f32]) -> f32>(&mut self, inputs: &[f32], func: F)
        -> f32;
    /// Calls `func` on all output values and their indexes. If the function returns Some, the value is used to compute the gradient, otherwise it's set to 0.
    fn process_partial_with<F: FnMut(f32, usize) -> Option<f32>>(
        &mut self,
        inputs: &[f32],
        func: F,
    ) -> f32;
    /// Update model weights based on collected data.
    fn update_model(&mut self, config: &TrainingConfig);
    /// Get input size of the network.
    fn in_size(&self) -> usize;
    /// Get output size of the network.
    fn out_size(&self) -> usize;
    /// Get an immutable reference to the network.
    fn network(&self) -> &Network;
    /// Get a mutable reference to the network.
    fn net_mut(&mut self) -> &mut Network;
}

/// Gradient descent is a very simple optimization method but is often outperformed by more sophisticated algorithms
pub struct GradientDescent<F: LossFunc> {
    //optimizes the assigned network using gradient descent
    network: Network,
    gradients: Vec<f32s>, //the gradients are stored in a single vector in order to prevent memory fragmentation and ease operations with them

    temp_mem1: Vec<f32s>, //temporary memory buffers which are used often enough that we dont want to reallocate them every time
    temp_mem2: Vec<f32s>,
    marker: std::marker::PhantomData<*const F>,
}

impl<F: LossFunc> Optimizer for GradientDescent<F> {
    fn process(&mut self, data: &[f32]) -> f32 {
        default_process::<F>(
            data,
            &mut self.network,
            &mut self.gradients,
            &mut self.temp_mem1,
            &mut self.temp_mem2,
        )
    }

    fn process_partial(&mut self, data: &[f32], index: usize) -> f32 {
        default_process_partial::<F>(
            data,
            index,
            &mut self.network,
            &mut self.gradients,
            &mut self.temp_mem1,
            &mut self.temp_mem2,
        )
    }

    fn process_with<T: FnMut(&[f32], &mut [f32]) -> f32>(
        &mut self,
        inputs: &[f32],
        func: T,
    ) -> f32 {
        default_process_with(
            inputs,
            func,
            &mut self.network,
            &mut self.gradients,
            &mut self.temp_mem1,
            &mut self.temp_mem2,
        )
    }

    fn process_partial_with<T: FnMut(f32, usize) -> Option<f32>>(
        &mut self,
        inputs: &[f32],
        func: T,
    ) -> f32 {
        default_process_partial_with::<F, _>(
            inputs,
            func,
            &mut self.network,
            &mut self.gradients,
            &mut self.temp_mem1,
            &mut self.temp_mem2,
        )
    }

    fn update_model(&mut self, config: &TrainingConfig) {
        //multiplying by k gives us the average of the gradients times the learning rate
        let k = f32s::splat(-config.learning_rate / (config.batch_size as f32));
        let decay = f32s::splat(1. - config.weight_decay);

        let weights = self.network.weights_mut();
        assert_eq!(weights.len(), self.gradients.len());

        for (w, d) in weights.into_iter().zip(&mut self.gradients) {
            *w = w.mul_add(decay, k * *d);
            *d = f32s::splat(0.);
        }
    }

    fn in_size(&self) -> usize {
        self.network.in_size()
    }
    fn out_size(&self) -> usize {
        self.network.out_size()
    }
    fn network(&self) -> &Network {
        &self.network
    }
    fn net_mut(&mut self) -> &mut Network {
        &mut self.network
    }
}

impl<F: LossFunc> GradientDescent<F> {
    pub fn new(network: Network) -> GradientDescent<F> {
        assert!(network.layers().len() > 0);
        let size = least_size(network.weight_count(), f32s::lanes());
        let gradients = vec![f32s::splat(0.); size];

        let largest = network.layers().iter().map(|l| l.get_size()).max().unwrap();

        GradientDescent {
            network,
            gradients,
            temp_mem1: splat_n(largest, 0.),
            temp_mem2: splat_n(largest, 0.),
            marker: std::marker::PhantomData,
        }
    }
}

pub struct Adam<F: LossFunc> {
    network: Network,
    gradients: Vec<f32s>, //the gradients are stored in a single vector in order to prevent memory fragmentation and ease operations with them
    momentum: Vec<f32s>,
    velocity: Vec<f32s>,

    beta1: f32,
    beta2: f32,
    epsilon: f32,

    beta1_pow: f32,
    beta2_pow: f32,

    temp_mem1: Vec<f32s>, //temporary memory buffers which are used often enough that we dont want to reallocate them every time
    temp_mem2: Vec<f32s>,

    marker: std::marker::PhantomData<*const F>,
}
impl<F: LossFunc> Optimizer for Adam<F> {
    fn process(&mut self, data: &[f32]) -> f32 {
        default_process::<F>(
            data,
            &mut self.network,
            &mut self.gradients,
            &mut self.temp_mem1,
            &mut self.temp_mem2,
        )
    }

    fn process_partial(&mut self, data: &[f32], index: usize) -> f32 {
        default_process_partial::<F>(
            data,
            index,
            &mut self.network,
            &mut self.gradients,
            &mut self.temp_mem1,
            &mut self.temp_mem2,
        )
    }

    fn process_with<T: FnMut(&[f32], &mut [f32]) -> f32>(
        &mut self,
        inputs: &[f32],
        func: T,
    ) -> f32 {
        default_process_with(
            inputs,
            func,
            &mut self.network,
            &mut self.gradients,
            &mut self.temp_mem1,
            &mut self.temp_mem2,
        )
    }

    fn process_partial_with<T: FnMut(f32, usize) -> Option<f32>>(
        &mut self,
        inputs: &[f32],
        func: T,
    ) -> f32 {
        default_process_partial_with::<F, _>(
            inputs,
            func,
            &mut self.network,
            &mut self.gradients,
            &mut self.temp_mem1,
            &mut self.temp_mem2,
        )
    }

    fn update_model(&mut self, config: &TrainingConfig) {
        assert_eq!(self.gradients.len(), self.network.weights.len());
        assert_eq!(self.gradients.len(), self.momentum.len());
        assert_eq!(self.gradients.len(), self.velocity.len());

        // compute the average
        for g in &mut self.gradients {
            *g /= config.batch_size as f32
        }
        for (m, g) in self.momentum.iter_mut().zip(&self.gradients) {
            *m = self.beta1 * *m + (1. - self.beta1) * *g;
        }
        for (v, g) in self.momentum.iter_mut().zip(&self.gradients) {
            *v = self.beta2 * *v + (1. - self.beta2) * *g * *g;
        }

        let alpha = f32s::splat(
            -config.learning_rate * f32::sqrt(1. - self.beta2_pow) / (1. - self.beta1_pow),
        );
        let decay = f32s::splat(1. - config.weight_decay);
        let epsilon = f32s::splat(self.epsilon);

        for ((w, m), v) in self
            .network
            .weights
            .iter_mut()
            .zip(&self.momentum)
            .zip(&self.velocity)
        {
            *w = w.mul_add(decay, alpha * *m / (v.sqrt() + epsilon));
        }

        self.gradients.iter_mut().for_each(|g| *g = f32s::splat(0.));

        self.beta1_pow *= self.beta1;
        self.beta2_pow *= self.beta2;
    }

    fn in_size(&self) -> usize {
        self.network.in_size()
    }
    fn out_size(&self) -> usize {
        self.network.out_size()
    }
    fn network(&self) -> &Network {
        &self.network
    }
    fn net_mut(&mut self) -> &mut Network {
        &mut self.network
    }
}

impl<F: LossFunc> Adam<F> {
    pub fn new(network: Network, beta1: f32, beta2: f32, epsilon: f32) -> Adam<F> {
        assert!(network.layers().len() > 0);
        let size = least_size(network.weight_count(), f32s::lanes());
        let gradients = vec![f32s::splat(0.); size];

        let largest = network.layers().iter().map(|l| l.get_size()).max().unwrap();

        Adam {
            network,
            gradients: gradients.clone(),
            momentum: gradients.clone(),
            velocity: gradients,

            beta1,
            beta2,
            epsilon,

            beta1_pow: beta1,
            beta2_pow: beta2,

            temp_mem1: splat_n(largest, 0.),
            temp_mem2: splat_n(largest, 0.),
            marker: std::marker::PhantomData,
        }
    }
    pub fn default(network: Network) -> Adam<F> {
        Self::new(network, 0.9, 0.999, 1e-8)
    }
}

/// Calculates all gradients in a network.
/// Both buffer1 and buffer2 must be large enough to fit the largest of the network's gradient arrays.
/// The values stored in `buffer1` are used as the starting error gradients.
/// If `buffer1` contains garbage values, nothing will panic but you will get garbage values out.
fn calculate_gradients<'a>(
    layers: &'a mut [LayerType],
    weights: &'a [f32s],
    gradients: &'a mut [f32s],
    mut buffer1: &'a mut [f32s],
    mut buffer2: &'a mut [f32s],
) {
    assert_eq!(buffer1.len(), buffer2.len());
    let mut iter = layers.iter_mut().rev().peekable();

    loop {
        if let Some(l) = iter.next() {
            if let Some(prev_l) = iter.peek() {
                let size = least_size(prev_l.get_size(), f32s::lanes());
                assert!(buffer1.len() >= size);
                zero_simd(&mut buffer2[..size]); //clear the space needed for the gradients of the inputs

                //TODO slice the gradient buffers to the correct size
                let inputs = prev_l.get_output();
                l.calculate_derivatives(
                    Mediator::new(weights),
                    Mediator::new(gradients),
                    inputs,
                    buffer1,
                    buffer2,
                );

                //hopefully swap references without unnecesarily copying the data as mem::swap does
                let temp = buffer1;
                buffer1 = buffer2;
                buffer2 = temp;
            } else {
                break;
            }
        }
    }
}

fn default_process<F: LossFunc>(
    data: &[f32],
    network: &mut Network,
    gradients: &mut [f32s],
    buffer1: &mut [f32s],
    buffer2: &mut [f32s],
) -> f32 {
    let input_size = network.in_size();
    let input = &data[..input_size]; //extract the part of data that contains the inputs
    network.eval(input);

    let target = &data[input_size..];
    F::gradients(
        network.output(),
        target,
        &mut to_scalar_mut(buffer1)[..network.out_size()],
    );
    calculate_gradients(
        &mut network.layers,
        &mut network.weights,
        gradients,
        buffer1,
        buffer2,
    );

    F::loss(network.output(), target)
}

fn default_process_partial<F: LossFunc>(
    data: &[f32],
    index: usize,
    network: &mut Network,
    gradients: &mut [f32s],
    buffer1: &mut [f32s],
    buffer2: &mut [f32s],
) -> f32 {
    let input_size = network.in_size();
    let input = &data[..input_size]; //extract the part of data that contains the inputs
    network.eval(input);

    let key = data[input_size];
    let out = network.output()[index];
    let deriv = F::deriv(out, key);

    //actual size of the network output
    let actual_out = least_size(network.out_size(), f32s::lanes());

    //zero enough space for the output layer gradients
    zero_simd(&mut buffer1[..actual_out]);
    //set the partial output gradient
    to_scalar_mut(buffer1)[index] = deriv;
    calculate_gradients(
        &mut network.layers,
        &mut network.weights,
        gradients,
        buffer1,
        buffer2,
    );

    F::eval(key, out)
}

fn default_process_with<F: FnMut(&[f32], &mut [f32]) -> f32>(
    inputs: &[f32],
    mut func: F,
    network: &mut Network,
    gradients: &mut [f32s],
    buffer1: &mut [f32s],
    buffer2: &mut [f32s],
) -> f32 {
    network.eval(inputs);
    let loss = func(
        network.output(),
        to_scalar_mut(&mut buffer1[..network.out_size()]),
    );
    calculate_gradients(
        &mut network.layers,
        &mut network.weights,
        gradients,
        buffer1,
        buffer2,
    );
    loss
}

fn default_process_partial_with<E: LossFunc, F: FnMut(f32, usize) -> Option<f32>>(
    inputs: &[f32],
    mut func: F,
    network: &mut Network,
    gradients: &mut [f32s],
    buffer1: &mut [f32s],
    buffer2: &mut [f32s],
) -> f32 {
    network.eval(inputs);
    let mut grads = 0;
    let mut loss = 0.;
    for (i, (o, g)) in network
        .output()
        .iter()
        .zip(to_scalar_mut(buffer1))
        .enumerate()
    {
        *g = if let Some(target) = func(*o, i) {
            grads += 1;
            loss += E::eval(*o, target);
            E::deriv(*o, target)
        } else {
            0.
        }
    }
    calculate_gradients(
        &mut network.layers,
        &mut network.weights,
        gradients,
        buffer1,
        buffer2,
    );
    if grads != 0 {
        loss / grads as f32
    } else {
        0.
    }
}
