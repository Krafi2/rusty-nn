#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

use rusty_nn::activation_functions::Sigmoid;
use rusty_nn::initializer::XavierInit;
use rusty_nn::layers::{DenseBuilder, InputBuilder, LayerType};
use rusty_nn::loss_functions::SquaredError;
use rusty_nn::network::{FeedForward, LinearBuilder};
use rusty_nn::optimizer::{GradientDescent, OptimizerBase};
use rusty_nn::trainer::{DefaultBuilder, Trainer, TrainerBuilder, TrainingConfig};

#[bench]
fn training_speed(b: &mut Bencher) {
    let config = TrainingConfig {
        batch_size: 100,
        epoch_count: 1,
        learning_rate: 0.1,
        weight_decay: 0.1,
    };

    let t_data = vec![[0., 0.]; 100];

    let mut xavier = XavierInit::new();
    let network: FeedForward<LayerType> = LinearBuilder::new()
        .add(InputBuilder::new(1))
        .add(DenseBuilder::<Sigmoid, _>::new(&mut xavier, 100))
        .add(DenseBuilder::<Sigmoid, _>::new(&mut xavier, 100))
        .add(DenseBuilder::<Sigmoid, _>::new(&mut xavier, 1))
        .build()
        .unwrap();

    let optimizer = OptimizerBase::<SquaredError, _, _>::new(network, GradientDescent);
    let mut trainer = DefaultBuilder::new()
        .config(config)
        .training_data(t_data)
        .optimizer(optimizer)
        .loss_handler(|_| {})
        .build();

    b.iter(move || {
        let _ = black_box(0);
        trainer.reset();
        trainer.train().unwrap();
    })
}
