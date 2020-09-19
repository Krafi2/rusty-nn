#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

use rusty_nn::activation_functions::Sigmoid;
use rusty_nn::initializer::XavierInit;
use rusty_nn::layers::{DenseLayer, InputLayer};
use rusty_nn::loss_functions::SquaredError;
use rusty_nn::network::Network;
use rusty_nn::optimizer::GradientDescent;
use rusty_nn::trainer::{DefaultBuilder, Trainer, TrainerBuilder, TrainingConfig};

#[bench]
fn training_speed(b: &mut Bencher) {
    let config = TrainingConfig {
        batch_size: 100,
        epoch_count: 1,
        learning_rate: 0.1,
        weight_decay: 0.1,
    };

    let t_data = vec![[0.,  0.]; 100];

    let mut xavier = XavierInit::new();
    let mut network = Network::new();
    network.add(&mut xavier, InputLayer::new(1));
    network.add(&mut xavier, DenseLayer::<Sigmoid>::new(100));
    network.add(&mut xavier, DenseLayer::<Sigmoid>::new(100));
    network.add(&mut xavier, DenseLayer::<Sigmoid>::new(1));
    let optimizer = GradientDescent::<SquaredError>::new(network);

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
