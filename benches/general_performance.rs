#![feature(test)]

extern crate test;
use test::Bencher;

use rusty_nn::a_funcs::Sigmoid;
use rusty_nn::initializer::XavierInit;
use rusty_nn::layers::{BasicLayer, DenseBuilder, InputBuilder};
use rusty_nn::loss_funcs::SquaredError;
use rusty_nn::network::{FeedForward, LinearBuilder};
use rusty_nn::optimizer::{GradientDescent, OptimizerBase};
use rusty_nn::trainer::trainer_from_tuples;

#[bench]
fn training_speed(b: &mut Bencher) {
    let batch_size = 100;
    let epoch_count = 1;
    let learning_rate = 0.1;

    let t_data = vec![([0.], [0.]); 100];

    let mut xavier = XavierInit::new();
    let network = LinearBuilder::<BasicLayer, FeedForward<_>>::new()
        .add_layer(InputBuilder::new(1))
        .add_layer(DenseBuilder::<Sigmoid, _>::new(
            &mut xavier,
            100,
            true,
            true,
        ))
        .add_layer(DenseBuilder::<Sigmoid, _>::new(
            &mut xavier,
            100,
            true,
            true,
        ))
        .add_layer(DenseBuilder::<Sigmoid, _>::new(&mut xavier, 1, true, true))
        .build()
        .unwrap();

    let optimizer =
        OptimizerBase::<SquaredError, _, _>::new(network, GradientDescent::builder(learning_rate));
    let mut trainer = trainer_from_tuples(t_data, epoch_count, batch_size, optimizer).unwrap();

    b.iter(move || trainer.do_epoch())
}
