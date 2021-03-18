#![feature(test)]
extern crate test;

use rusty_nn::{
    a_funcs::Sigmoid,
    initializer::Xavier,
    layers::DenseBuilder,
    loss_funcs::SquaredError,
    network::{FeedForward, LinearBuilder},
    optimizer::{GradientDescent, OptimizerBase},
    trainer::Stochaistic,
};

#[cfg(test)]
mod tests {
    use super::*;
    use test::{Bencher, black_box};

    #[bench]
    fn training_speed(b: &mut Bencher) {
        let batch_size = 100;
        let epoch_count = 1;
        let learning_rate = 0.1;

        let t_data = vec![([0.], [0.]); 100];

        let network = LinearBuilder::new(1)
            .layer(DenseBuilder::<Sigmoid>::new(Xavier::new(), 100, true, true))
            .layer(DenseBuilder::<Sigmoid>::new(Xavier::new(), 100, true, true))
            .layer(DenseBuilder::<Sigmoid>::new(Xavier::new(), 1, true, true))
            .build::<FeedForward>()
            .unwrap();

        let optimizer = OptimizerBase::<SquaredError, _, _>::new(
            network,
            GradientDescent::builder(learning_rate),
        );
        let mut trainer =
            Stochaistic::from_tuples(t_data, epoch_count, batch_size, optimizer).unwrap();

        b.iter(|| black_box(trainer.do_epoch()))
    }
}
