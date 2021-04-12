#![feature(test)]
extern crate test;

use rusty_nn::{
    a_funcs::Sigmoid,
    initializer::Xavier,
    layers::DenseBuilder,
    loss::MeanSquared,
    network::{FeedForward, LinearBuilder},
    optimizer::GradientDescent
};

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_nn::{
        optimizer::OptimizerBuilder,
        trainer::{Data, Trainer},
    };
    use test::{black_box, Bencher};

    #[bench]
    fn training_speed(b: &mut Bencher) {
        let batch_size = 100;
        let epoch_count = 1;
        let learning_rate = 0.1;

        let t_data = vec![Data::new([0.], [0.]); 100];

        let network = LinearBuilder::new(1)
            .layer(DenseBuilder::new(Sigmoid, Xavier::new(), 100, true, true))
            .layer(DenseBuilder::new(Sigmoid, Xavier::new(), 100, true, true))
            .layer(DenseBuilder::new(Sigmoid, Xavier::new(), 1, true, true))
            .build::<FeedForward>()
            .unwrap();

        let optimizer = OptimizerBuilder::new()
            .network(network)
            .loss(MeanSquared)
            .optimizer(GradientDescent::builder().l_rate(learning_rate).build())
            .build()
            .unwrap();

        let mut trainer = Trainer::new()
            .data(t_data)
            .batch_size(batch_size)
            .optimizer(optimizer)
            .build()
            .unwrap();

        b.iter(|| black_box(trainer.train(epoch_count)))
    }
}
