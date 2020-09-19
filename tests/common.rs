use rusty_nn::activation_functions::{ActivFunc, Sigmoid};
use rusty_nn::initializer::XavierInit;
use rusty_nn::layers::{DenseBuilder, InputBuilder};
use rusty_nn::loss_functions::SquaredError;
use rusty_nn::network::NetworkBuilder;
use rusty_nn::optimizer::GradientDescent;
use rusty_nn::trainer::*;

pub fn t_data() -> Vec<[f32; 2]> {
    const DATA_POINTS: usize = 200;
    (0..DATA_POINTS)
        .map(|x| (x as f32 / DATA_POINTS as f32) * 2. - 1.)
        .map(|x| [x, Sigmoid::evaluate(x)])
        .collect()
}

pub fn basic_trainer() -> DefaultTrainer<GradientDescent<SquaredError>, [f32; 2]> {
    let config = TrainingConfig {
        batch_size: 10,
        epoch_count: 100,
        learning_rate: 0.1,
        weight_decay: 0.,
    };

    let t_data = t_data();

    let network = NetworkBuilder::new()
        .add(InputBuilder::new(1))
        .add(DenseBuilder::<Sigmoid, _>::new(XavierInit::new(), 1))
        .build()
        .unwrap();

    let optimizer = GradientDescent::<SquaredError>::new(network);
    DefaultBuilder::new()
        .optimizer(optimizer)
        .config(config)
        .training_data(t_data)
        .build()
}
