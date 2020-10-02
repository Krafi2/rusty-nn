use rusty_nn::activation_functions::{ActivFunc, Sigmoid};
use rusty_nn::initializer::XavierInit;
use rusty_nn::layers::{DenseBuilder, InputBuilder, BasicLayer};
use rusty_nn::loss_functions::SquaredError;
use rusty_nn::network::{FeedForward, LinearBuilder, Network};
use rusty_nn::optimizer::{GradientDescent, OptimizerBase};
use rusty_nn::trainer::*;

pub fn t_data() -> Vec<[f32; 2]> {
    const DATA_POINTS: usize = 200;
    (0..DATA_POINTS)
        .map(|x| (x as f32 / DATA_POINTS as f32) * 2. - 1.)
        .map(|x| [x, Sigmoid::evaluate(x)])
        .collect()
}

pub fn basic_config() -> TrainingConfig {
    TrainingConfig {
        batch_size: 10,
        epoch_count: 100,
        learning_rate: 0.1,
        weight_decay: 0.,
    }
}

pub fn basic_trainer(
) -> DefaultTrainer<OptimizerBase<SquaredError, GradientDescent, FeedForward<BasicLayer>>, [f32; 2]>
{
    let network = LinearBuilder::new()
        .add(InputBuilder::new(1))
        .add(DenseBuilder::<Sigmoid, _>::new(XavierInit::new(), 1, true, true))
        .build()
        .unwrap();

    trainer_from_nn(network, basic_config(), t_data())
}

pub fn trainer_from_nn<T: Network>(
    nn: T,
    config: TrainingConfig,
    data: Vec<[f32; 2]>,
) -> DefaultTrainer<OptimizerBase<SquaredError, GradientDescent, T>, [f32; 2]> {

    let optimizer = OptimizerBase::new(nn, GradientDescent);
    DefaultBuilder::new()
        .optimizer(optimizer)
        .config(config)
        .training_data(data)
        .build()
}
