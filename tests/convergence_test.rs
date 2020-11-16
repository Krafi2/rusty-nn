use rusty_nn::a_funcs::{ActivFunc, Identity, Sigmoid};
use rusty_nn::initializer::XavierInit;
use rusty_nn::layers::{BasicLayer, DenseBuilder, InputBuilder};
use rusty_nn::loss_funcs::SquaredError;
use rusty_nn::network::{FeedForward, LinearBuilder};
use rusty_nn::optimizer::{Adam, GradientDescent, OptimizerBase};
use rusty_nn::trainer::Stochaistic;

use anyhow;

#[test]
fn sigmoid_convergence() -> anyhow::Result<()> {
    let mut init = XavierInit::new();
    let network = LinearBuilder::<BasicLayer, FeedForward<_>>::new()
        .add_layer(InputBuilder::new(1))
        .add_layer(DenseBuilder::<Sigmoid, _>::new(&mut init, 1, true, true))
        .build()
        .unwrap();

    let data = (0..100)
        .map(|x| x as f32 / 50.)
        .map(|x| ([x], [Sigmoid::evaluate(x)]));

    let epochs = 100;
    let batch_size = 10;
    let learning_rate = 0.1;

    let optimizer =
        OptimizerBase::<SquaredError, _, _>::new(network, GradientDescent::builder(learning_rate));

    let trainer = Stochaistic::from_tuples(data, epochs, batch_size, optimizer).unwrap();
    let loss = trainer.last().unwrap();
    assert!(loss < 0.0001);
    Ok(())
}

#[test]
fn sqrt_convergence() -> anyhow::Result<()> {
    let mut init = XavierInit::new();
    let network = LinearBuilder::<BasicLayer, FeedForward<_>>::new()
        .add_layer(InputBuilder::new(1))
        .add_layer(DenseBuilder::<Sigmoid, _>::new(&mut init, 5, true, true))
        .add_layer(DenseBuilder::<Identity, _>::new(&mut init, 1, true, true))
        .build()
        .unwrap();

    let batch_size = 30;
    let epoch_count = 1000;
    let learning_rate = 0.05;

    let data: Vec<(_, _)> = (0..100)
        .map(|x| x as f32 / 10.)
        .map(|x| ([x], [x.sqrt()]))
        .collect();

    let optimizer = OptimizerBase::<SquaredError, _, _>::new(
        network,
        Adam::builder(0.9, 0.999, 0.1, learning_rate),
    );
    let mut trainer = Stochaistic::from_tuples(data, epoch_count, batch_size, optimizer).unwrap();

    let loss = (&mut trainer).last().unwrap();
    assert!(loss < 0.001, "Failed to converge, loss was {}", loss);

    Ok(())
}
