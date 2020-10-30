mod common;

use rusty_nn::activation_functions::{Identity, Sigmoid};
use rusty_nn::initializer::XavierInit;
use rusty_nn::layers::{BasicLayer, DenseBuilder, InputBuilder};
use rusty_nn::loss_functions::SquaredError;
use rusty_nn::network::{FeedForward, LinearBuilder, Network};
use rusty_nn::optimizer::{Adam, OptimizerBase};
use rusty_nn::trainer::{DefaultBuilder, Trainer, TrainerBuilder, TrainingConfig};

use anyhow;

#[test]
fn sigmoid_convergence() -> anyhow::Result<()> {
    let trainer = common::basic_trainer();
    let loss = trainer.last().unwrap()?;
    assert!(loss < 0.0001);
    Ok(())
}

#[test]
fn sqrt_convergence() -> anyhow::Result<()> {
    let mut init = XavierInit::new();
    let network = LinearBuilder::<BasicLayer, FeedForward<_>>::new()
        .add(InputBuilder::new(1))
        .add(DenseBuilder::<Sigmoid, _>::new(&mut init, 5, true, true))
        // .add(DenseBuilder::<Sigmoid, _>::new(&mut init, 5, true, true))
        .add(DenseBuilder::<Identity, _>::new(&mut init, 1, true, true))
        .build()
        .unwrap();

    let config = TrainingConfig {
        batch_size: 30,
        epoch_count: 1000,
        learning_rate: 0.05,
        weight_decay: 0.,
    };

    let data = (0..100)
        .map(|x| x as f32 / 10.)
        .map(|x| [x, x.sqrt()])
        .collect();

    let optimizer =
        OptimizerBase::<SquaredError, _, _>::new(network, Adam::builder(0.9, 0.999, 0.1));
    let mut trainer = DefaultBuilder::new()
        .config(config)
        .training_data(data)
        .optimizer(optimizer)
        .build();

    let loss = (&mut trainer).last().unwrap()?;
    println!("Loss {}", loss);
    for x in (0..100).map(|x| x as f32 / 10.) {
        trainer.predict(&[x]);
        println!("{}, {}", x, trainer.output_scalar()[0]);
    }
    trainer.debug();
    assert!(loss < 0.001, "Failed to converge, loss was {}", loss);

    Ok(())
}
