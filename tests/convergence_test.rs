mod common;

use rusty_nn::layers::{Layer, InputBuilder, DenseBuilder, BasicLayer};
use rusty_nn::network::{FeedForward, LinearBuilder, Network};
use rusty_nn::activation_functions::{ActivFunc, AFunc, Sigmoid, Identity, ReLU};
use rusty_nn::initializer::XavierInit;
use rusty_nn::trainer::{TrainingConfig, Trainer};
use rusty_nn::optimizer::OptimizerBase;

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
        .add(DenseBuilder::<Sigmoid, _>::new(&mut init, 15, true, true))
        // .add(DenseBuilder::<Sigmoid, _>::new(&mut init, 5, true, true))
        .add(DenseBuilder::<Sigmoid, _>::new(&mut init, 1, true, true))
        .build()
        .unwrap();

    let config = TrainingConfig {
        batch_size: 30,
        epoch_count: 1000,
        learning_rate: 0.001,
        weight_decay: 0.,
    };
        
    let data = (0..100).map(|x| x as f32 / 10.).map(|x| [x, x.sqrt()]).collect();
    
    let mut trainer = common::trainer_from_nn(network, config, data);

    // let optimizer: OptimizerBase<_, _, _> = trainer.into();
    
    let loss = (&mut trainer).last().unwrap()?;
    println!("Loss {}", loss);
    for x in (0..100).map(|x| x as f32 / 10.) {
        trainer.predict(&[x]);
        println!("{}, {}", x, trainer.output_scalar()[0]);
    }
    trainer.debug();
    assert!(loss < 0.0001);
    
    Ok(())
}
