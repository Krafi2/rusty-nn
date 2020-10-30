use rusty_nn::activation_functions::{ActivFunc, Identity};
use rusty_nn::f32s;
use rusty_nn::initializer::IdentityInit;
use rusty_nn::layers::{DenseBuilder, InputBuilder, Layer};
use rusty_nn::loss_functions::SquaredError;
use rusty_nn::network::{feed_forward::FeedForwardNoSer, LinearBuilder, Network};
use rusty_nn::optimizer::{GradientDescent, OptimizerBase, OptimizerManager};
use rusty_nn::trainer::{DefaultBuilder, DefaultTrainer, Trainer, TrainerBuilder, TrainingConfig};

#[derive(Clone)]
struct Sin;
impl ActivFunc for Sin {
    fn evaluate(x: f32) -> f32 {
        x.sin()
    }

    fn derivative(inp: f32, _out: f32) -> f32 {
        inp.cos()
    }
}

fn data() -> Vec<(&'static str, Vec<[f32; 2]>)> {
    let labels = vec![
        (
            "Callisto",
            [-7.5, -8.0, -9.5, -11.0, -13.0, -13.0, -9.5, -9.6],
        ),
        ("Ganymed", [6.2, 4.1, 1.0, 0.0, -3.0, -5.0, -5.9, -6.0]),
        ("Europa", [-3.1, -0.6, 2.3, 4.0, 2.5, 0.0, -3.1, -3.5]),
        ("Io", [2.3, -1.4, -0.9, 2.0, 0.3, -2.5, -0.1, 1.9]),
    ];

    // data in hours from the beginning od the observation
    let data = [0.75, 12.00, 24.75, 36.00, 48.75, 60.00, 72.75, 84.00];

    // now we have to put the data and labels together
    labels
        .into_iter()
        .map(|(name, labels)| {
            (
                name,
                data.clone()
                    .iter()
                    .zip(labels.iter())
                    .map(|(a, b)| [*a, *b])
                    .collect(),
            )
        })
        .collect()
}

fn trainer() -> DefaultTrainer<
    OptimizerBase<SquaredError, GradientDescent, FeedForwardNoSer<Box<dyn Layer>>>,
    [f32; 2],
> {
    let config = TrainingConfig {
        batch_size: 8,
        epoch_count: 10000,
        learning_rate: 0.001,
        weight_decay: 0.,
    };

    let network = LinearBuilder::<Box<dyn Layer>, FeedForwardNoSer<_>>::new()
        .add_boxed(InputBuilder::new(1))
        .add_boxed(DenseBuilder::<Sin, _>::new(IdentityInit, 1, true, true))
        .add_boxed(DenseBuilder::<Identity, _>::new(
            IdentityInit,
            1,
            true,
            false,
        ))
        .build()
        .unwrap();
    network.debug();
    let optimizer = OptimizerBase::<SquaredError, _, _>::new(network, GradientDescent::builder());
    let trainer = DefaultBuilder::<_, [f32; 2]>::new()
        .config(config)
        .training_data(vec![[0., 0.]])
        .optimizer(optimizer)
        .build();

    trainer
}

fn main() -> anyhow::Result<()> {
    let data = data();
    let mut trainer = trainer();

    for (name, data) in data {
        println!("Beginning fitting for {}", name);
        trainer.reset();
        trainer.set_data(data);
        trainer
            .tb_mut()
            .optimizer
            .net_mut()
            .weights_mut()
            .iter_mut()
            .for_each(|w| *w = f32s::splat(0.1));
        let loss = (&mut trainer).last().unwrap()?;
        // for i in &mut trainer {
        //     println!("{}", i?);
        // }
        println!("Finished fitting {} with loss {}", name, loss);
        trainer.debug();
        println!("");
    }

    Ok(())
}
