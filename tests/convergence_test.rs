use rusty_nn::{
    a_funcs::{ActivFunc, Identity, Sigmoid},
    initializer::{Ones, Xavier},
    layers::DenseBuilder,
    loss::MeanSquared,
    network::{FeedForward, LinearBuilder},
    optimizer::{AdamBuilder, OptimizerBuilder, OptimizerExt},
    trainer::{Data, LogFile, Trainer},
};

#[test]
fn sigmoid_convergence() {
    let epochs = 20;
    let batch_size = 10;
    let l_rate = 0.1;

    let data = (0..100)
        .map(|x| x as f32 / 10. - 5.)
        .map(|x| Data::new([x], [Sigmoid.evaluate(x)]))
        .collect::<Vec<_>>();

    let network = LinearBuilder::new(1)
        .layer(DenseBuilder::new(Sigmoid, Xavier::new(), 1, true, true))
        // .layer(MapBuilder::new(Identity, Ones))
        // .layer(DenseBuilder::new(Identity, Ones, 1, true, true))
        .build::<FeedForward>()
        .unwrap();

    let optimizer = OptimizerBuilder::new()
        .network(network)
        .loss(MeanSquared)
        .optimizer(AdamBuilder::new().l_rate(l_rate).epsilon(1e-8).build())
        .build()
        .unwrap();

    let mut trainer = Trainer::new()
        .data(data.clone())
        .batch_size(batch_size)
        .optimizer(optimizer)
        .build()
        .unwrap();

    trainer.train(epochs);
    let loss = trainer.test(&data);

    assert!(loss < 1e-7, "Failed to converge, loss was {}", loss);
}

#[test]
fn sqrt_convergence() {
    let batch_size = 30;
    let epochs = 500;
    let l_rate = 0.05;

    let data = (0..100)
        .map(|x| x as f32 / 10.)
        .map(|x| Data::new([x], [x.sqrt()]))
        .collect::<Vec<_>>();

    let network = LinearBuilder::new(1)
        .layer(DenseBuilder::new(Sigmoid, Xavier::new(), 5, true, true))
        .layer(DenseBuilder::new(Identity, Ones, 1, true, true))
        .build::<FeedForward>()
        .unwrap();

    let optimizer = OptimizerBuilder::new()
        .network(network)
        .loss(MeanSquared)
        .optimizer(AdamBuilder::new().l_rate(l_rate).build())
        .build()
        .unwrap();

    let mut trainer = Trainer::new()
        .data(data.clone())
        .batch_size(batch_size)
        .optimizer(optimizer)
        .build()
        .unwrap();

    trainer.train(epochs);
    let loss = trainer.test(&data);

    assert!(loss < 1e-3, "Failed to converge, loss was {}", loss);
}
