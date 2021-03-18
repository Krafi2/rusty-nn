use rusty_nn::{
    a_funcs::{ActivFunc, Sigmoid},
    initializer::Xavier,
    layers::DenseBuilder,
    loss_funcs::SquaredError,
    network::{FeedForward, LinearBuilder},
    optimizer::{Adam, GradientDescent, OptimizerBase},
    trainer::Stochaistic,
};

#[test]
fn sigmoid_convergence() {
    let epochs = 100;
    let batch_size = 10;
    let learning_rate = 0.1;

    let data = (0..100)
        .map(|x| x as f32 / 50.)
        .map(|x| ([x], [Sigmoid::evaluate(x)]));

    let network = LinearBuilder::new(1)
        .layer(DenseBuilder::<Sigmoid>::new(Xavier::new(), 1, true, true))
        .build::<FeedForward>()
        .unwrap();

    let optimizer =
        OptimizerBase::<SquaredError, _, _>::new(network, GradientDescent::builder(learning_rate));
    let trainer = Stochaistic::from_tuples(data, epochs, batch_size, optimizer).unwrap();

    let loss = trainer.last().unwrap();
    assert!(loss < 0.001, "Failed to converge, loss was {}", loss);
}

#[test]
fn sqrt_convergence() {
    let batch_size = 30;
    let epoch_count = 1000;
    let learning_rate = 0.05;

    let data = (0..100).map(|x| x as f32 / 10.).map(|x| ([x], [x.sqrt()]));

    let network = LinearBuilder::new(1)
        .layer(DenseBuilder::<Sigmoid>::new(Xavier::new(), 5, true, true))
        .layer(DenseBuilder::<Sigmoid>::new(Xavier::new(), 1, true, true))
        .build::<FeedForward>()
        .unwrap();

    let optimizer = OptimizerBase::<SquaredError, _, _>::new(
        network,
        Adam::builder(0.9, 0.999, 0.1, learning_rate),
    );

    let trainer = Stochaistic::from_tuples(data, epoch_count, batch_size, optimizer).unwrap();
    let loss = trainer.last().unwrap();
    assert!(loss < 0.001, "Failed to converge, loss was {}", loss);
}
