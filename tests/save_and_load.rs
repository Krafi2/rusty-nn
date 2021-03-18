use rusty_nn::{
    a_funcs::{Identity, Sigmoid},
    initializer::{Ones, Xavier},
    layers::{DenseBuilder, MapBuilder},
    network::{FeedForward, LinearBuilder, Network},
};

#[test]
fn save_and_load() -> anyhow::Result<()> {
    let mut network = LinearBuilder::new(3)
        .layer(DenseBuilder::<Sigmoid>::new(Xavier::new(), 5, true, true))
        .layer(MapBuilder::<Identity>::new(Ones))
        .build::<FeedForward>()
        .unwrap()
        .unwrap();

    let ser = serde_json::to_string(&network)?;
    let mut loaded: FeedForward = serde_json::from_str(&ser)?;

    let input = [1., 2., 3.];
    let correct = network.predict(&input).as_scalar();
    let prediction = loaded.predict(&input).as_scalar();

    assert_eq!(
        correct, prediction,
        "Network structure damaged during saving."
    );
    Ok(())
}
