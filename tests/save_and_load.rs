use rusty_nn::a_funcs::{Identity, Sigmoid};
use rusty_nn::helpers::{as_scalar, AsScalarExt};
use rusty_nn::initializer::XavierInit;
use rusty_nn::layers::{BasicLayer, DenseBuilder, InputBuilder, NormBuilder};
use rusty_nn::network::{FeedForward, LinearBuilder, Network};

#[test]
fn save_and_load() {
    let mut init = XavierInit::new();
    let mut network = LinearBuilder::<BasicLayer, FeedForward<_>>::new()
        .add_layer(InputBuilder::new(3))
        .add_layer(DenseBuilder::<Sigmoid, _>::new(&mut init, 5, true, true))
        .add_layer(NormBuilder::<Identity, _>::new(&mut init))
        .build()
        .unwrap();

    let input = [1., 2., 3.];

    network.predict(&input);
    let prediction = as_scalar(network.output());
    let str = serde_json::to_string(&network).expect("Serialization failed");
    let mut network: FeedForward<BasicLayer> =
        serde_json::from_str(&str).expect("Derserialization failed");

    assert!(
        network.predict(&input).as_scalar() == prediction,
        "Network structure damaged during saving."
    );
}
