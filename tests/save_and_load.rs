use rusty_nn::loss_functions::SquaredError;
use rusty_nn::network::Network;
use rusty_nn::optimizer::GradientDescent;
use rusty_nn::trainer::{DefaultBuilder, Trainer, TrainerBuilder, TrainingConfig};

mod common;

#[test]
fn save_and_load() -> anyhow::Result<()> {
    let mut trainer = common::basic_trainer();
    (&mut trainer).for_each(|_| ());

    assert!(trainer.tb().loss() < 0.0001); //assert we have converged

    trainer
        .network()
        .save("temporary_file_to_test_whether_saving_works")?;
    let network = Network::from_file("temporary_file_to_test_whether_saving_works")?;
    std::fs::remove_file("temporary_file_to_test_whether_saving_works")?;

    let config = TrainingConfig {
        batch_size: 10,
        epoch_count: 1,
        learning_rate: 0.,
        weight_decay: 0.,
    };

    let optimizer = GradientDescent::<SquaredError>::new(network);
    let mut trainer = DefaultBuilder::new()
        .optimizer(optimizer)
        .training_data(common::t_data())
        .config(config)
        .build();

    trainer.do_epoch()?; //test training still works
    trainer.do_batch()?;

    assert!(trainer.tb().loss() < 0.0001); //assert nothing got screwed up when saving and loading
    Ok(())
}
