use rusty_nn::layers::BasicLayer;
use rusty_nn::network::{FeedForward, Network};
use rusty_nn::trainer::Trainer;

mod common;

#[test]
fn save_and_load() -> anyhow::Result<()> {
    let mut trainer = common::basic_trainer();
    trainer.train()?;

    assert!(trainer.tb().loss() < 0.0001); //assert we've converged

    trainer
        .network()
        .try_save("temporary_file_to_test_whether_saving_works".as_ref())?;
    let network =
        FeedForward::<BasicLayer>::from_file("temporary_file_to_test_whether_saving_works")?;
    std::fs::remove_file("temporary_file_to_test_whether_saving_works")?;

    let mut trainer = common::trainer_from_nn(network, common::basic_config(), common::t_data());

    trainer.do_epoch()?; //test training still works
    trainer.do_batch()?;

    assert!(trainer.tb().loss() < 0.0001); //assert nothing got screwed up when saving and loading
    Ok(())
}
