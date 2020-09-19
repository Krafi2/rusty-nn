mod common;

use anyhow;

#[test]
fn sigmoid_convergence() -> anyhow::Result<()> {
    let trainer = common::basic_trainer();
    let loss = trainer.last().unwrap()?;
    assert!(loss < 0.0001);
    Ok(())
}
