use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use super::{Data, Logger};
use crate::optimizer::Optimizer;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;


#[derive(Debug)]
pub struct Stochaistic<O, const IN: usize, const OUT: usize> {
    optimizer: O,
    logger: Box<dyn Logger>,
    batch_size: u32,
    batch_count: u32,
    epoch: u32,
    rng: Pcg64Mcg,
    data: Box<[Data<IN, OUT>]>,
}

impl<O, const IN: usize, const OUT: usize> Stochaistic<O, IN, OUT>
where
    O: Optimizer,
{
    pub fn new<D>(
        optimizer: O,
        data: D,
        logger: Box<dyn Logger>,
        batch_size: u32,
        seed: u64,
    ) -> Self
    where
        D: Into<Box<[Data<IN, OUT>]>>,
    {
        let data = data.into();
        let data_size = data.len();
        assert!(
            data_size >= batch_size as usize,
            "Batch size cannot be larger than data length. batch_size: {}, data_len: {}",
            batch_size,
            data_size
        );
        
        Self {
            optimizer,
            data,
            logger,
            rng: SeedableRng::seed_from_u64(seed),
            batch_count: data_size as u32 / batch_size,
            batch_size,
            epoch: 0,
        }
    }

    pub fn iter(&mut self) -> Iter<'_, O, IN, OUT> {
        Iter { inner: self }
    }

    pub fn train(&mut self, epochs: u32) {
        for _ in 0..epochs {
            self.do_epoch();
        }
    }

    /// Process a batch of data, returns the average loss.
    /// Although this function is public, you should probably make use of the provided Iterator implementation
    /// as it provides nicer interface.
    pub fn do_batch(&mut self, batch: u32) -> f32 {
        let mut accumulator = 0.;
        for _ in 0..self.batch_size {
            let idx = self.rng.gen_range(0, self.data.len());
            let data = &self.data[idx];
            accumulator += self.optimizer.process(&data.input, &data.target);
        }
        self.optimizer.update_model();
        let loss = accumulator / self.batch_size as f32;
        self.logger.batch_loss(self.epoch, batch, loss);
        loss
    }

    /// Process all of the data in batches, returns the average loss.
    /// Although this function is public, you should probably make use of the provided Iterator implementation
    /// as it provides nicer interface.
    pub fn do_epoch(&mut self) -> f32 {
        let mut accumulator = 0.;
        for batch in 0..self.batch_count {
            accumulator += self.do_batch(batch);
        }
        let loss = accumulator / self.batch_count as f32;
        self.logger.epoch_loss(self.epoch, loss);
        self.epoch += 1;
        loss
    }
}

impl<O, const IN: usize, const OUT: usize> Deref for Stochaistic<O, IN, OUT> {
    type Target = O;

    fn deref(&self) -> &Self::Target {
        &self.optimizer
    }
}

impl<O, const IN: usize, const OUT: usize> DerefMut for Stochaistic<O, IN, OUT> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.optimizer
    }
}

impl<'a, O, const IN: usize, const OUT: usize> IntoIterator for &'a mut Stochaistic<O, IN, OUT>
where
    O: Optimizer,
{
    type Item = f32;
    type IntoIter = Iter<'a, O, IN, OUT>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct Iter<'a, O, const IN: usize, const OUT: usize> {
    inner: &'a mut Stochaistic<O, IN, OUT>,
}

impl<'a, O, const IN: usize, const OUT: usize> Iterator for Iter<'a, O, IN, OUT>
where
    O: Optimizer,
{
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.inner.do_epoch())
    }
}
