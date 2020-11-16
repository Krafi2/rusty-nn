use crate::helpers::IndexShuffler;
use crate::optimizer::Optimizer;

use std::ops::{Deref, DerefMut};

mod processor;
use processor::{GenericProcessor, TwoVecs};

/// This trait must be implemented by all objects which want to make use of the [StochasticTrainer](self::StochaisticTrainer).
/// The object is first asked about the size of the data it contains. It will then be provided randomly chosen
/// indexes in the data to perform training on. Additionally it will be notified of the start and end of every batch and epoch with the current
/// batch or epoch number respectively. Every epoch contains one or more batches.
pub trait Processor {
    fn process(&mut self, idx: usize) -> f32;
    fn size(&self) -> usize;
    fn begin_batch(&mut self, _batch: usize) {}
    fn end_batch(&mut self, _batch: usize) {}
    fn begin_epoch(&mut self, _epoch: usize) {}
    fn end_epoch(&mut self, _epoch: usize) {}
}

/// This struct contains the configuration information for stochaistic training.
/// You can create a trainer instance using the [`train`](self::Stochaistic::train) method which facilitates training.
#[derive(Clone)]
pub struct Stochaistic {
    batch_size: usize,
    epochs: usize,
}

impl Stochaistic {
    /// Constructs a new instance
    pub fn new(batch_size: usize, epochs: usize) -> Self {
        Self { batch_size, epochs }
    }

    /// This method creates a StochasticTrainer instance from the provided processor and the
    /// configuration data contained in self. See [`StochaisticTrainer`](self::StochaisticTrainer) for more details.
    pub fn train<T: Processor>(&self, processor: T) -> StochaisticTrainer<T> {
        StochaisticTrainer::new(self.batch_size, self.epochs, processor)
    }
}

/// This struct drives the provided processor to perform stochaistic training.
/// The processor will be called on batches of data of the configured size every epoch.
/// It will then return the average loss returned by the processor that epoch.
pub struct StochaisticTrainer<T: Processor> {
    // constants
    batch_size: usize,
    batch_count: usize,
    epoch_count: usize,

    // counters
    epoch: usize,

    processor: T,
    idx_gen: IndexShuffler,
}

impl<T: Processor> StochaisticTrainer<T> {
    fn new(batch_size: usize, epochs: usize, processor: T) -> Self {
        Self {
            batch_size,
            batch_count: processor.size() / batch_size,
            epoch_count: epochs,
            epoch: 0,
            idx_gen: IndexShuffler::new(processor.size()),
            processor,
        }
    }

    /// Process a batch of data, returns the accumnulated loss.
    /// Although this function is public, you should probably make use of the provided Iterator implementation
    /// as it provides nicer interface.
    pub fn do_batch(&mut self, batch: usize) -> f32 {
        self.processor.begin_batch(batch);
        let mut acc = 0.;
        for idx in (&mut self.idx_gen).take(self.batch_size) {
            acc += self.processor.process(idx)
        }
        self.processor.end_batch(batch);
        acc
    }

    /// Process all of the data in batches, returns the average loss.
    /// Although this function is public, you should probably make use of the provided Iterator implementation
    /// as it provides nicer interface.
    pub fn do_epoch(&mut self) -> f32 {
        self.processor.begin_epoch(self.epoch);
        let mut acc = 0.;
        for batch in 0..self.batch_count {
            acc += self.do_batch(batch)
        }
        self.processor.end_epoch(self.epoch);
        acc / (self.batch_size * self.batch_count) as f32
    }
}

/// The trainer allows you to iterate through the training epochs.
impl<T: Processor> Iterator for StochaisticTrainer<T> {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.epoch < self.epoch_count {
            let loss = self.do_epoch();
            self.epoch += 1;
            Some(loss)
        } else {
            None
        }
    }
}

impl<T: Processor> Deref for StochaisticTrainer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.processor
    }
}

impl<T: Processor> DerefMut for StochaisticTrainer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.processor
    }
}


/// This is a helper method that constructs a trainer from the supplied arguments.
/// Returns None if the data vectors arent compatible with the optimizer.
pub fn trainer_from_vecs<D, L, U, V, O>(
    data: D,
    labels: L,
    epochs: usize,
    batch_size: usize,
    optimizer: O,
) -> Option<StochaisticTrainer<GenericProcessor<TwoVecs, O>>>
where
    D: IntoIterator<Item = U>,
    L: IntoIterator<Item = V>,
    U: AsRef<[f32]>,
    V: AsRef<[f32]>,
    D::IntoIter: ExactSizeIterator,
    L::IntoIter: ExactSizeIterator,
    O: Optimizer,
{
    let data = data.into_iter();
    let labels = labels.into_iter();
    assert_eq!(data.len(), labels.len());

    let len = data.len();

    let data_len = optimizer.in_size();
    let mut data_vec = Vec::with_capacity(len * data_len);
    for d in data {
        let d = d.as_ref();
        if d.len() == data_len {
            data_vec.extend(d);
        } else {
            return None;
        }
    }

    let label_len = optimizer.out_size();
    let mut label_vec = Vec::with_capacity(len * label_len);
    for l in labels {
        let l = l.as_ref();
        if l.len() == label_len {
            label_vec.extend(l);
        } else {
            return None;
        }
    }

    let data = TwoVecs::new(
        data_vec.into_boxed_slice(),
        label_vec.into_boxed_slice(),
        len,
        data_len,
        label_len,
    );
    let processor = GenericProcessor::new(data, optimizer);
    let stochaistic = Stochaistic::new(batch_size, epochs);
    Some(stochaistic.train(processor))
}

pub fn trainer_from_tuples<D, U, V, O>(
    data: D,
    epochs: usize,
    batch_size: usize,
    optimizer: O,
) -> Option<StochaisticTrainer<GenericProcessor<TwoVecs, O>>>
where
    D: IntoIterator<Item = (U, V)>,
    U: AsRef<[f32]>,
    V: AsRef<[f32]>,
    D::IntoIter: ExactSizeIterator,
    O: Optimizer,
{
    let data = data.into_iter();
    let len = data.len();

    let data_len = optimizer.in_size();
    let mut data_vec = Vec::with_capacity(len * data_len);

    let label_len = optimizer.out_size();
    let mut label_vec = Vec::with_capacity(len * label_len);

    for (d, l) in data {
        let d = d.as_ref();
        if d.len() == data_len {
            data_vec.extend(d);
        } else {
            return None;
        }

        let l = l.as_ref();
        if l.len() == label_len {
            label_vec.extend(l);
        } else {
            return None;
        }
    }

    let data = TwoVecs::new(data_vec.into(), label_vec.into(), len, data_len, label_len);
    let processor = GenericProcessor::new(data, optimizer);
    let stochaistic = Stochaistic::new(batch_size, epochs);
    Some(stochaistic.train(processor))
}
