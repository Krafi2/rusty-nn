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
/// You can create a trainer instance using the [`train`](self::Config::train) method which facilitates training.
#[derive(Clone, Debug)]
pub struct Config {
    pub batch_size: usize,
    pub epochs: usize,
}

impl Config {
    /// Constructs a new instance
    pub fn new(batch_size: usize, epochs: usize) -> Self {
        Self { batch_size, epochs }
    }

    /// This method creates a Stochaistic instance from the provided processor and the
    /// configuration data contained in self. See [`StochaisticTrainer`](self::StochaisticTrainer) for more details.
    pub fn train<T: Processor>(&self, processor: T) -> Stochaistic<T> {
        Stochaistic::new(self.batch_size, self.epochs, processor)
    }
}

/// This struct wraps a reference to a type that implements Processors such that it can be used as a processor itself.
pub struct ProcRef<'a, T: Processor> {
    inner: &'a mut T,
}

impl<'a, T: Processor> ProcRef<'a, T> {
    /// Constructs the wrapper
    pub fn new(ref_: &'a mut T) -> Self {
        Self { inner: ref_ }
    }
}

impl<'a, T: Processor> Processor for ProcRef<'a, T> {
    fn process(&mut self, idx: usize) -> f32 {
        self.inner.process(idx)
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl<'a, T: Processor> Deref for ProcRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl<'a, T: Processor> DerefMut for ProcRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
    }
}

impl<'a, T: Processor> From<&'a mut T> for ProcRef<'a, T> {
    fn from(refer: &'a mut T) -> Self {
        ProcRef { inner: refer }
    }
}

/// This struct drives the provided processor to perform stochaistic training.
/// The processor will be called on batches of data of the configured size every epoch.
/// It will then return the average loss returned by the processor that epoch.
pub struct Stochaistic<T: Processor> {
    // constants
    batch_size: usize,
    batch_count: usize,
    epoch_count: usize,

    // counters
    epoch: usize,

    processor: T,
    idx_gen: IndexShuffler,
}

impl<T: Processor> Stochaistic<T> {
    pub fn new(batch_size: usize, epochs: usize, processor: T) -> Self {
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

impl<O: Optimizer> Stochaistic<GenericProcessor<TwoVecs, O>> {
    /// This is a helper method that constructs a stochaistic trainer from separate data
    /// and label vectors. Returns None if the data vectors don't satisfy the size requirements of the optimizer.
    pub fn from_vecs<D, L, U, V>(
        data: D,
        labels: L,
        epochs: usize,
        batch_size: usize,
        optimizer: O,
    ) -> Option<Self>
    where
        D: IntoIterator<Item = U>,
        L: IntoIterator<Item = V>,
        U: AsRef<[f32]>,
        V: AsRef<[f32]>,
        D::IntoIter: ExactSizeIterator,
        L::IntoIter: ExactSizeIterator,
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
        let stochaistic = Config::new(batch_size, epochs);
        Some(stochaistic.train(processor))
    }

    /// This is a helper method that constructs a stochaistic trainer from a vector of tuples
    /// where each tuple contains a data and label pair in this order. Returns None if the data
    /// vectors don't satisfy the size requirements of the optimizer.
    pub fn from_tuples<D, U, V>(
        data: D,
        epochs: usize,
        batch_size: usize,
        optimizer: O,
    ) -> Option<Self>
    where
        D: IntoIterator<Item = (U, V)>,
        U: AsRef<[f32]>,
        V: AsRef<[f32]>,
        D::IntoIter: ExactSizeIterator,
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
        let stochaistic = Config::new(batch_size, epochs);
        Some(stochaistic.train(processor))
    }
}

/// The trainer allows you to iterate through the training epochs.
impl<T: Processor> Iterator for Stochaistic<T> {
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

impl<T: Processor> Deref for Stochaistic<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.processor
    }
}

impl<T: Processor> DerefMut for Stochaistic<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.processor
    }
}
