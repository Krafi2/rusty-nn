use super::Processor;
use crate::optimizer::Optimizer;
use std::ops::{Deref, DerefMut};

pub trait TrainingData {
    /// Get the training data and label at idx respectively.
    fn data(&self, idx: usize) -> (&[f32], &[f32]);
    /// Get the length of training data.
    fn len(&self) -> usize;
}

/// This struct stores training data and labels as two packed vectors of floats.
/// It provides access to this data through the [data](self::<TwoVecs as TrainingData>::data) and [label](self::<TwoVecs as TrainingData>::label) functions.
pub struct TwoVecs {
    data: Box<[f32]>,
    labels: Box<[f32]>,
    len: usize,
    data_len: usize,
    label_len: usize,
}

impl TwoVecs {
    pub fn new(
        data: Box<[f32]>,
        labels: Box<[f32]>,
        len: usize,
        data_len: usize,
        label_len: usize,
    ) -> Self {
        Self {
            data,
            labels,
            len,
            data_len,
            label_len,
        }
    }
}

impl TrainingData for TwoVecs {
    fn data(&self, idx: usize) -> (&[f32], &[f32]) {
        let data_offset = idx * self.data_len;
        let data = &self.data[data_offset..data_offset + self.data_len];
        let label_offset = idx * self.label_len;
        let label = &self.labels[label_offset..label_offset + self.label_len];
        (data, label)
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// This is a struct implementing basic Processor functionality which is generic
/// over the data representation and optimizer.
pub struct GenericProcessor<T: TrainingData, O: Optimizer> {
    data: T,
    optimizer: O,
}

impl<T: TrainingData, O: Optimizer> GenericProcessor<T, O> {
    pub fn new(data: T, optimizer: O) -> Self {
        Self { data, optimizer }
    }
}

impl<T: TrainingData, O: Optimizer> Processor for GenericProcessor<T, O> {
    fn process(&mut self, idx: usize) -> f32 {
        let (data, label) = self.data.data(idx);
        self.optimizer.process(data, label)
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn end_batch(&mut self, _batch: usize) {
        self.optimizer.update_model();
    }
}

impl<T: TrainingData, O: Optimizer> Deref for GenericProcessor<T, O> {
    type Target = O;

    fn deref(&self) -> &Self::Target {
        &self.optimizer
    }
}

impl<T: TrainingData, O: Optimizer> DerefMut for GenericProcessor<T, O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.optimizer
    }
}
