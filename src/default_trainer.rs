use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::optimizer::Optimizer;
use crate::trainer::{BuilderBase, Trainer, TrainerBase, TrainerBuilder};

pub struct DefaultTrainer<O: Optimizer, U: AsRef<[f32]>> {
    tb: TrainerBase<O>,
    t_data: Vec<U>, //training data
    rng: SmallRng,
}

impl<O: 'static + Optimizer, U: AsRef<[f32]>> Trainer<O> for DefaultTrainer<O, U> {
    ///get immutable reference to the TrainerBase object
    fn tb(&self) -> &TrainerBase<O> {
        &self.tb
    }
    ///get mutable reference to the TrainerBase object
    fn tb_mut(&mut self) -> &mut TrainerBase<O> {
        &mut self.tb
    }

    ///process a single unit of training data nd return the corresponding loss
    fn process_one(&mut self, idx: usize) -> f32 {
        let index = self.tb.batch * self.tb.batch_size() + idx; //calculate data index
        let data = &self.t_data[index];
        self.tb.optimizer.process(data.as_ref())
    }
    ///this method is called before every epoch so you can do preparations like shuffle the training data for example
    fn prep_epoch(&mut self) {
        self.t_data.shuffle(&mut self.rng);
    }
}

impl<O: 'static + Optimizer, U: AsRef<[f32]>> Iterator for DefaultTrainer<O, U> {
    type Item = anyhow::Result<f32>;
    fn next(&mut self) -> Option<Self::Item> {
        (|| {
            if !self.do_epoch()? {
                Ok(None)
            } else {
                while self.do_batch()? {}
                Ok(Some(self.tb.loss()))
            }
        })()
        .transpose()
    }
}

pub struct DefaultBuilder<O: Optimizer, U: AsRef<[f32]>> {
    base: BuilderBase<O>,

    t_data: Option<Vec<U>>,
}
impl<O: Optimizer, U: AsRef<[f32]>> DefaultBuilder<O, U> {
    pub fn new() -> Self {
        DefaultBuilder {
            base: BuilderBase::new(),
            t_data: None,
        }
    }

    pub fn training_data(mut self, t_data: Vec<U>) -> Self {
        self.t_data = Some(t_data);
        self
    }
    pub fn build(self) -> DefaultTrainer<O, U> {
        let t_data = self
            .t_data
            .expect("Training data must be initialized, use builder method training_data");
        DefaultTrainer {
            tb: self.base.build(t_data.len()),
            t_data,
            rng: SmallRng::seed_from_u64(0),
        }
    }
}

impl<O: Optimizer, U: AsRef<[f32]>> TrainerBuilder<O> for DefaultBuilder<O, U> {
    fn bb(&self) -> &BuilderBase<O> {
        &self.base
    }
    fn bb_mut(&mut self) -> &mut BuilderBase<O> {
        &mut self.base
    }
}
