use serde::{ser::SerializeStruct, Deserialize, Serialize, Serializer};

use anyhow::Context;
use std::fs;

use crate::network::Network;
use crate::optimizer::OptimizerManager;

pub use crate::default_trainer::{DefaultBuilder, DefaultTrainer};

use std::path::PathBuf;

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub epoch_count: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
}

///Snapshot of training progress
#[derive(Deserialize)]
pub struct Snapshot {
    pub config: TrainingConfig,
    pub batch_count: usize,
    pub save_every: u32,
    pub epoch: usize,
}

impl Snapshot {
    pub fn from_file(path: &str) -> anyhow::Result<Snapshot> {
        let s = fs::read_to_string(path)?;
        let snap = serde_json::from_str(&s)?;
        Ok(snap)
    }
}

///This trait implements common methods to assist in creating Trainer structs.
///Your struct should also implement an iterator over f32 values which evaluates epochs and returns the average loss
pub trait Trainer
where
    Self: IntoIterator<Item = anyhow::Result<f32>> + Iterator<Item = anyhow::Result<f32>>,
{
    type Optimizer: OptimizerManager;

    ///get immutable reference to the TrainerBase object
    fn tb(&self) -> &TrainerBase<Self::Optimizer>;
    ///get mutable reference to the TrainerBase object
    fn tb_mut(&mut self) -> &mut TrainerBase<Self::Optimizer>;

    ///process a single unit of training data nd return the corresponding loss
    fn process_one(&mut self, idx: usize) -> f32;
    ///this method is called before every epoch so you can do preparations like shuffle the training data for example
    fn prep_epoch(&mut self);

    ///This method prepares the object for evaluating batches and returns a bool specifying whether epoch iteration should continue.
    ///Suitable for use in a while loop if you want to customize your training loop.
    fn do_epoch(&mut self) -> anyhow::Result<bool> {
        if self.tb().epoch >= self.tb().epoch_count() {
            return Ok(false);
        }

        self.prep_epoch();
        self.tb_mut().loss_accumulator = 0.;
        self.tb_mut().batch = 0;
        Ok(true)
    }

    ///This method evaluates all of the batch's minibatches and returns bool specifying whether batch iteration should continue.
    ///Suitable for use in a while loop if you want to customize your training loop.
    fn do_batch(&mut self) -> anyhow::Result<bool> {
        if self.tb().batch >= self.tb().batch_count {
            self.tb_mut().epoch += 1;

            let loss = self.tb().loss();
            (self.tb_mut().loss_handler)(loss);

            if self.should_save() {
                self.save()?;
            }
            return Ok(false);
        }

        let mut loss_accumulator = 0f32;
        for i in 0..self.tb().batch_size() {
            loss_accumulator += self.process_one(i);
            if self.tb().debug {
                self.debug()
            };
        }
        let loss = loss_accumulator / self.tb().batch_size() as f32;
        self.tb_mut().loss_accumulator += loss;
        self.tb_mut().batch += 1;
        self.tb_mut().update_model();
        Ok(true)
    }

    fn train(&mut self) -> anyhow::Result<()> {
        for r in self.into_iter() {
            r?;
        }
        Ok(())
    }

    fn debug(&self) {
        self.network().debug();
    }

    fn into_snapshot(&self) -> Snapshot {
        Snapshot {
            config: self.tb().config.clone(),
            batch_count: self.tb().batch,
            save_every: self.tb().save_every,
            epoch: self.tb().epoch,
        }
    }

    ///Returns true when the training should be saved
    fn should_save(&mut self) -> bool {
        self.tb().epoch % self.tb().save_every as usize == 0
    }

    fn save_path(&self) -> PathBuf {
        (self.tb().save_handler)(self.tb().epoch as u32)
    }

    ///Saves training progress
    fn save(&self) -> anyhow::Result<()> {
        let path = &self.save_path(); //get file name from save_handler
        (|| -> anyhow::Result<()> {
            fs::create_dir_all(path)?;
            fs::write(path.join("config.json"), serde_json::to_string(self.tb())?)?;
            self.tb()
                .optimizer
                .net()
                .try_save(&path.join("model.json"))?;
            Ok(())
        })()
        .with_context(|| {
            format!(
                "Failed to save training progress into a snapshot, filename is {}",
                path.to_string_lossy()
            )
        })
    }

    ///Returns immutable reference to the Network object stored inside
    fn network(&self) -> &<Self::Optimizer as OptimizerManager>::Network {
        self.tb().optimizer.net()
    }

    ///Returns mutable reference to the Network object stored inside
    fn net_mut(&mut self) -> &mut <Self::Optimizer as OptimizerManager>::Network {
        self.tb_mut().optimizer.net_mut()
    }

    ///Reset training
    fn reset(&mut self) {
        self.tb_mut().epoch = 0;
        self.tb_mut().batch = 0;
        self.tb_mut().loss_accumulator = 0.;
    }
}

pub struct TrainerBase<O: OptimizerManager> {
    //constants
    pub config: TrainingConfig,
    pub batch_count: usize,
    pub save_every: u32, //number of epochs between saves
    // keep_n: usize, //number of snapshots to keep
    pub debug: bool, //should run debug after every network evaluation

    //counters
    pub epoch: usize,
    pub batch: usize,
    pub loss_accumulator: f32,

    pub optimizer: O,
    pub loss_handler: Box<dyn FnMut(f32)>, //called on every loss value produced by training
    pub save_handler: Box<dyn Fn(u32) -> PathBuf>, //called every time a network is saved to provide the file name
}
impl<O: OptimizerManager> TrainerBase<O> {
    //getters for ease of use

    pub fn batch_size(&self) -> usize {
        self.config.batch_size
    }
    pub fn epoch_count(&self) -> usize {
        self.config.epoch_count
    }
    pub fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }
    pub fn weight_decay(&self) -> f32 {
        self.config.weight_decay
    }
    pub fn loss(&self) -> f32 {
        self.loss_accumulator / self.batch as f32
    }

    pub fn update_model(&mut self) {
        //special function because rust has ownership issues when we do this through references
        self.optimizer.update_model(&self.config)
    }

    pub fn change_data_len(&mut self, new_len: usize) {
        self.batch_count = new_len / self.batch_size();
    }
}
impl<O: OptimizerManager> Serialize for TrainerBase<O> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut st = serializer.serialize_struct("TrainerBase", 5)?;
        st.serialize_field("config", &self.config)?;
        st.serialize_field("batch_count", &self.batch_count)?;
        st.serialize_field("save_every", &self.save_every)?;
        st.serialize_field("epoch", &self.epoch)?;
        st.end()
    }
}

//base for the TrainerBuilder trait
pub struct BuilderBase<O: OptimizerManager> {
    //constants
    pub config: Option<TrainingConfig>,
    pub save_every: Option<u32>, //number of epochs between saves
    // keep_n: usize, //number of snapshots to keep
    pub debug: Option<bool>,

    //counters
    pub epoch: Option<usize>,
    pub batch: Option<usize>,
    pub loss_accumulator: Option<f32>,

    pub optimizer: Option<O>,
    pub loss_handler: Option<Box<dyn FnMut(f32)>>, //called on every loss value produced by training
    pub save_handler: Option<Box<dyn Fn(u32) -> PathBuf>>, //called every time a network is saved to provide the file name
}
impl<O: OptimizerManager> BuilderBase<O> {
    pub fn new() -> BuilderBase<O> {
        BuilderBase {
            config: None,
            save_every: None,
            debug: None,
            epoch: None,
            batch: None,
            loss_accumulator: None,
            optimizer: None,
            loss_handler: None,
            save_handler: None,
        }
    }
    pub fn build(self, data_len: usize) -> TrainerBase<O> {
        let config = self.config.expect("Config needs to be initialized");
        let batch_count = data_len / config.batch_size;
        TrainerBase {
            config,
            batch_count,
            save_every: self.save_every.unwrap_or(u32::MAX),
            debug: self.debug.unwrap_or(false),
            epoch: self.epoch.unwrap_or(0),
            batch: self.batch.unwrap_or(0),
            loss_accumulator: self.loss_accumulator.unwrap_or(0.),
            optimizer: self
                .optimizer
                .expect("OptimizerManager needs to be initialized"),
            loss_handler: self.loss_handler.unwrap_or_else(|| Box::new(|_| {})),
            save_handler: self
                .save_handler
                .unwrap_or_else(|| Box::new(|e| PathBuf::from(format!("epoch_{}", e)))),
        }
    }
}

impl<O: OptimizerManager> Default for BuilderBase<O> {
    fn default() -> Self {
        Self::new()
    }
}

pub trait TrainerBuilder<O: OptimizerManager>
where
    Self: Sized,
{
    ///get reference to BaseBuilder
    fn bb(&self) -> &BuilderBase<O>;
    fn bb_mut(&mut self) -> &mut BuilderBase<O>;

    fn config(mut self, config: TrainingConfig) -> Self {
        self.bb_mut().config = Some(config);
        self
    }
    fn save_every(mut self, save_every: u32) -> Self {
        self.bb_mut().save_every = Some(save_every);
        self
    }
    fn debug(mut self, debug: bool) -> Self {
        self.bb_mut().debug = Some(debug);
        self
    }
    fn loss_accumulator(mut self, loss_accumulator: f32) -> Self {
        self.bb_mut().loss_accumulator = Some(loss_accumulator);
        self
    }
    fn loss_handler(mut self, loss_handler: impl FnMut(f32) + 'static) -> Self {
        self.bb_mut().loss_handler = Some(Box::new(loss_handler));
        self
    }
    fn save_handler(mut self, save_handler: impl Fn(u32) -> PathBuf + 'static) -> Self {
        self.bb_mut().save_handler = Some(Box::new(save_handler));
        self
    }
    fn optimizer(mut self, optimizer: O) -> Self {
        self.bb_mut().optimizer = Some(optimizer);
        self
    }

    fn from_snapshot(mut self, snapshot: &Snapshot) -> Self {
        self.bb_mut().config = Some(snapshot.config.clone());
        self.bb_mut().save_every = Some(snapshot.save_every);
        self.bb_mut().epoch = Some(snapshot.epoch);
        self
    }

    fn from_file(self, path: &str) -> anyhow::Result<Self> {
        let snapshot = Snapshot::from_file(path)?;
        Ok(self.from_snapshot(&snapshot))
    }
}
