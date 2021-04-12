pub use logger::{Logger, MockLogger, LogFile};
pub mod logger;
pub use stochaistic::Stochaistic;
pub mod stochaistic;

#[derive(Debug, Clone)]
pub struct Data<const IN: usize, const OUT: usize> {
    pub input: [f32; IN],
    pub target: [f32; OUT],
}

impl<const IN: usize, const OUT: usize> Data<IN, OUT> {
    pub fn new(input: [f32; IN], target: [f32; OUT]) -> Self {
        Self { input, target }
    }
}

impl<const IN: usize, const OUT: usize> From<([f32; IN], [f32; OUT])> for Data<IN, OUT> {
    fn from(tuple: ([f32; IN], [f32; OUT])) -> Self {
        let (input, target) = tuple;
        Self::new(input, target)
    }
}

pub use errors::Error;

use crate::optimizer::Optimizer;
mod errors {
    use crate::misc::error::BuilderError;
    use std::fmt::{Debug, Display};

    pub enum Error {
        Missing(BuilderError),
        WrongSize { batch_size: u32, data_size: usize },
    }

    impl Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Error::Missing(e) => Display::fmt(e, f),
                Error::WrongSize {
                    batch_size,
                    data_size,
                } => write!(
                    f,
                    "Batch size cannot be larger than data length. batch_size: {}, data_len: {}",
                    batch_size, data_size
                ),
            }
        }
    }

    impl Error {
        pub fn missing(name: &'static str) -> Self {
            Self::Missing(BuilderError::new(name))
        }
    }

    impl Debug for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            Display::fmt(self, f)
        }
    }
}

#[derive(Debug)]
pub struct Trainer<O, const IN: usize, const OUT: usize> {
    optimizer: Option<O>,
    logger: Box<dyn Logger>,
    seed: u64,
    batch_size: Option<u32>,
    data: Option<Box<[Data<IN, OUT>]>>,
}

impl<O, const IN: usize, const OUT: usize> Default for Trainer<O, IN, OUT> {
    fn default() -> Self {
        Self {
            optimizer: None,
            data: None,
            logger: Box::new(MockLogger),
            seed: 0,
            batch_size: None,
        }
    }
}

impl<O, const IN: usize, const OUT: usize> Trainer<O, IN, OUT>
where
    O: Optimizer,
{
    pub fn new() -> Self {
        Default::default()
    }

    pub fn optimizer(mut self, optimizer: O) -> Self {
        self.optimizer = Some(optimizer);
        self
    }
    pub fn data<D: Into<Box<[Data<IN, OUT>]>>>(mut self, data: D) -> Self {
        self.data = Some(data.into());
        self
    }
    pub fn logger<L: Logger + 'static>(mut self, logger: L) -> Self {
        self.logger = Box::new(logger);
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    pub fn batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn build(self) -> Result<Stochaistic<O, IN, OUT>, Error> {
        let optimizer = self.optimizer.ok_or(Error::missing("optimizer"))?;
        let data = self.data.ok_or(Error::missing("data"))?;
        let batch_size = self.batch_size.ok_or(Error::missing("batch_size"))?;
        let logger = self.logger;
        let seed = self.seed;

        if batch_size as usize > data.len() {
            return Err(Error::WrongSize {
                batch_size,
                data_size: data.len(),
            });
        }

        Ok(Stochaistic::new(optimizer, data, logger, batch_size, seed))
    }
}