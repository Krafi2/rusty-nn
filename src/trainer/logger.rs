use std::{
    fmt::Debug,
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

pub trait Logger: Debug {
    fn epoch_loss(&mut self, epoch: u32, loss: f32);

    fn batch_loss(&mut self, epoch: u32, batch: u32, loss: f32);
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MockLogger;

impl Logger for MockLogger {
    fn epoch_loss(&mut self, _epoch: u32, _loss: f32) {}

    fn batch_loss(&mut self, _epoch: u32, _batch: u32, _loss: f32) {}
}

#[derive(Debug)]
pub struct LogFile {
    file: PathBuf,
    writer: BufWriter<File>,
}

impl LogFile {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(Self {
            file: path.as_ref().to_owned(),
            writer: BufWriter::new(File::create(path)?),
        })
    }
}

impl Logger for LogFile {
    fn epoch_loss(&mut self, _epoch: u32, loss: f32) {
        if let Err(e) = writeln!(self.writer, "{}", loss) {
            eprintln!(
                "Error while logging loss to file: {}\nError: {}",
                self.file.display(),
                e
            );
        }
    }

    fn batch_loss(&mut self, _epoch: u32, _batch: u32, _loss: f32) {}
}
