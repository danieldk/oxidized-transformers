use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use crate::error::BoxedError;
use crate::repository::repo::Repo;
use candle_core::safetensors::MmapedSafetensors;
use candle_nn::var_builder::SimpleBackend;
use serde::Deserialize;
use snafu::{ResultExt, Snafu};

static SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
static SAFETENSORS_SINGLE: &str = "model.safetensors";

#[derive(Debug, Snafu)]
pub enum CheckpointError {
    #[snafu(display("Cannot download checkpoint: {name}"))]
    Download { source: BoxedError, name: String },

    #[snafu(display("Cannot get file from index {}", path.to_string_lossy()))]
    Index { source: BoxedError, path: PathBuf },

    #[snafu(display("Cannot open or load checkpoint"))]
    LoadCheckpoint { source: candle_core::Error },

    #[snafu(display("Shard does not exist: {}", name))]
    NonExistentShard { name: String },

    #[snafu(display("Cannot open SafeTensors index file: {}", path.to_string_lossy()))]
    OpenSafeTensorsIndex { source: io::Error, path: PathBuf },

    #[snafu(display("Cannot parse SafeTensors index file: {}", path.to_string_lossy()))]
    ParseSafeTensorsIndex {
        source: serde_json::Error,
        path: PathBuf,
    },
}

#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum Checkpoint {
    SafeTensors,
}

impl Checkpoint {
    pub fn load(self, repo: &impl Repo) -> Result<Box<dyn SimpleBackend>, CheckpointError> {
        match self {
            Checkpoint::SafeTensors => Self::load_safetensors(repo),
        }
    }

    fn load_safetensors(repo: &impl Repo) -> Result<Box<dyn SimpleBackend>, CheckpointError> {
        let file = repo.file(SAFETENSORS_INDEX).context(IndexSnafu {
            path: SAFETENSORS_INDEX,
        })?;
        let paths = match file {
            Some(index_path) => Self::load_safetensors_multi(repo, index_path),
            None => Self::load_safetensors_single(repo),
        }?;

        Ok(Box::new(unsafe {
            MmapedSafetensors::multi(&paths).context(LoadCheckpointSnafu)?
        }))
    }
    fn load_safetensors_multi(
        repo: &impl Repo,
        index_path: impl AsRef<Path>,
    ) -> Result<Vec<PathBuf>, CheckpointError> {
        // Parse the shard index.
        let index_path = index_path.as_ref();
        let index_file = BufReader::new(
            File::open(index_path).context(OpenSafeTensorsIndexSnafu { path: index_path })?,
        );
        let index: SafeTensorsIndex = serde_json::from_reader(index_file)
            .context(ParseSafeTensorsIndexSnafu { path: index_path })?;

        // Get shards.
        let shards = index
            .shards()
            .into_iter()
            .map(|shard_name| {
                repo.file(&shard_name)
                    .context(DownloadSnafu {
                        name: shard_name.clone(),
                    })
                    .and_then(|f| {
                        f.ok_or(CheckpointError::NonExistentShard {
                            name: shard_name.clone(),
                        })
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(shards)
    }

    fn load_safetensors_single(repo: &impl Repo) -> Result<Vec<PathBuf>, CheckpointError> {
        let file = repo.file(SAFETENSORS_SINGLE).context(DownloadSnafu {
            name: SAFETENSORS_SINGLE,
        })?;
        let path = file.ok_or(CheckpointError::NonExistentShard {
            name: SAFETENSORS_SINGLE.to_string(),
        })?;

        Ok(vec![path])
    }
}

#[derive(Debug, Deserialize)]
struct SafeTensorsIndex {
    weight_map: HashMap<String, String>,
}

impl SafeTensorsIndex {
    /// Get the names of the shards.
    fn shards(&self) -> HashSet<String> {
        self.weight_map.values().cloned().collect()
    }
}
