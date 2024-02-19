use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use candle_core::safetensors::MmapedSafetensors;
use candle_nn::var_builder::SimpleBackend;
use hf_hub::api::sync::{ApiError, ApiRepo};
use serde::Deserialize;
use snafu::{ResultExt, Snafu};

static SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
static SAFETENSORS_SINGLE: &str = "model.safetensors";

#[derive(Debug, Snafu)]
pub enum CheckpointError {
    #[snafu(display("Cannot download checkpoint: {name}"))]
    Download { source: ApiError, name: String },

    #[snafu(display("Cannot open or load checkpoint"))]
    LoadCheckpoint { source: candle_core::Error },

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
    pub fn load(self, api_repo: &ApiRepo) -> Result<Box<dyn SimpleBackend>, CheckpointError> {
        match self {
            Checkpoint::SafeTensors => Self::load_safetensors(api_repo),
        }
    }

    fn load_safetensors(api_repo: &ApiRepo) -> Result<Box<dyn SimpleBackend>, CheckpointError> {
        let paths = match api_repo.get(SAFETENSORS_INDEX) {
            Ok(index_path) => Self::load_safetensors_multi(api_repo, index_path),
            Err(_) => Self::load_safetensors_single(api_repo),
        }?;

        Ok(Box::new(unsafe {
            MmapedSafetensors::multi(&paths).context(LoadCheckpointSnafu)?
        }))
    }
    fn load_safetensors_multi(
        api_repo: &ApiRepo,
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
                api_repo
                    .get(&shard_name)
                    .context(DownloadSnafu { name: shard_name })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(shards)
    }

    fn load_safetensors_single(api_repo: &ApiRepo) -> Result<Vec<PathBuf>, CheckpointError> {
        Ok(vec![api_repo.get(SAFETENSORS_SINGLE).context(
            DownloadSnafu {
                name: SAFETENSORS_SINGLE,
            },
        )?])
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
