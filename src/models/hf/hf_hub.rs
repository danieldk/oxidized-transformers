use candle_core::Device;
use hf_hub::api::sync::ApiError;
use serde::de::DeserializeOwned;
use snafu::{ResultExt, Snafu};
use std::fs::File;
use std::path::PathBuf;

use crate::error::BoxedError;
use crate::models::hf::from_hf::{FromHF, FromHFError};
use crate::models::hf::{HFCheckpoint, HFCheckpointError};
use crate::repository::hf_hub::HfHubRepo;
use crate::repository::repo::Repo;

#[derive(Debug, Snafu)]
pub enum FromHfHubError {
    #[snafu(display("Model configuration file does not exist"))]
    ConfigPath,

    #[snafu(display("Cannot convert Hugging Face model"))]
    FromHF { source: FromHFError },

    #[snafu(display("Hugging Face Hub error"))]
    HFHub { source: ApiError },

    #[snafu(display("Hugging Face Hub repository error"))]
    HFHUbRepo { source: BoxedError },

    #[snafu(display("Cannot deserialize JSON"))]
    JSON { source: serde_json::Error },

    #[snafu(display("Cannot open or load checkpoint"))]
    LoadCheckpoint { source: HFCheckpointError },

    #[snafu(display("Cannot open file for reading: {path:?}"))]
    Open {
        path: PathBuf,
        source: std::io::Error,
    },
}

pub trait FromHFHub
where
    Self: Sized,
{
    type Model;

    fn from_hf_hub(
        name: &str,
        revision: Option<&str>,
        device: Device,
    ) -> Result<Self::Model, FromHfHubError>;
}

impl<HF, C, HC> FromHFHub for HF
where
    HF: FromHF<Config = C, HFConfig = HC>,
    HC: DeserializeOwned,
    C: TryFrom<HC, Error = BoxedError>,
{
    type Model = HF::Model;

    fn from_hf_hub(
        name: &str,
        revision: Option<&str>,
        device: Device,
    ) -> Result<Self::Model, FromHfHubError> {
        let repo = HfHubRepo::new(name, revision).context(HFHUbRepoSnafu)?;
        let config_file = repo.file("config.json").context(HFHUbRepoSnafu)?;
        let config_path = config_file.ok_or(FromHfHubError::ConfigPath)?;
        let config_file = File::open(&config_path).context(OpenSnafu { path: config_path })?;
        let hf_config: HC = serde_json::from_reader(config_file).context(JSONSnafu)?;

        let backend = HFCheckpoint::SafeTensors
            .load(&repo)
            .context(LoadCheckpointSnafu)?;

        Self::from_hf(hf_config, backend, device).context(FromHFSnafu)
    }
}
