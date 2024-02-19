use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::api::sync::{Api, ApiError};
use hf_hub::{Repo, RepoType};
use serde::de::DeserializeOwned;
use snafu::{ResultExt, Snafu};

use crate::error::BoxedError;
use crate::models::hf::{Checkpoint, CheckpointError};
use crate::util::renaming_backend::RenamingBackend;

pub trait TransformerFromConfig
where
    Self: Sized,
{
    type Config;
    fn from_config(vb: VarBuilder, config: &Self::Config) -> Result<Self, BoxedError>
    where
        Self: Sized;
}

pub trait FromHFStateDict {
    fn from_hf_state_dict<T>(params: &HashMap<impl AsRef<str>, T>) -> HashMap<String, T>;
}

pub trait HFRenames {
    fn hf_renames() -> impl Fn(&str) -> String + Send + Sync;
}

#[derive(Debug, Snafu)]
pub enum FromHfHubError {
    #[snafu(display("Cannot convert Hugging Face model config"))]
    ConvertConfig { source: BoxedError },

    #[snafu(display("Hugging Face Hub error"))]
    HFHub { source: ApiError },

    #[snafu(display("Cannot deserialize JSON"))]
    JSON { source: serde_json::Error },

    #[snafu(display("Cannot open or load checkpoint"))]
    LoadCheckpoint { source: CheckpointError },

    #[snafu(display("Cannot construct model from configuration"))]
    ModelFromConfig { source: BoxedError },

    #[snafu(display("Cannot open file for reading: {path:?}"))]
    Open {
        path: PathBuf,
        source: std::io::Error,
    },
}

pub trait FromHFHub
where
    Self: Sized + HFRenames + TransformerFromConfig,
    Self::Config: TryFrom<Self::HFConfig, Error = BoxedError>,
{
    type HFConfig: DeserializeOwned;

    fn from_hf_hub(
        name: &str,
        revision: Option<&str>,
        device: Device,
    ) -> Result<Self, FromHfHubError> {
        let revision = revision
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| "main".to_string());
        let hub_api = Api::new().context(HFHubSnafu)?;
        let repo = hub_api.repo(Repo::with_revision(
            name.to_string(),
            RepoType::Model,
            revision,
        ));
        let config_path = repo.get("config.json").context(HFHubSnafu)?;
        let config_file = File::open(&config_path).context(OpenSnafu { path: config_path })?;
        let hf_config: Self::HFConfig = serde_json::from_reader(config_file).context(JSONSnafu)?;
        let config = Self::Config::try_from(hf_config).context(ConvertConfigSnafu)?;

        let backend = Checkpoint::SafeTensors
            .load(&repo)
            .context(LoadCheckpointSnafu)?;
        let rename_backend = RenamingBackend::new(backend, Self::hf_renames());
        let vb = VarBuilder::from_backend(Box::new(rename_backend), DType::F32, device);
        Self::from_config(vb, &config).context(ModelFromConfigSnafu)
    }
}
