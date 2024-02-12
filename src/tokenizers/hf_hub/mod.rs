use hf_hub::api::sync::{Api, ApiError};
use hf_hub::{Repo, RepoType};
use snafu::{ResultExt, Snafu};
use std::collections::HashMap;

use super::tokenizer::{TokenizerError, TokenizerFromConfigFiles};

#[derive(Debug, Snafu)]
pub enum FromHfHubError {
    #[snafu(display("Hugging Face Hub error"))]
    HFHub { source: ApiError },

    #[snafu(display("Cannot construct tokenizer from configuration"))]
    TokenizerFromConfig { source: TokenizerError },
}

/// Trait implemented by tokenziers that can be loaded from the Hugging Face Hub.
pub trait FromHFHub
where
    Self: Sized + TokenizerFromConfigFiles,
{
    /// Load a tokenizer from the Hugging Face Hub.
    ///
    /// * name - Name of the model on the Hugging Face Hub.
    /// * revision - Revision of the model to load. If `None`, the main branch is used.
    fn from_hf_hub(name: &str, revision: Option<&str>) -> Result<Self, FromHfHubError> {
        let revision = revision
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| "main".to_string());
        let hub_api = Api::new().context(HFHubSnafu)?;
        let repo = hub_api.repo(Repo::with_revision(
            name.to_string(),
            RepoType::Model,
            revision,
        ));

        let config_remote_paths = Self::config_files();
        let mut config_local_paths = HashMap::with_capacity(config_remote_paths.len());
        for file in config_remote_paths {
            let path = if file.optional {
                repo.download(&file.name).ok()
            } else {
                Some(repo.download(&file.name).context(HFHubSnafu)?)
            };

            config_local_paths.insert((*file).clone(), path);
        }

        Ok(Self::from_config_files(config_local_paths).context(TokenizerFromConfigSnafu)?)
    }
}
