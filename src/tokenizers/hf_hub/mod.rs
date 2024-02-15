use std::error::Error;

use crate::error::BoxedError;
use hf_hub::api::sync::{Api, ApiError};
use hf_hub::{Repo, RepoType};
use snafu::{ResultExt, Snafu};

use super::config::{TokenizerConfigWithPaths, TokenizerFromConfig};

#[derive(Debug, Snafu)]
pub enum FromHfHubError {
    #[snafu(display("Hugging Face Hub error"))]
    HFHub { source: ApiError },

    #[snafu(display("Could not resolve filepath in tokenizer config"))]
    ResolveConfigPaths { source: BoxedError },

    #[snafu(display("Could not create tokenizer from config"))]
    FromConfig { source: BoxedError },
}

/// Trait implemented by tokenziers that can be loaded from the Hugging Face Hub.
pub trait FromHFHub
where
    Self: Sized + TokenizerFromConfig<Config = Self::TokenizerConfig>,
{
    type TokenizerConfig: TokenizerConfigWithPaths;

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

        let mut config = Self::default_config();
        let path_resolver = |filename| {
            repo.download(filename)
                .context(HFHubSnafu)
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)
        };

        config
            .resolve_paths(path_resolver)
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)
            .context(ResolveConfigPathsSnafu)?;

        Ok(Self::from_config(&config).context(FromConfigSnafu)?)
    }
}
