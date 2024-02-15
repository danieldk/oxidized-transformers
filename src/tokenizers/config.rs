use crate::error::BoxedError;
use std::path::PathBuf;

use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(super)))]
pub enum TokenizerConfigError {
    #[snafu(display("Couldn't resolve filepaths in tokenizer configuration"))]
    ResolvePaths { source: BoxedError },

    #[snafu(display("Cannot open tokenizer config for reading: {path:?}"))]
    OpenConfigFile {
        path: PathBuf,
        source: std::io::Error,
    },

    #[snafu(display("Cannot deserialize tokenizer JSON configuation"))]
    DeserializeConfigJSON { source: serde_json::Error },
}

/// Trait for tokenizer configs that have paths to local files.
pub trait TokenizerConfigWithPaths {
    /// Resolves all file paths in the tokenizer config.
    ///
    /// * `resolver` - A function that takes a filename and returns a resolved path.
    fn resolve_paths<'a, F>(&'a mut self, resolver: F) -> Result<(), TokenizerConfigError>
    where
        F: Fn(/* filename: */ &'a str) -> Result<PathBuf, BoxedError>;
}

/// Trait implemented by tokenizers that can be created from a configuration.
pub trait TokenizerFromConfig
where
    Self: Sized,
    Self::Config: Default,
{
    type Config;

    fn from_config(config: &Self::Config) -> Result<Self, BoxedError>
    where
        Self: Sized;

    fn default_config() -> Self::Config {
        Default::default()
    }
}
