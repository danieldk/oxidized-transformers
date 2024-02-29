use crate::architectures::BuildArchitecture;
use crate::error::BoxedError;
use crate::util::renaming_backend::RenamingBackend;
use candle_core::{DType, Device};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use snafu::{ResultExt, Snafu};

#[derive(Debug, Snafu)]
pub enum FromHFError {
    #[snafu(display("Cannot convert Hugging Face model config"))]
    ConvertConfig { source: BoxedError },

    #[snafu(display("Cannot build model"))]
    BuildModel { source: BoxedError },
}

pub trait FromHF {
    type Config: BuildArchitecture<Architecture = Self::Model>
        + TryFrom<Self::HFConfig, Error = BoxedError>;

    type HFConfig;

    type Model;

    fn from_hf(
        hf_config: Self::HFConfig,
        backend: Box<dyn SimpleBackend>,
        device: Device,
    ) -> Result<Self::Model, FromHFError> {
        let config = Self::Config::try_from(hf_config).context(ConvertConfigSnafu)?;
        let rename_backend = RenamingBackend::new(backend, Self::rename_parameters());
        let vb = VarBuilder::from_backend(Box::new(rename_backend), DType::F32, device);
        config.build(vb).context(BuildModelSnafu)
    }

    fn rename_parameters() -> impl Fn(&str) -> String + Send + Sync;
}
