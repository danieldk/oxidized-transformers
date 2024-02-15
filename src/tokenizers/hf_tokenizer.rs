use self::config::ConfigWithEosToken;

use super::{
    config::{
        DeserializeConfigJSONSnafu, OpenConfigFileSnafu, ResolvePathsSnafu, TokenizerConfigError,
        TokenizerConfigWithPaths, TokenizerFromConfig,
    },
    hf_hub::FromHFHub,
    tokenizer::{DecodeSnafu, EncodeSnafu, Tokenizer, TokenizerEncodeInput, TokenizerError},
};
use crate::error::BoxedError;
use snafu::ResultExt;
use std::{fs::File, path::PathBuf};
use tokenizers::tokenizer::Tokenizer as HuggingFaceTokenizer;

use super::pieces::PiecesWithIds;

/// Configuration for the Hugging Face tokenizer.
#[derive(Debug, Clone, Default)]
pub struct HfTokenizerConfig {
    /// Path to the `tokenizer.json` file.
    pub tokenizer_json: PathBuf,
    /// Path to the `tokenizer_config.json` file.
    pub tokenizer_config_json: Option<PathBuf>,
    /// Path to the `special_tokens_map.json` file.
    pub special_tokens_map_json: Option<PathBuf>,
}

impl TokenizerConfigWithPaths for HfTokenizerConfig {
    fn resolve_paths<'a, F>(&'a mut self, resolver: F) -> Result<(), TokenizerConfigError>
    where
        F: Fn(/* filename: */ &'a str) -> Result<PathBuf, BoxedError>,
    {
        self.tokenizer_json = resolver("tokenizer.json").context(ResolvePathsSnafu)?;
        self.tokenizer_config_json = resolver("tokenizer_config.json").ok();
        self.special_tokens_map_json = resolver(" special_tokens_map.json").ok();

        Ok(())
    }
}

/// Wraps the tokenizers from the HuggingFace `tokenizers` package. It supports a
/// wide range of piece tokenizers, including word piece, byte pair encoding, and
/// sentencepiece unigram tokenizers. This is the tokenizer that should be used
/// in the majority of cases
pub struct HfTokenizer {
    tokenizer: HuggingFaceTokenizer,
    eos_piece: Option<String>,
}

impl HfTokenizer {
    pub(crate) fn new(
        tokenizer: HuggingFaceTokenizer,
        config: Option<&config::ConfigWithEosToken>,
        special_tokens_map: Option<&config::ConfigWithEosToken>,
    ) -> Self {
        let eos_piece = config
            .map(|e| e.eos_token())
            .flatten()
            .or_else(|| special_tokens_map.map(|e| e.eos_token()).flatten());

        Self {
            tokenizer,
            eos_piece: eos_piece.cloned(),
        }
    }
}

impl Tokenizer for HfTokenizer {
    type PieceId = u32;

    fn encode<V, I>(&self, input: V) -> Result<PiecesWithIds<Self::PieceId>, TokenizerError>
    where
        V: AsRef<[TokenizerEncodeInput<I>]>,
        I: AsRef<str>,
    {
        let converted_input = input
            .as_ref()
            .iter()
            .map(|input| match input {
                TokenizerEncodeInput::RawString(s) => {
                    tokenizers::EncodeInput::Single(s.as_ref().into())
                }
            })
            .collect::<Vec<_>>();

        let encoding = self
            .tokenizer
            .encode_batch(converted_input, true)
            .context(EncodeSnafu)?;

        Ok(PiecesWithIds {
            ids: encoding
                .iter()
                .map(|ids| ids.get_ids().to_owned())
                .collect(),
            pieces: encoding
                .iter()
                .map(|ids| ids.get_tokens().to_owned())
                .collect(),
        })
    }

    fn decode<V, I>(
        &self,
        input: V,
        skip_special_pieces: bool,
    ) -> Result<Vec<String>, TokenizerError>
    where
        V: AsRef<[I]>,
        I: AsRef<[Self::PieceId]>,
    {
        let converted_input = input
            .as_ref()
            .iter()
            .map(|input| input.as_ref())
            .collect::<Vec<_>>();

        self.tokenizer
            .decode_batch(&converted_input, skip_special_pieces)
            .context(DecodeSnafu)
    }

    fn piece_to_id(&self, piece: impl AsRef<str>) -> Option<Self::PieceId> {
        self.tokenizer.token_to_id(piece.as_ref())
    }

    fn eos_piece(&self) -> Option<&str> {
        self.eos_piece.as_deref()
    }
}

impl TokenizerFromConfig for HfTokenizer {
    type Config = HfTokenizerConfig;

    fn from_config(config: &Self::Config) -> Result<Self, BoxedError>
    where
        Self: Sized,
    {
        let tokenizer = HuggingFaceTokenizer::from_file(&config.tokenizer_json)?;

        let open_config_file =
            |p: &PathBuf| -> Result<Option<ConfigWithEosToken>, TokenizerConfigError> {
                let file = File::open(p).context(OpenConfigFileSnafu { path: p.clone() })?;
                let deserialized: Option<ConfigWithEosToken> =
                    serde_json::from_reader(file).context(DeserializeConfigJSONSnafu)?;
                Ok(deserialized)
            };

        let tokenizer_config = config
            .tokenizer_config_json
            .as_ref()
            .map(&open_config_file)
            .transpose()?
            .flatten();

        let special_tokens_map = config
            .special_tokens_map_json
            .as_ref()
            .map(&open_config_file)
            .transpose()?
            .flatten();

        Ok(Self::new(
            tokenizer,
            tokenizer_config.as_ref(),
            special_tokens_map.as_ref(),
        ))
    }
}

impl FromHFHub for HfTokenizer {
    type TokenizerConfig = HfTokenizerConfig;
}

mod config {
    use std::collections::HashMap;

    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    /// Represents an EOS token in the tokenizer configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub(crate) enum EosTokenInConfig {
        Default(String),
        Wrapped { content: Option<String> },
    }

    /// Represents a tokenizer configuration that includes an EOS token.
    /// Primarily used to with `tokenizer_config.json` and `special_tokens_map.json` files.
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub(crate) struct ConfigWithEosToken {
        #[serde(default)]
        eos_token: Option<EosTokenInConfig>,
        #[serde(flatten)]
        _extra: HashMap<String, Value>,
    }

    impl ConfigWithEosToken {
        pub fn eos_token(&self) -> Option<&String> {
            self.eos_token
                .as_ref()
                .map(|e| match e {
                    EosTokenInConfig::Default(s) => Some(s),
                    EosTokenInConfig::Wrapped { content } => content.as_ref(),
                })
                .flatten()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_can_load_from_hf_hub() {
        let tokenizer = HfTokenizer::from_hf_hub("tiiuae/falcon-7b", None)
            .expect("Failed to load tokenizer from HF Hub");
        assert_eq!(tokenizer.eos_piece(), Some("<|endoftext|>"));

        let tokenizer = HfTokenizer::from_hf_hub("bert-base-cased", None)
            .expect("Failed to load tokenizer from HF Hub");
        assert_eq!(tokenizer.eos_piece(), None);
    }
}
