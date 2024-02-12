use super::{
    hf_hub::FromHFHub,
    tokenizer::{
        Tokenizer, TokenizerConfigFile, TokenizerEncodeInput, TokenizerError,
        TokenizerFromConfigFiles,
    },
};
use lazy_static::lazy_static;
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};
use tokenizers::tokenizer::Tokenizer as HuggingFaceTokenizer;

use super::pieces::PiecesWithIds;

/// Wraps the tokenizers from the HuggingFace `tokenizers` package. It supports a
/// wide range of piece tokenizers, including word piece, byte pair encoding, and
/// sentencepiece unigram tokenizers. This is the tokenizer that should be used
/// in the majority of cases
pub struct HfTokenizer {
    tokenizer: HuggingFaceTokenizer,
    eos_piece: Option<String>,
}

impl HfTokenizer {
    pub fn new(
        tokenizer: HuggingFaceTokenizer,
        config: Option<serde_json::Value>,
        special_tokens_map: Option<serde_json::Value>,
    ) -> Self {
        Self {
            tokenizer,
            eos_piece: Self::_get_special_piece("eos_token", &config, &special_tokens_map),
        }
    }

    fn _get_special_piece(
        piece_name: &str,
        tokenizer_config: &Option<serde_json::Value>,
        special_tokens_map: &Option<serde_json::Value>,
    ) -> Option<String> {
        fn lookup_piece(piece_name: &str, json_value: &serde_json::Value) -> Option<String> {
            match json_value {
                serde_json::Value::Object(o) => o.get(piece_name).and_then(|piece| match piece {
                    serde_json::Value::String(s) => Some(s.clone()),
                    serde_json::Value::Object(_) => lookup_piece(piece_name, json_value),
                    _ => None,
                }),
                _ => None,
            }
        }

        if let Some(special_tokens_map) = special_tokens_map {
            lookup_piece(piece_name, special_tokens_map)
        } else if let Some(tokenizer_config) = tokenizer_config {
            lookup_piece(piece_name, tokenizer_config)
        } else {
            None
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
            .map_err(|e| TokenizerError::Decode { source: e })?;

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

        Ok(self
            .tokenizer
            .decode_batch(&converted_input, skip_special_pieces)
            .map_err(|e| TokenizerError::Decode { source: e })?)
    }

    fn piece_to_id(&self, piece: impl AsRef<str>) -> Option<Self::PieceId> {
        self.tokenizer.token_to_id(piece.as_ref())
    }

    fn eos_piece(&self) -> Option<&str> {
        self.eos_piece.as_deref()
    }
}

lazy_static! {
    static ref CONFIG_FILE_TOKENIZER_JSON: TokenizerConfigFile = TokenizerConfigFile {
        name: "tokenizer.json".to_string(),
        optional: false,
    };
    static ref CONFIG_FILE_TOKENIZER_CONFIG_JSON: TokenizerConfigFile = TokenizerConfigFile {
        name: "tokenizer_config.json".to_string(),
        optional: true,
    };
    static ref CONFIG_FILE_SPECIAL_TOKENS_MAP: TokenizerConfigFile = TokenizerConfigFile {
        name: "special_tokens_map.json".to_string(),
        optional: true,
    };
    static ref CONFIG_FILES: Vec<&'static TokenizerConfigFile> = vec![
        &CONFIG_FILE_TOKENIZER_JSON,
        &CONFIG_FILE_TOKENIZER_CONFIG_JSON,
        &CONFIG_FILE_SPECIAL_TOKENS_MAP,
    ];
}

impl TokenizerFromConfigFiles for HfTokenizer {
    fn config_files() -> &'static [&'static super::tokenizer::TokenizerConfigFile] {
        CONFIG_FILES.as_slice()
    }

    fn from_config_files(
        config_files: HashMap<TokenizerConfigFile, Option<PathBuf>>,
    ) -> Result<Self, TokenizerError> {
        let tokenizer_json_path = config_files
            .get(&CONFIG_FILE_TOKENIZER_JSON)
            .and_then(|p| p.as_ref())
            .ok_or_else(|| TokenizerError::MissingConfigFile {
                filename: CONFIG_FILE_TOKENIZER_JSON.name.clone(),
            })?;
        let tokenizer_config_json_path = config_files
            .get(&CONFIG_FILE_TOKENIZER_CONFIG_JSON)
            .and_then(|p| p.as_ref());
        let special_tokens_map_path = config_files
            .get(&CONFIG_FILE_SPECIAL_TOKENS_MAP)
            .and_then(|p| p.as_ref());

        let hf_tokenizer = HuggingFaceTokenizer::from_file(tokenizer_json_path)
            .map_err(|e| TokenizerError::HFTokenizer { source: e })?;

        let deserialized_tokenizer_config_json = match tokenizer_config_json_path {
            Some(p) => Some(parse_tokenizer_config_file_as_json(p)?),
            None => None,
        };
        let deserialized_special_tokens_map = match special_tokens_map_path {
            Some(p) => Some(parse_tokenizer_config_file_as_json(p)?),
            None => None,
        };

        Ok(HfTokenizer::new(
            hf_tokenizer,
            deserialized_tokenizer_config_json,
            deserialized_special_tokens_map,
        ))
    }
}

impl FromHFHub for HfTokenizer {}

fn parse_tokenizer_config_file_as_json(path: &Path) -> Result<serde_json::Value, TokenizerError> {
    let config_file = File::open(path).map_err(|e| TokenizerError::OpenConfigFile {
        path: path.to_path_buf(),
        source: e,
    })?;

    let json_value: serde_json::Value = serde_json::from_reader(config_file)
        .map_err(|e| TokenizerError::DeserializeConfigJSON { source: e })?;

    Ok(json_value)
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
