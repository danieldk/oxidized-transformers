use serde::{Deserialize, Serialize};

use crate::error::BoxedError;
use crate::layers::activation::Activation;
use crate::layers::attention::{AttentionHeads, QkvMode, SelfAttentionConfig};
use crate::layers::embeddings::QueryKeyRotaryEmbeddingsConfig;
use crate::layers::feedforward::PointwiseFeedForwardConfig;
use crate::layers::layer_norm::RMSNormConfig;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerLayerConfig};
use crate::models::transformer::{TransformerCausalLMConfig, TransformerDecoderConfig};

impl TryFrom<HfLlamaCausalLMConfig> for TransformerCausalLMConfig {
    type Error = BoxedError;

    fn try_from(config: HfLlamaCausalLMConfig) -> Result<Self, Self::Error> {
        Ok(Self::default()
            .hidden_size(config.decoder.hidden_size)
            // Input and output vocab sizes are the same.
            .n_pieces(config.decoder.vocab_size)
            .decoder(Box::new(TransformerDecoderConfig::try_from(
                config.decoder,
            )?)))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HfLlamaCausalLMConfig {
    #[serde(flatten)]
    decoder: HFLlamaDecoderConfig,
}

impl TryFrom<HFLlamaDecoderConfig> for TransformerDecoderConfig {
    type Error = BoxedError;

    fn try_from(hf_config: HFLlamaDecoderConfig) -> Result<Self, Self::Error> {
        let layer_norm = Box::new(
            RMSNormConfig::default()
                .eps(hf_config.rms_norm_eps as f64)
                .size(hf_config.hidden_size),
        );

        let embeddings = TransformerEmbeddingsConfig::default()
            .embedding_width(hf_config.hidden_size)
            .hidden_width(hf_config.hidden_size)
            .n_pieces(hf_config.vocab_size);

        let feedforward = PointwiseFeedForwardConfig::default()
            .activation(Box::new(hf_config.hidden_act))
            .hidden_width(hf_config.hidden_size)
            .intermediate_width(hf_config.intermediate_size)
            .layer_norm(layer_norm.clone())
            .use_bias(false)
            .use_gate(true);

        let attention = SelfAttentionConfig::default()
            .attention_heads(AttentionHeads {
                n_query_heads: hf_config.num_attention_heads,
                n_key_value_heads: hf_config.num_key_value_heads,
                qkv_mode: QkvMode::Separate,
            })
            .hidden_width(hf_config.hidden_size)
            .layer_norm(layer_norm.clone())
            .rotary_embeddings(Some(
                QueryKeyRotaryEmbeddingsConfig::default()
                    .head_width(hf_config.hidden_size / hf_config.num_attention_heads)
                    .seq_len(hf_config.max_position_embeddings),
            ));

        let layer = TransformerLayerConfig::default()
            .attention(attention)
            .feedforward(feedforward);

        Ok(TransformerDecoderConfig::default()
            .embeddings(embeddings)
            .layer(Box::new(layer))
            .n_hidden_layers(hf_config.num_hidden_layers)
            .output_layer_norm(layer_norm))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HFLlamaDecoderConfig {
    hidden_act: Activation,
    hidden_size: usize,
    initializer_range: f32,
    intermediate_size: usize,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    vocab_size: usize,
}
