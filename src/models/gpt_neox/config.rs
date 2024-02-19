use serde::{Deserialize, Serialize};

use crate::error::BoxedError;
use crate::layers::activation::Activation;
use crate::layers::attention::{
    AttentionHeads, QkvMode, ScaledDotProductAttentionConfig, SelfAttentionConfig,
};
use crate::layers::dropout::DropoutConfig;
use crate::layers::embeddings::QueryKeyRotaryEmbeddingsConfig;
use crate::layers::feedforward::PointwiseFeedForwardConfig;
use crate::layers::layer_norm::LayerNormConfig;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerLayerConfig};
use crate::models::transformer::TransformerDecoderConfig;

impl TryFrom<HFGPTNeoXConfig> for TransformerDecoderConfig {
    type Error = BoxedError;

    fn try_from(hf_config: HFGPTNeoXConfig) -> Result<Self, Self::Error> {
        let attention_dropout =
            Box::new(DropoutConfig::default().p(hf_config.attention_probs_dropout_prob));

        let layer_norm = Box::new(
            LayerNormConfig::default()
                .eps(hf_config.layer_norm_eps as f64)
                .size(hf_config.hidden_size),
        );

        let embeddings = TransformerEmbeddingsConfig::default()
            .embedding_width(hf_config.hidden_size)
            .hidden_width(hf_config.hidden_size)
            .n_pieces(hf_config.vocab_size);

        let dropout = Box::new(DropoutConfig::default().p(hf_config.hidden_dropout_prob));
        let feedforward = PointwiseFeedForwardConfig::default()
            .activation(Box::new(hf_config.hidden_act))
            .dropout(dropout.clone())
            .hidden_width(hf_config.hidden_size)
            .intermediate_width(hf_config.intermediate_size)
            .layer_norm(layer_norm.clone());

        let attention = SelfAttentionConfig::default()
            .attention_heads(AttentionHeads {
                n_query_heads: hf_config.num_attention_heads,
                n_key_value_heads: hf_config.num_attention_heads,
                qkv_mode: QkvMode::MergedSplitBefore,
            })
            .attention_scorer(Box::new(
                ScaledDotProductAttentionConfig::default().dropout(attention_dropout),
            ))
            .dropout(dropout)
            .hidden_width(hf_config.hidden_size)
            .layer_norm(layer_norm.clone())
            .rotary_embeddings(Some(
                QueryKeyRotaryEmbeddingsConfig::default()
                    .base(hf_config.rotary_emb_base)
                    .fraction(hf_config.rotary_pct)
                    .head_width(hf_config.hidden_size / hf_config.num_attention_heads)
                    .seq_len(hf_config.max_position_embeddings),
            ));

        let layer = TransformerLayerConfig::default()
            .attention(attention)
            .feedforward(feedforward)
            .use_parallel_attention(hf_config.use_parallel_residual);

        Ok(TransformerDecoderConfig::default()
            .embeddings(embeddings)
            .layer(Box::new(layer))
            .n_hidden_layers(hf_config.num_hidden_layers)
            .output_layer_norm(layer_norm))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HFGPTNeoXConfig {
    attention_probs_dropout_prob: f32,
    hidden_act: Activation,
    hidden_dropout_prob: f32,
    hidden_size: usize,
    initializer_range: f32,
    intermediate_size: usize,
    layer_norm_eps: f32,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    rotary_emb_base: usize,
    rotary_pct: f32,
    tie_word_embeddings: bool,
    type_vocab_size: usize,
    use_parallel_residual: bool,
    vocab_size: usize,
}
