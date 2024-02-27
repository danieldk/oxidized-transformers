use std::sync::OnceLock;

use candle_core::Tensor;
use candle_nn::VarBuilder;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::architectures::{BuildDecoder, Decoder, DecoderOutput};
use crate::error::BoxedError;
use crate::kv_cache::KeyValueCache;
use crate::layers::activation::Activation;
use crate::layers::attention::{
    AttentionHeads, AttentionMask, QkvMode, ScaledDotProductAttentionConfig, SelfAttentionConfig,
};
use crate::layers::dropout::DropoutConfig;
use crate::layers::embeddings::QueryKeyRotaryEmbeddingsConfig;
use crate::layers::feedforward::PointwiseFeedForwardConfig;
use crate::layers::layer_norm::LayerNormConfig;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerLayerConfig};
use crate::models::hf_hub::{FromHFHub, HFRenames, TransformerFromConfig};
use crate::models::transformer::{TransformerDecoder, TransformerDecoderConfig};

pub struct GPTNeoXDecoder {
    inner: TransformerDecoder,
}

impl Decoder for GPTNeoXDecoder {
    type Cache = KeyValueCache;

    fn forward_t(
        &self,
        piece_ids: &Tensor,
        mask: &AttentionMask,
        cache: &mut Self::Cache,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<DecoderOutput, BoxedError> {
        self.inner
            .forward_t(piece_ids, mask, cache, positions, train)
    }
}

impl TransformerFromConfig for GPTNeoXDecoder {
    type Config = TransformerDecoderConfig;

    fn from_config(vb: VarBuilder, config: &Self::Config) -> Result<Self, BoxedError> {
        Ok(Self {
            inner: config.build(vb)?,
        })
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

impl FromHFHub for GPTNeoXDecoder {
    type HFConfig = HFGPTNeoXConfig;
}

impl HFRenames for GPTNeoXDecoder {
    fn hf_renames() -> impl Fn(&str) -> String {
        |name| {
            let mut name = if name.starts_with("decoder.") {
                name.replace("decoder.", "gpt_neox.")
            } else {
                format!("gpt_neox.{name}")
            };

            // Embedding layer.
            name = name.replace("embeddings.piece_embeddings", "embed_in");

            // Attention layer.
            name = name.replace("attention.layer_norm", "input_layernorm");
            name = name.replace(".attention.output", ".attention.dense");
            name = name.replace("qkv", "query_key_value");

            // Feed-forward layer.
            name = name.replace(".ffn.layer_norm", ".post_attention_layernorm");
            name = name.replace(".intermediate", ".dense_h_to_4h");
            name = name.replace(".ffn.output", ".ffn.dense_4h_to_h");
            name = name.replace(".ffn", ".mlp");

            // Layer norm after all layers.
            name = name.replace("output_layer_norm", "final_layer_norm");

            static LAYER_RE: OnceLock<Regex> = OnceLock::new();
            let layer_re =
                LAYER_RE.get_or_init(|| Regex::new(r"layer_(\d+)").expect("Invalid regex"));
            name = layer_re.replace(&name, "layers.$1").to_string();
            name
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};
    use ndarray::array;
    use snafu::{report, FromString, ResultExt, Whatever};

    use crate::architectures::{Decoder, LayerOutputs};
    use crate::kv_cache::KeyValueCache;
    use crate::layers::attention::AttentionMask;
    use crate::models::gpt_neox::GPTNeoXDecoder;
    use crate::models::hf_hub::FromHFHub;
    use crate::util::tests::assert_tensor_eq;

    #[test]
    #[report]
    fn gpt_neox_decoder() -> Result<(), Whatever> {
        let decoder = GPTNeoXDecoder::from_hf_hub(
            "trl-internal-testing/tiny-random-GPTNeoXForCausalLM-safetensors-sharded",
            None,
            Device::Cpu,
        )
        .with_whatever_context(|_| "Cannot load model")?;

        let input = Tensor::arange(0i64, 24, &Device::Cpu)
            .and_then(|t| t.reshape((3, 8)))
            .with_whatever_context(|_| "Cannot create input tensor")?;

        let mask = Tensor::from_slice(
            &[
                1u32, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0,
            ],
            (3, 8),
            &Device::Cpu,
        )
        .with_whatever_context(|_| "Cannot create attention mask tensor")?;
        let mask =
            AttentionMask::new(mask).with_whatever_context(|_| "Cannot create attention mask")?;

        let output = decoder
            .forward_t(&input, &mask, &mut KeyValueCache::no_cache(5), None, false)
            .map_err(|e| Whatever::with_source(e, "Cannot decode input".to_string()))?;

        // TODO: sort out why this does not work.
        //.whatever_context("Cannot decode input")?;

        let summed_states = output
            .layer_outputs()
            .last()
            .unwrap()
            .sum(2)
            .with_whatever_context(|_| "Cannot sum model outputs")?;

        assert_tensor_eq::<f32>(
            summed_states,
            array![
                [
                    3.5763e-07,
                    2.3842e-07,
                    1.3709e-06,
                    7.1526e-07,
                    1.7881e-06,
                    -1.9073e-06,
                    -4.4703e-07,
                    -9.5367e-07
                ],
                [
                    -5.9605e-07,
                    -2.3842e-07,
                    0.0000e+00,
                    -5.5134e-07,
                    -1.1921e-07,
                    8.3447e-07,
                    2.8312e-07,
                    -1.1325e-06
                ],
                [
                    0.0000e+00,
                    4.7684e-07,
                    1.1921e-07,
                    2.3842e-07,
                    5.9605e-07,
                    4.1723e-07,
                    7.1526e-07,
                    -4.7684e-07
                ]
            ],
            1e-4,
        );

        Ok(())
    }
}
