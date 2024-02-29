use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::BoxedError;
use crate::layers::activation::Activation;
use crate::layers::attention::{
    AttentionHeads, QkvMode, ScaledDotProductAttentionConfig, SelfAttentionConfig,
};
use crate::layers::dropout::DropoutConfig;
use crate::layers::feedforward::PointwiseFeedForwardConfig;
use crate::layers::layer_norm::LayerNormConfig;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerLayerConfig};
use crate::models::hf::FromHF;
use crate::models::roberta::embeddings::RobertaEmbeddingsConfig;
use crate::models::transformer::{TransformerEncoder, TransformerEncoderConfig};

pub struct RobertaEncoder;

#[derive(Debug, Deserialize, Serialize)]
pub struct HFRobertaEncoderConfig {
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
    pad_token_id: u32,
    type_vocab_size: usize,
    vocab_size: usize,
}

impl TryFrom<HFRobertaEncoderConfig> for TransformerEncoderConfig {
    type Error = BoxedError;

    fn try_from(hf_config: HFRobertaEncoderConfig) -> Result<Self, Self::Error> {
        let attention_probs_dropout =
            Box::new(DropoutConfig::default().p(hf_config.attention_probs_dropout_prob));
        let hidden_dropout = Box::new(DropoutConfig::default().p(hf_config.hidden_dropout_prob));
        let layer_norm = Box::new(
            LayerNormConfig::default()
                .eps(hf_config.layer_norm_eps as f64)
                .size(hf_config.hidden_size),
        );

        let embeddings = RobertaEmbeddingsConfig::default()
            .padding_id(hf_config.pad_token_id)
            .transformer_embeddings(Box::new(
                TransformerEmbeddingsConfig::default()
                    .embedding_dropout(hidden_dropout.clone())
                    .embedding_layer_norm(layer_norm.clone())
                    .embedding_width(hf_config.hidden_size)
                    .hidden_width(hf_config.hidden_size)
                    .n_pieces(hf_config.vocab_size)
                    .n_positions(Some(hf_config.max_position_embeddings))
                    .n_types(Some(hf_config.type_vocab_size)),
            ));

        let attention = SelfAttentionConfig::default()
            .attention_heads(AttentionHeads {
                n_query_heads: hf_config.num_attention_heads,
                n_key_value_heads: hf_config.num_attention_heads,
                qkv_mode: QkvMode::Separate,
            })
            .attention_scorer(Box::new(
                ScaledDotProductAttentionConfig::default().dropout(attention_probs_dropout),
            ))
            .hidden_width(hf_config.hidden_size)
            .layer_norm(layer_norm.clone());

        let feedforward = PointwiseFeedForwardConfig::default()
            .activation(Box::new(hf_config.hidden_act))
            .dropout(hidden_dropout)
            .hidden_width(hf_config.hidden_size)
            .intermediate_width(hf_config.intermediate_size)
            .layer_norm(layer_norm);

        let layer = TransformerLayerConfig::default()
            .attention(attention)
            .feedforward(feedforward);

        Ok(Self::default()
            .embeddings(Box::new(embeddings))
            .layer(Box::new(layer))
            .n_hidden_layers(hf_config.num_hidden_layers))
    }
}

impl FromHF for RobertaEncoder {
    type Config = TransformerEncoderConfig;

    type HFConfig = HFRobertaEncoderConfig;

    type Model = TransformerEncoder;

    fn rename_parameters() -> impl Fn(&str) -> String {
        |name| {
            let mut name = if name.starts_with("encoder.") {
                name.replace("encoder.", "roberta.")
            } else if !name.starts_with("lm_head") {
                format!("roberta.{name}")
            } else {
                name.to_string()
            };

            // Embeddings
            name = name.replace("embeddings.piece_embeddings", "embeddings.word_embeddings");
            name = name.replace(
                "embeddings.type_embeddings",
                "embeddings.token_type_embeddings",
            );
            name = name.replace("embeddings.embedding_layer_norm", "embeddings.LayerNorm");

            // Layers
            name = name.replace("roberta.layer", "roberta.encoder.layer");
            static LAYER_RE: OnceLock<Regex> = OnceLock::new();
            let layer_re =
                LAYER_RE.get_or_init(|| Regex::new(r"layer_(\d+)").expect("Invalid regex"));
            name = layer_re.replace(&name, "layer.$1").to_string();

            // Attention layer.
            name = name.replace("attention.output", "attention.output.dense");
            name = name.replace("query", "self.query");
            name = name.replace("key", "self.key");
            name = name.replace("value", "self.value");
            name = name.replace("attention.layer_norm", "attention.output.LayerNorm");

            // Feed-forward layer.
            name = name.replace("ffn.intermediate", "intermediate.dense");
            name = name.replace("ffn.output", "output.dense");
            name = name.replace("ffn.layer_norm", "output.LayerNorm");

            name
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};
    use ndarray::array;
    use snafu::{report, FromString, ResultExt, Whatever};

    use crate::architectures::{Encoder, LayerOutputs};
    use crate::layers::attention::AttentionMask;
    use crate::models::hf::FromHFHub;
    use crate::models::roberta::encoder::RobertaEncoder;
    use crate::util::tests::assert_tensor_eq;

    fn sample_inputs() -> Result<(Tensor, Tensor), Whatever> {
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

        Ok((input, mask))
    }

    #[test]
    #[ignore]
    #[report]
    fn roberta_encoder_emits_correct_output() -> Result<(), Whatever> {
        let encoder =
            RobertaEncoder::from_hf_hub("explosion-testing/roberta-test", None, Device::Cpu)
                .with_whatever_context(|_| "Cannot load model")?;

        let (input, mask) = sample_inputs()?;

        let output = encoder
            .forward_t(
                &input,
                &AttentionMask::new(mask).whatever_context("Cannot build attention mask")?,
                None,
                false,
            )
            .map_err(|e| Whatever::with_source(e, "Cannot encode input".to_string()))?;

        let summed_states = output.layer_outputs()[0]
            .sum(2)
            .with_whatever_context(|_| "Cannot sum model outputs")?;

        assert_tensor_eq::<f32>(
            summed_states,
            array![
                [
                    -4.7684e-07,
                    -2.3842e-07,
                    7.1526e-07,
                    -2.3842e-07,
                    -8.3447e-07,
                    -3.5763e-07,
                    2.3842e-07,
                    -4.7684e-07
                ],
                [
                    -2.3842e-07,
                    -2.3842e-07,
                    1.1921e-07,
                    5.9605e-07,
                    6.5565e-07,
                    -2.3842e-07,
                    5.9605e-07,
                    -4.7684e-07
                ],
                [
                    0.0000e+00,
                    0.0000e+00,
                    -7.1526e-07,
                    -4.7684e-07,
                    -7.1526e-07,
                    -1.2517e-06,
                    4.7684e-07,
                    -2.2352e-07
                ]
            ],
            1e-4,
        );

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
                    -1.1921e-06,
                    -2.9802e-07,
                    3.5763e-07,
                    -4.7684e-07,
                    -2.3842e-07,
                    0.0000e+00,
                    -4.7684e-07,
                    0.0000e+00
                ],
                [
                    0.0000e+00,
                    -4.7684e-07,
                    -9.5367e-07,
                    1.1921e-07,
                    -1.1921e-07,
                    0.0000e+00,
                    -4.7684e-07,
                    -1.3113e-06
                ],
                [
                    -7.1526e-07,
                    -1.3113e-06,
                    -5.9605e-07,
                    -1.1921e-07,
                    -9.5367e-07,
                    -4.1723e-07,
                    -4.7684e-07,
                    -5.9605e-07
                ]
            ],
            1e-4,
        );

        Ok(())
    }
}
