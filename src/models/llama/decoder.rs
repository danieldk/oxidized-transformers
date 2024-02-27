use std::sync::OnceLock;

use candle_core::Tensor;
use candle_nn::VarBuilder;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::architectures::{BuildDecoder, Decoder, DecoderOutput};
use crate::error::BoxedError;
use crate::kv_cache::KeyValueCache;
use crate::layers::activation::Activation;
use crate::layers::attention::{AttentionHeads, AttentionMask, QkvMode, SelfAttentionConfig};
use crate::layers::embeddings::QueryKeyRotaryEmbeddingsConfig;
use crate::layers::feedforward::PointwiseFeedForwardConfig;
use crate::layers::layer_norm::RMSNormConfig;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerLayerConfig};
use crate::models::hf_hub::{FromHF, TransformerFromConfig};
use crate::models::transformer::{TransformerDecoder, TransformerDecoderConfig};

pub struct LlamaDecoder {
    inner: TransformerDecoder,
}

impl Decoder for LlamaDecoder {
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

impl TransformerFromConfig for LlamaDecoder {
    type Config = TransformerDecoderConfig;

    fn from_config(vb: VarBuilder, config: &Self::Config) -> Result<Self, BoxedError> {
        Ok(Self {
            inner: config.build(vb)?,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HFLlamaDecoderConfig {
    hidden_act: Activation,
    pub(crate) hidden_size: usize,
    initializer_range: f32,
    intermediate_size: usize,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    pub(crate) vocab_size: usize,
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

impl FromHF for LlamaDecoder {
    type HFConfig = HFLlamaDecoderConfig;

    fn rename_parameters() -> impl Fn(&str) -> String {
        |name| {
            let mut name = if name.starts_with("decoder.") {
                name.replace("decoder.", "model.")
            } else if !name.starts_with("output_embeddings") {
                format!("model.{name}")
            } else {
                name.to_string()
            };
            name = name.replace("embeddings.piece_embeddings", "embed_tokens");

            // Attention layer.
            name = name.replace("attention.query", "attention.q_proj");
            name = name.replace("attention.key", "attention.k_proj");
            name = name.replace("attention.value", "attention.v_proj");
            name = name.replace("attention.output", "attention.o_proj");
            name = name.replace("attention.layer_norm", "input_layernorm");
            name = name.replace("attention.", "self_attn.");

            // Feed-forward layer.
            name = name.replace("ffn.layer_norm", "post_attention_layernorm");
            name = name.replace("ffn.output", "ffn.down_proj");
            name = name.replace("ffn.", "mlp.");
            name = name.replace("intermediate", "up_proj");
            name = name.replace("gate", "gate_proj");

            // Layer norm after all layers.
            name = name.replace("output_layer_norm", "norm");

            // Output vocab.
            name = name.replace("output_embeddings", "lm_head");

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
    use candle_core::{DType, Device, Tensor};
    use ndarray::array;
    use snafu::{report, FromString, ResultExt, Whatever};

    use crate::architectures::{Decoder, LayerOutputs};
    use crate::kv_cache::KeyValueCache;
    use crate::layers::attention::AttentionMask;
    use crate::models::hf_hub::FromHFHub;
    use crate::models::llama::LlamaDecoder;
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
    #[report]
    fn llama_decoder_emits_correct_output() -> Result<(), Whatever> {
        let decoder =
            LlamaDecoder::from_hf_hub("explosion-testing/llama2-kv-sharing", None, Device::Cpu)
                .with_whatever_context(|_| "Cannot load model")?;

        let (input, mask) = sample_inputs()?;

        let output = decoder
            .forward_t(
                &input,
                &AttentionMask::new(mask).whatever_context("Cannot build attention mask")?,
                &mut KeyValueCache::no_cache(5),
                None,
                false,
            )
            .map_err(|e| Whatever::with_source(e, "Cannot decode input".to_string()))?;

        let summed_states = output
            .layer_outputs()
            .last()
            .unwrap()
            .sum(2)
            .with_whatever_context(|_| "Cannot sum model outputs")?;

        assert_tensor_eq::<f32>(
            summed_states,
            array![
                [0.0000f32, 12.4694, -3.2950, -8.7229, -4.2094, 4.5980, 1.9919, -0.7404,],
                [17.1286, 16.0776, 17.8116, 12.2431, 12.9443, 12.4742, 11.5011, 8.0226,],
                [-22.4760, -7.3823, -6.9333, -6.1485, -6.1616, -10.4572, -19.8962, -18.7712,],
            ],
            1e-4,
        );

        Ok(())
    }

    #[test]
    #[report]
    fn llama_decoder_give_correct_output_with_cache() -> Result<(), Whatever> {
        let decoder =
            LlamaDecoder::from_hf_hub("explosion-testing/llama2-kv-sharing", None, Device::Cpu)
                .with_whatever_context(|_| "Cannot load model")?;

        let (input, mask) = sample_inputs()?;

        let mut cache =
            KeyValueCache::cache(input.shape().dims()[0], 64, 1, 5, DType::F32, &Device::Cpu)
                .whatever_context("Cannot create cache")?;
        let attention_mask = AttentionMask::new(
            mask.narrow(1, 0, 7)
                .whatever_context("Cannot slice attention mask")?,
        )
        .whatever_context("Cannot build attention mask")?;

        let _ = decoder
            .forward_t(
                &input
                    .narrow(1, 0, 7)
                    .whatever_context("Cannot slice input")?,
                &attention_mask,
                &mut cache,
                None,
                false,
            )
            .map_err(|e| Whatever::with_source(e, "Cannot decode input".to_string()))?;

        let attention_mask = attention_mask
            .extend(
                &AttentionMask::new(
                    mask.narrow(1, 7, 1)
                        .whatever_context("Cannot slice attention mask")?,
                )
                .whatever_context("Cannot build attention mask")?,
            )
            .whatever_context("Cannot extend attention mask")?;

        let output = decoder
            .forward_t(
                &input
                    .narrow(1, 7, 1)
                    .whatever_context("Cannot slice input")?,
                &attention_mask,
                &mut cache,
                None,
                false,
            )
            .map_err(|e| Whatever::with_source(e, "Cannot decode input".to_string()))?;

        let output = output.layer_outputs().last().unwrap();

        let summed_states = output
            .sum(2)
            .with_whatever_context(|_| "Cannot sum model outputs")?;

        assert_tensor_eq::<f32>(
            summed_states,
            array![[-0.74043036], [8.022626], [-18.771225]].into_dyn(),
            1e-4,
        );

        Ok(())
    }
}
