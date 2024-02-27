use candle_core::Tensor;
use candle_nn::VarBuilder;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::architectures::{BuildCausalLM, CausalLM, CausalLMOutput};
use crate::error::BoxedError;
use crate::kv_cache::KeyValueCache;
use crate::layers::attention::AttentionMask;
use crate::models::hf_hub::{FromHFHub, HFRenames, TransformerFromConfig};
use crate::models::llama::decoder::HFLlamaDecoderConfig;
use crate::models::transformer::{
    TransformerCausalLM, TransformerCausalLMConfig, TransformerDecoderConfig,
};

pub struct LlamaCausalLM {
    inner: TransformerCausalLM,
}

impl CausalLM for LlamaCausalLM {
    type Cache = KeyValueCache;

    fn forward_t(
        &self,
        piece_ids: &Tensor,
        mask: &AttentionMask,
        cache: &mut Self::Cache,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<CausalLMOutput, BoxedError> {
        self.inner
            .forward_t(piece_ids, mask, cache, positions, train)
    }
}

impl TransformerFromConfig for LlamaCausalLM {
    type Config = TransformerCausalLMConfig;

    fn from_config(vb: VarBuilder, config: &Self::Config) -> Result<Self, BoxedError> {
        Ok(Self {
            inner: config.build(vb)?,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HfLlamaCausalLMConfig {
    #[serde(flatten)]
    decoder: HFLlamaDecoderConfig,
}

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

impl FromHFHub for LlamaCausalLM {
    type HFConfig = HfLlamaCausalLMConfig;
}

impl HFRenames for LlamaCausalLM {
    fn hf_renames() -> impl Fn(&str) -> String {
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
    use candle_core::{Device, Tensor};
    use ndarray::array;
    use snafu::{report, FromString, ResultExt, Whatever};

    use crate::architectures::CausalLM;
    use crate::kv_cache::KeyValueCache;
    use crate::layers::attention::AttentionMask;
    use crate::models::hf_hub::FromHFHub;
    use crate::models::llama::causal_lm::LlamaCausalLM;
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
    fn llama_causal_lm_emits_correct_output() -> Result<(), Whatever> {
        let causal_lm =
            LlamaCausalLM::from_hf_hub("explosion-testing/llama2-kv-sharing", None, Device::Cpu)
                .whatever_context("Cannot load model")?;

        let (input, mask) = sample_inputs()?;

        let output = causal_lm
            .forward_t(
                &input,
                &AttentionMask::new(mask).unwrap(),
                &mut KeyValueCache::no_cache(5),
                None,
                false,
            )
            .map_err(|e| Whatever::with_source(e, "Cannot decode input".to_string()))?;

        let logits = output.logits();
        let max = logits.max(2).whatever_context("Cannot find max logits")?;

        assert_tensor_eq::<f32>(
            max,
            array![
                [0.0000, 1.0400, 1.2941, 1.2213, 1.2502, 1.2024, 1.2425, 1.1819],
                [0.8718, 0.8682, 0.8892, 1.1047, 1.1033, 0.9681, 0.9128, 0.8966],
                [0.8555, 1.1557, 1.1066, 1.0975, 1.1119, 0.9627, 0.8936, 0.9689]
            ],
            1e-4,
        );

        Ok(())
    }
}
