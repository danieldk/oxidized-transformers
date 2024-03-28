//! Embedding Layer.

// Implementation from Candle. Copied because we need to use our own `VarBuilder`.

use candle_core::Tensor;
use candle_nn::{Init, Module};

use crate::varbuilder::VarBuilder;

#[derive(Clone, Debug)]
pub struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    /// Get the hidden size of the embedding matrix
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Module for Embedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

pub fn embedding(
    in_size: usize,
    out_size: usize,
    vb: VarBuilder,
) -> Result<Embedding, candle_core::Error> {
    let embeddings = vb.get_with_init(
        (in_size, out_size),
        "weight",
        Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;
    Ok(Embedding::new(embeddings, out_size))
}
