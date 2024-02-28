use candle_core::Tensor;
use candle_nn::VarBuilder;
use std::fmt::Debug;

use crate::error::BoxedError;

pub trait BuildEmbeddings: Debug {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn Embeddings>, BoxedError>;
}

pub trait Embeddings {
    fn forward(
        &self,
        piece_ids: &Tensor,
        train: bool,
        positions: Option<&Tensor>,
        type_ids: Option<&Tensor>,
    ) -> Result<Tensor, BoxedError>;
}
