use candle_core::{Device, Tensor};
use snafu::{OptionExt, ResultExt, Snafu};

use crate::layers::attention::AttentionMask;

#[derive(Debug, Snafu)]
pub enum PiecesWithIdsError {
    #[snafu(display("Cannot calculat maximum sequence length"))]
    MaxLength,

    #[snafu(display("Cannot create padded tensor"))]
    PaddedTensor { source: candle_core::Error },
}

/// Encoded output of tokenizers.
#[derive(Debug, Clone, Default)]
pub struct PiecesWithIds<D> {
    /// Piece identifiers of each input sequence.
    pub ids: Vec<Vec<D>>,
    /// Piece strings of each input sequence.
    pub pieces: Vec<Vec<String>>,
}

impl PiecesWithIds<u32> {
    /// Generate a padded tensor of the piece identifiers.
    ///
    /// * padding_id - Piece identifier of the padding piece. The actual identifier
    ///   generally doesn't matter when an attention mask is used (and
    ///   as long as it is a valid vocabulary index).
    /// * pad_left - When `false`, sequences shorter than the longest sequence are
    ///   right-padded. Otherwise, sequences are left-padding..
    /// * device - Device on which the padded tensor is created.
    ///
    /// Returns: The padded piece ids.
    /// *Shape:* ``(batch_size, max_seq_len)``
    pub fn padded_tensor(
        &self,
        padding_id: u32,
        pad_left: bool,
        device: &Device,
    ) -> Result<Tensor, PiecesWithIdsError> {
        todo!();
        // let n_sequences = self.ids.len();
        // let max_len = self
        //     .ids
        //     .iter()
        //     .map(|ids| ids.len())
        //     .max()
        //     .context(MaxLengthSnafu)?;
        // let mask = Tensor::full(0u32, (n_sequences, max_len), device).context(PaddedTensorSnafu)?;

        // self.ids.iter().enumerate().fold(Ok(mask), |m, (i, ids)| {
        //     let len = ids.len();
        //     let ids = Tensor::new(ids.as_slice(), device).context(PaddedTensorSnafu)?;
        //     if pad_left {
        //         m.and_then(|m| {
        //             m.slice_assign(&(i, -len..), &ids)
        //                 .context(PaddedTensorSnafu)
        //         })
        //     } else {
        //         m.and_then(|m| m.slice_assign((i, ..len), &ids).context(PaddedTensorSnafu))
        //     }
        // })
    }

    pub fn attention_mask(
        &self,
        pad_left: bool,
        device: &Device,
    ) -> Result<AttentionMask, PiecesWithIdsError> {
        todo!();
        // let n_sequences = self.ids.len();
        // let max_len = self
        //     .ids
        //     .iter()
        //     .map(|ids| ids.len())
        //     .max()
        //     .context(AttentionMaskSnafu)?;
        // let mask = Tensor::full(
        //     Tensor::new(0u32, device).context(AttentionMaskSnafu)?,
        //     (n_sequences, max_len),
        //     device,
        // )
        // .context(AttentionMaskSnafu)?;

        // let true_scalar = Tensor::new(1u32, device).context(AttentionMaskSnafu)?;
        // let mask = self.ids.iter().enumerate().fold(mask, |m, (i, ids)| {
        //     let len = ids.len();
        //     if pad_left {
        //         mask.slice_assign((i, -len..), &true_scalar)
        //             .context(AttentionMaskSnafu)?;
        //     } else {
        //         mask.slice_assign((i, ..len), &true_scalar)
        //             .context(AttentionMaskSnafu)?;
        //     }
        //     mask
        // });

        // Ok(AttentionMask::new(mask))
    }
}
