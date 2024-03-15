use candle_core::{DType, Tensor, D};
use snafu::{ensure, ResultExt, Snafu};

/// Errors for attention masks.
#[derive(Debug, Snafu)]
pub enum AttentionMaskError {
    #[snafu(display("Cannot concatenate masks"))]
    ConcatMasks { source: candle_core::Error },

    #[snafu(display("Attention mask must be 2D, was {}D", n_dims))]
    InvalidDims { n_dims: usize },
}

/// Attention mask.
///
/// Sequence elements for which the corresponding mask element is set to
/// `False` are ignored during attention calculation. Guaranteed to be
/// a 2D array.
#[derive(Clone, Debug)]
pub struct AttentionMask {
    pub(crate) bool_mask: Tensor,
}

impl AttentionMask {
    /// Create an input attention mask.
    ///
    /// * `bool_mask` - Boolean mask tensor.
    ///   *Shape:* `(batch_size, seq_len)`
    pub fn new(bool_mask: Tensor) -> Result<Self, AttentionMaskError> {
        let n_dims = bool_mask.dims().len();
        ensure!(n_dims == 2, InvalidDimsSnafu { n_dims });
        Ok(AttentionMask { bool_mask })
    }

    /// Boolean mask tensor.
    /// *Shape:* `(batch_size, seq_len)`
    pub fn bool_mask(&self) -> &Tensor {
        &self.bool_mask
    }

    /// Extend the mask using another mask.
    pub fn extend(&self, other: &Self) -> Result<Self, AttentionMaskError> {
        Ok(AttentionMask {
            bool_mask: Tensor::cat(&[&self.bool_mask, &other.bool_mask], 1)
                .context(ConcatMasksSnafu)?,
        })
    }

    /// Get the indices of pieces that are not masked.
    pub fn unmasked_indices(&self) -> Result<Tensor, candle_core::Error> {
        let flat_mask = self.bool_mask.flatten_all()?.to_dtype(DType::I64)?;

        let piece_indices = Tensor::arange(0, flat_mask.dim(0)? as i64, self.bool_mask.device())?
            .mul(&flat_mask)?;

        let target_indices = flat_mask
            .to_dtype(DType::F32)?
            .cumsum(D::Minus1)?
            .to_dtype(DType::I64)?
            .broadcast_sub(&Tensor::try_from(1i64)?)?
            .mul(&flat_mask)?;

        Tensor::zeros(
            flat_mask.sum_all()?.to_scalar::<i64>()? as usize,
            DType::I64,
            self.bool_mask.device(),
        )?
        .index_add(&target_indices, &piece_indices, 0)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use crate::layers::attention::AttentionMask;

    #[test]
    fn test_indices() {
        let mask = AttentionMask::new(
            Tensor::from_vec(vec![0u32, 1, 1, 1, 0, 1], &[2, 3], &Device::Cpu).unwrap(),
        )
        .unwrap();
        let indices = mask.unmasked_indices().unwrap();
        eprintln!("indices: {:?}", indices);
        assert!(false);
    }
}
