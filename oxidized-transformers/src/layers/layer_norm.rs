use candle_core::{DType, Tensor, D};
use candle_nn::{Init, Module};

use crate::error::BoxedError;
use crate::layers::module::{BuildModule, ModuleT};
use crate::varbuilder::VarBuilder;

/// Layer norm configuration.
#[derive(Clone, Debug)]
pub struct LayerNormConfig {
    pub affine: bool,
    pub eps: f64,
    pub remove_mean: bool,
    pub size: usize,
}

impl LayerNormConfig {
    /// Whether to use an affine transformation.
    ///
    /// Default: `true`
    pub fn affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    /// Epsilon value.
    ///
    /// Default: `1e-12`
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Whether to remove the mean.
    ///
    /// If the mean is not removed, this layer is equivalent to `RMSNorm`.
    ///
    /// Default: `true`
    pub fn remove_mean(mut self, remove_mean: bool) -> Self {
        self.remove_mean = remove_mean;
        self
    }

    /// Dimensionality of the layer.
    ///
    /// Default: `768`
    pub fn size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            affine: true,
            eps: 1e-12,
            remove_mean: true,
            size: 768,
        }
    }
}

impl BuildModule for LayerNormConfig {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError> {
        Ok(Box::new(layer_norm(
            LayerNormConfig {
                affine: self.affine,
                eps: self.eps,
                remove_mean: self.remove_mean,
                size: self.size,
            },
            vb,
        )?))
    }
}

// Implementations below are from Candle. Copied because we need to use our
// own `VarBuilder`.

/// RMS norm configuration.
#[derive(Clone, Debug)]
pub struct RMSNormConfig {
    pub eps: f64,
    pub size: usize,
}

impl RMSNormConfig {
    /// Epsilon value.
    ///
    /// Default: `1e-12`
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Dimensionality of the layer.
    ///
    /// Default: `768`
    pub fn size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }
}

impl Default for RMSNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-12,
            size: 768,
        }
    }
}

impl BuildModule for RMSNormConfig {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError> {
        Ok(Box::new(rms_norm(self.size, self.eps, vb)?))
    }
}

// This layer norm version handles both weight and bias so removes the mean.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    remove_mean: bool,
    eps: f64,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: Some(bias),
            remove_mean: true,
            eps,
        }
    }

    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: true,
            eps,
        }
    }

    pub fn rms_norm(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: false,
            eps,
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let x = if self.remove_mean {
            let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn layer_norm<C: Into<LayerNormConfig>>(
    config: C,
    vb: VarBuilder,
) -> Result<LayerNorm, candle_core::Error> {
    let config = config.into();
    let weight = vb.get_with_init(config.size, "weight", Init::Const(1.))?;
    let bias = if config.affine {
        Some(vb.get_with_init(config.size, "bias", Init::Const(0.))?)
    } else {
        None
    };
    Ok(LayerNorm {
        weight,
        bias,
        remove_mean: config.remove_mean,
        eps: config.eps,
    })
}

/// RmsNorm is a specialized version of the LayerNorm module.
#[derive(Clone, Debug)]
pub struct RmsNorm(LayerNorm);

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self(LayerNorm::rms_norm(weight, eps))
    }

    pub fn into_inner(self) -> LayerNorm {
        self.0
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.0.forward(xs)
    }
}

pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm, candle_core::Error> {
    let config = LayerNormConfig {
        eps,
        remove_mean: false,
        size: size,
        affine: false,
    };
    Ok(RmsNorm(layer_norm(config, vb)?))
}
