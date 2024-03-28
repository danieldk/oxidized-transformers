use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init, VarBuilder as CandleVarBuilder};

mod quantized;
pub use quantized::VarBuilder as QVarBuilder;

#[derive(Clone)]
pub enum VarBuilder<'a> {
    NonQuantized(CandleVarBuilder<'a>),
    Quantized(QVarBuilder),
}

impl<'a> VarBuilder<'a> {
    /// Construct a new `VarBuilder` from a backend.
    pub fn from_backend(
        backend: Box<dyn SimpleBackend + 'a>,
        dtype: DType,
        device: Device,
    ) -> Self {
        VarBuilder::NonQuantized(CandleVarBuilder::from_backend(backend, dtype, device))
    }

    /// Create a `VarBuilder` that uses zeros for any tensor.
    pub fn zeros(dtype: DType, dev: &Device) -> Self {
        VarBuilder::NonQuantized(CandleVarBuilder::zeros(dtype, dev))
    }

    /// Get the default data type for the varbuilder.
    pub fn dtype(&self) -> DType {
        match self {
            VarBuilder::NonQuantized(vb) => vb.dtype(),
            VarBuilder::Quantized(_vb) => DType::F32,
        }
    }

    /// Returns a new `VarBuilder` with the prefix set to `prefix`.
    pub fn push_prefix<S: ToString>(&self, s: S) -> Self {
        match self {
            VarBuilder::NonQuantized(vb) => VarBuilder::NonQuantized(vb.push_prefix(s)),
            VarBuilder::Quantized(vb) => VarBuilder::Quantized(vb.push_prefix(s)),
        }
    }

    /// The device used by default.
    pub fn device(&self) -> &Device {
        match self {
            VarBuilder::NonQuantized(vb) => vb.device(),
            VarBuilder::Quantized(vb) => vb.device(),
        }
    }

    /// Retrieve the tensor.
    ///
    /// - `s` - Shape of the tensor.
    /// - `name` - Name of the tensor.
    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Tensor, candle_core::Error> {
        self.get_with_init(s, name, Default::default())
    }

    /// Retrieve the tensor.
    ///
    /// Initialize the tensor with the given initialization if it is fresh.
    ///
    /// - `s` - Shape of the tensor.
    /// - `name` - Name of the tensor.
    /// - `init` - Initialization method.
    pub fn get_with_init<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        init: Init,
    ) -> Result<Tensor, candle_core::Error> {
        match self {
            VarBuilder::NonQuantized(vb) => vb.get_with_hints(s, name, init),
            VarBuilder::Quantized(vb) => vb.get(s, name).and_then(|xs| xs.dequantize(vb.device())),
        }
    }
}
