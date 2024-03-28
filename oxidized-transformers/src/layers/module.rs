use std::fmt::Debug;

use candle_core::{ModuleT as CandleModuleT, Tensor};
use snafu::ResultExt;

use crate::error::BoxedError;
use crate::varbuilder::VarBuilder;

/// Module in a trainable model.
///
/// This trait is similar to [candle_core::ModuleT], but uses an opaque
/// (boxed) error.
pub trait ModuleT {
    /// Apply the module to the input.
    ///
    /// - `xs`: Input tensor.
    /// - `train`: Whether the module is used in training.
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor, BoxedError>;
}

impl<M> ModuleT for M
where
    M: CandleModuleT,
{
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor, BoxedError> {
        (self as &dyn CandleModuleT).forward_t(xs, train).boxed()
    }
}

/// Traits for types that can build modules.
pub trait BuildModule: Debug {
    /// Build a module.
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError>;
}
