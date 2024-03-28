use candle_core::Tensor;

use crate::error::BoxedError;
use crate::layers::module::{BuildModule, ModuleT};
use crate::varbuilder::VarBuilder;

/// Identity module.
///
/// This module passes through input as-is. It is especially useful in
/// cases where a configurable module (such as dropout or normalization)
/// needs to be stubbed with a module that does not do anything.
#[derive(Clone, Debug)]
pub struct Identity;

impl BuildModule for Identity {
    fn build(&self, _vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError> {
        Ok(Box::new(Identity))
    }
}

impl ModuleT for Identity {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor, BoxedError> {
        Ok(xs.clone())
    }
}
