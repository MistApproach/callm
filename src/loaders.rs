//! This module provides loaders for different model formats.

pub mod gguf;
pub use gguf::LoaderGguf;
pub mod safetensors;
pub use safetensors::LoaderSafetensors;

use crate::device::DeviceConfig;
use crate::error::CallmError;
use crate::models::ModelImpl;
use crate::templates::TemplateImpl;
use std::sync::{Arc, Mutex};
use tokenizers::tokenizer::Tokenizer;

/// A trait for defining the interface of model loaders.
pub trait LoaderImpl: Send {
    /// Sets the device configuration for the loader.
    fn set_device(&mut self, device: Arc<DeviceConfig>);

    /// Loads the model and returns it wrapped in an `Arc<Mutex<dyn ModelImpl>>`.
    fn load(&mut self) -> Result<Arc<Mutex<dyn ModelImpl>>, CallmError>;

    /// Returns the tokenizer associated with the model.
    fn tokenizer(&mut self) -> Result<Tokenizer, CallmError>;

    /// Returns the template associated with the model.
    fn template(&mut self) -> Result<Box<dyn TemplateImpl>, CallmError>;
}

