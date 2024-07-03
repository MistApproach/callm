//! This module provides various model implementations for different architectures.

pub mod llama;
pub use llama::ModelLlama;
pub mod llama_quantized;
pub use llama_quantized::ModelLlamaQuantized;
pub mod mistral;
pub use mistral::ModelMistral;
pub mod phi3;
pub use phi3::ModelPhi3;
pub mod qwen2;
pub use qwen2::ModelQwen2;

use crate::error::CallmError;
use candle_core::Tensor;

/// Enum representing different model architectures supported by the system.
#[derive(Clone, Debug, Default)]
pub enum ModelArchitecture {
    /// Default value for unsupported architectures.
    #[default]
    Unsupported,
    Llama,
    LlamaQuantized,
    Mistral,
    Phi3,
    Qwen2,
}

/// A trait defining the interface for model implementations.
pub trait ModelImpl: Send {
    /// Loads the model.
    ///
    /// # Returns
    /// - `Ok(())` if the model is successfully loaded.
    /// - `Err(CallmError)` if an error occurs during loading.
    fn load(&mut self) -> Result<(), CallmError> {
        Ok(())
    }

    /// Unloads the model.
    ///
    /// # Returns
    /// - `Ok(())` if the model is successfully unloaded.
    /// - `Err(CallmError)` if an error occurs during unloading.
    fn unload(&mut self) -> Result<(), CallmError> {
        Ok(())
    }

    /// Performs a forward pass through the model.
    ///
    /// # Arguments
    /// - `input`: The input tensor to the model.
    /// - `index_pos`: The position index for the input tensor.
    ///
    /// # Returns
    /// - `Ok(Tensor)` with the output tensor if the forward pass is successful.
    /// - `Err(CallmError)` if an error occurs during the forward pass.
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor, CallmError>;

    /// Clears the key-value cache of the model.
    ///
    /// # Returns
    /// - `Ok(())` if the cache is successfully cleared.
    /// - `Err(CallmError)` if an error occurs during cache clearing.
    fn clear_kv_cache(&mut self) -> Result<(), CallmError> {
        Ok(())
    }
}

