use super::ModelImpl;
use crate::{device::DeviceConfig, error::CallmError};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM as Model};
use std::path::Path;

pub struct ModelQwen2 {
    model: Model,
}

impl ModelQwen2 {
    pub fn from_paths<P: AsRef<Path>>(
        paths: &[P],
        config: &Config,
        device: DeviceConfig,
    ) -> Result<Self, CallmError> {
        // NOTE: unsafe inherited from memmap2::MmapOptions
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                paths,
                device.candle_dtype(),
                device.candle_device(),
            )?
        };

        Ok(Self {
            model: Model::new(config, vb)?,
        })
    }
}

impl ModelImpl for ModelQwen2 {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor, CallmError> {
        Ok(self.model.forward(input, index_pos)?)
    }

    fn clear_kv_cache(&mut self) -> Result<(), CallmError> {
        self.model.clear_kv_cache();
        Ok(())
    }
}
