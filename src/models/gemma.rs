use super::ModelImpl;
use crate::{device::DeviceConfig, error::CallmError};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::gemma::{Config, Model};
use std::path::Path;
use std::sync::Arc;

pub struct ModelGemma {
    model: Model,
}

impl ModelGemma {
    pub fn from_paths<P: AsRef<Path>>(
        paths: &[P],
        config: &Config,
        device: Arc<DeviceConfig>,
        use_flash_attention: bool,
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
            model: Model::new(use_flash_attention, config, vb)?,
        })
    }
}

impl ModelImpl for ModelGemma {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor, CallmError> {
        Ok(self.model.forward(input, index_pos)?)
    }

    fn clear_kv_cache(&mut self) -> Result<(), CallmError> {
        self.model.clear_kv_cache();
        Ok(())
    }
}
