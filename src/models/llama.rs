use super::ModelImpl;
use crate::{device::DeviceConfig, error::CallmError};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama as Model};
use std::path::Path;
use std::sync::Arc;

const USE_KV_CACHE: bool = true;

pub struct ModelLlama {
    model: Model,
    cache: Cache,
    config: Config,
    device: Arc<DeviceConfig>,
}

impl ModelLlama {
    pub fn from_paths<P: AsRef<Path>>(
        paths: &[P],
        config: &Config,
        device: Arc<DeviceConfig>,
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
            model: Model::load(vb, config)?,
            cache: Self::spawn_kv_cache(config, &device)?,
            config: config.clone(),
            device,
        })
    }

    fn spawn_kv_cache(config: &Config, device: &DeviceConfig) -> Result<Cache, CallmError> {
        let cache = Cache::new(
            USE_KV_CACHE,
            device.candle_dtype(),
            config,
            device.candle_device(),
        )?;

        Ok(cache)
    }
}

impl ModelImpl for ModelLlama {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor, CallmError> {
        Ok(self.model.forward(input, index_pos, &mut self.cache)?)
    }

    fn clear_kv_cache(&mut self) -> Result<(), CallmError> {
        self.cache = Self::spawn_kv_cache(&self.config, &self.device)?;
        Ok(())
    }
}
