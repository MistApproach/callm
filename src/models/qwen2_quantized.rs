use super::ModelImpl;
use crate::device::DeviceConfig;
use crate::error::CallmError;
use candle_core::quantized::gguf_file::Content;
use candle_core::Tensor;
use candle_transformers::models::quantized_qwen2::ModelWeights as Model;
use std::io::{Read, Seek};
use std::sync::Arc;

pub struct ModelQwen2Quantized {
    model: Model,
}

impl ModelQwen2Quantized {
    pub fn from_weights(model: Model) -> Self {
        Self { model }
    }

    pub fn from_gguf<R>(
        content: Content,
        reader: &mut R,
        device: Arc<DeviceConfig>,
    ) -> Result<Self, CallmError>
    where
        R: Seek + Read,
    {
        Ok(Self {
            model: Model::from_gguf(content, reader, device.candle_device())?,
        })
    }
}

impl ModelImpl for ModelQwen2Quantized {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor, CallmError> {
        self.model
            .forward(input, index_pos)
            .map_err(CallmError::CandleError)
    }
}
