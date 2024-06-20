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

#[derive(Clone, Debug, Default)]
pub enum ModelArchitecture {
    #[default]
    Unsupported,
    Llama,
    LlamaQuantized,
    Mistral,
    Phi3,
    Qwen2,
}

pub trait ModelImpl {
    fn load(&mut self) -> Result<(), CallmError> {
        Ok(())
    }

    fn unload(&mut self) -> Result<(), CallmError> {
        Ok(())
    }

    fn forward(&mut self, _input: &Tensor, _index_pos: usize) -> Result<Tensor, CallmError> {
        unimplemented!()
    }

    fn clear_kv_cache(&mut self) -> Result<(), CallmError> {
        Ok(())
    }
}
