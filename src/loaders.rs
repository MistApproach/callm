pub mod gguf;
pub use gguf::LoaderGguf;
pub mod safetensors;
pub use safetensors::LoaderSafetensors;

use crate::device::DeviceConfig;
use crate::error::CallmError;
use crate::models::ModelImpl;
use crate::templates::TemplateImpl;
use tokenizers::tokenizer::Tokenizer;

pub trait LoaderImpl {
    fn set_device(&mut self, device: Option<DeviceConfig>);
    fn load(&mut self) -> Result<Box<dyn ModelImpl + Send>, CallmError>;
    fn tokenizer(&mut self) -> Result<Tokenizer, CallmError>;
    fn template(&mut self) -> Result<Box<dyn TemplateImpl>, CallmError>;
}
