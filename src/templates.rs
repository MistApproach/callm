pub mod dummy;
pub use dummy::TemplateDummy;
pub mod jinja;
pub use jinja::TemplateJinja;

use crate::error::CallmError;
use std::fmt;

pub trait TemplateImpl {
    fn get_bos_token(&self) -> Option<&str>;
    fn set_bos_token(&mut self, bos_token: Option<String>);
    fn get_eos_token(&self) -> Option<&str>;
    fn set_eos_token(&mut self, eos_token: Option<String>);
    fn apply(&self, messages: &[(MessageRole, String)]) -> Result<String, CallmError>;
}

#[derive(Clone, Debug, PartialEq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl fmt::Display for MessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
        }
    }
}
