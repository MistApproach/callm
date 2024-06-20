/// Dummy template that does not process text in any way
/// Only first message content is taken as input
use super::MessageRole;
use super::TemplateImpl;
use crate::error::CallmError;

#[derive(Clone, Debug, Default)]
pub struct TemplateDummy {
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl TemplateDummy {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
}

impl TemplateImpl for TemplateDummy {
    fn apply(&self, messages: &[(MessageRole, String)]) -> Result<String, CallmError> {
        if messages.is_empty() {
            return Err(CallmError::TemplateError(
                "Error applying template (no messages passed in)".to_string(),
            ));
        }

        Ok(messages[0].1.clone())
    }

    fn get_bos_token(&self) -> Option<&str> {
        if let Some(bos) = &self.bos_token {
            return Some(bos.as_str());
        }
        None
    }

    fn set_bos_token(&mut self, bos_token: Option<String>) {
        self.bos_token = bos_token;
    }

    fn get_eos_token(&self) -> Option<&str> {
        if let Some(eos) = &self.eos_token {
            return Some(eos.as_str());
        }
        None
    }

    fn set_eos_token(&mut self, eos_token: Option<String>) {
        self.eos_token = eos_token;
    }
}
