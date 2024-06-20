use super::MessageRole;
use super::TemplateImpl;
use crate::error::CallmError;
use minijinja::{context, render};

#[derive(Clone, Debug, Default)]
pub struct TemplateJinja {
    template: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
    add_generation_prompt: bool,
}

impl TemplateJinja {
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
            add_generation_prompt: true,
            ..Default::default()
        }
    }

    pub fn set_add_generation_prompt(&mut self, add_generation_prompt: bool) {
        self.add_generation_prompt = add_generation_prompt;
    }
}

impl TemplateImpl for TemplateJinja {
    fn apply(&self, messages: &[(MessageRole, String)]) -> Result<String, CallmError> {
        // parse messages into String tuples
        let msgs: Vec<_> = messages
            .iter()
            .map(|(role, content)| context!(role => role.to_string(), content => content.to_string()))
            .collect();

        let bos_token = if let Some(tkn) = &self.bos_token {
            tkn.as_str()
        } else {
            ""
        };

        let eos_token = if let Some(tkn) = &self.eos_token {
            tkn.as_str()
        } else {
            ""
        };

        let output = render!(&self.template, messages => msgs, bos_token, eos_token, add_generation_prompt => self.add_generation_prompt);

        Ok(output)
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
