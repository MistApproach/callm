//! This module provides template implementations for different templating engines.

pub mod dummy;
pub use dummy::TemplateDummy;
pub mod jinja;
pub use jinja::TemplateJinja;

use crate::error::CallmError;
use std::fmt;

/// A trait defining the interface for template implementations.
pub trait TemplateImpl {
    /// Returns the beginning-of-sequence (BOS) token.
    fn get_bos_token(&self) -> Option<&str>;

    /// Sets the beginning-of-sequence (BOS) token.
    fn set_bos_token(&mut self, bos_token: Option<String>);

    /// Returns the end-of-sequence (EOS) token.
    fn get_eos_token(&self) -> Option<&str>;

    /// Sets the end-of-sequence (EOS) token.
    fn set_eos_token(&mut self, eos_token: Option<String>);

    /// Applies the template to the given messages and returns the formatted string.
    fn apply(&self, messages: &[(MessageRole, String)]) -> Result<String, CallmError>;
}

/// An enum representing the roles in a message exchange.
#[derive(Clone, Debug, PartialEq)]
pub enum MessageRole {
    /// The system role.
    System,
    /// The user role.
    User,
    /// The assistant role.
    Assistant,
}

impl fmt::Display for MessageRole {
    /// Formats the `MessageRole` enum for display.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
        }
    }
}

