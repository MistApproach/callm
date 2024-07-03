//! This module provides custom Error type.

use std::io;
use thiserror::Error;

/// An enumeration of all possible errors.
#[derive(Error, Debug)]
pub enum CallmError {
    /// A generic error with a custom message.
    #[error("Error: `{0}`")]
    GenericError(String),

    /// An error indicating a failure in the model loader with a custom message.
    #[error("Loader failure: `{0}`")]
    LoaderFail(String),

    /// An error indicating that the model is unsupported.
    #[error("Unsupported model")]
    UnsupportedModel,

    /// An error wrapping an I/O error from the standard library.
    #[error("I/O error")]
    IOError(#[from] io::Error),

    /// An error wrapping a Candle error from the `candle_core` crate.
    #[error("Candle error")]
    CandleError(#[from] candle_core::Error),

    /// An error indicating a failure in the template with a custom message.
    #[error("Template error `{0}`")]
    TemplateError(String),

    /// An error indicating a failure in the tokenizer with a custom message.
    #[error("Tokenizer error")]
    TokenizerError { msg: String },

    /// An error wrapping a serialization/deserialization error from the `serde_json` crate.
    #[error("Serialization/Deserialization error")]
    SerdeError(#[from] serde_json::Error),
}
