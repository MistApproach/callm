use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CallmError {
    #[error("Error: `{0}`")]
    GenericError(String),
    #[error("Loader failure: `{0}`")]
    LoaderFail(String),
    #[error("Unsupported model")]
    UnsupportedModel,
    #[error("I/O error")]
    IOError(#[from] io::Error),
    #[error("Candle error")]
    CandleError(#[from] candle_core::Error),
    #[error("Template error `{0}`")]
    TemplateError(String),
    #[error("Tokenizer error")]
    TokenizerError { msg: String },
    #[error("Serialization/Deserialization error")]
    SerdeError(#[from] serde_json::Error),
}
