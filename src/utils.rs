use crate::error::CallmError;
use crate::loaders::{LoaderGguf, LoaderImpl, LoaderSafetensors};
use std::fs;
use std::path::PathBuf;

/// Guess suitable model loader for a given path
pub fn autodetect_loader(path: &str) -> Result<Box<dyn LoaderImpl>, CallmError> {
    // get path metadata
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        let pthbuf = PathBuf::from(path);
        match pthbuf
            .as_path()
            .extension()
            .expect("Unable to extract file extension name")
            .to_str()
            .expect("Unable to convert file extension name")
        {
            "gguf" => {
                return Ok(Box::new(LoaderGguf::new(path)));
            }
            "ggml" => todo!("GGML format is not supported, yet."),
            "safetensors" => return Ok(Box::new(LoaderSafetensors::new(path))),
            _ => {}
        }
    } else {
        return Ok(Box::new(LoaderSafetensors::new(path)));
    }

    Err(CallmError::LoaderFail(
        "No suitable loader found".to_string(),
    ))
}
