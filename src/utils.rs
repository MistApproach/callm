//! This module provides utility functions.

use crate::error::CallmError;
use crate::loaders::{LoaderGguf, LoaderImpl, LoaderSafetensors};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Attempts to determine the appropriate model loader for a given file or directory path.
///
/// This function checks the file extension or directory structure to decide which loader to use.
/// If the path points to a file, it will check the file extension to determine the loader.
/// If the path points to a directory, it will default to using the `LoaderSafetensors`.
///
/// # Arguments
///
/// * `path` - A string slice that represents the path to the model file or directory.
///
/// # Returns
///
/// * `Ok(Arc<Mutex<dyn LoaderImpl>>)` - If a suitable loader is found, it returns the loader wrapped in an `Arc<Mutex<dyn LoaderImpl>>`.
/// * `Err(CallmError)` - If no suitable loader is found, it returns an error of type `CallmError`.
///
/// # Errors
///
/// This function will return an error if:
/// * The path does not exist.
/// * The file extension is not recognized.
/// * The function fails to extract or convert the file extension.
pub fn autodetect_loader(path: &str) -> Result<Arc<Mutex<dyn LoaderImpl>>, CallmError> {
    // Get path metadata
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
                return Ok(Arc::new(Mutex::new(LoaderGguf::new(path))));
            }
            "ggml" => todo!("GGML format is not supported, yet."),
            "safetensors" => return Ok(Arc::new(Mutex::new(LoaderSafetensors::new(path)))),
            _ => {
                // As a last resort, try pointing the loader to the parent directory
                if let Some(parent) = pthbuf.parent() {
                    return autodetect_loader(parent.to_str().unwrap());
                }
            }
        }
    } else {
        return Ok(Arc::new(Mutex::new(LoaderSafetensors::new(path))));
    }

    Err(CallmError::LoaderFail(
        "No suitable loader found".to_string(),
    ))
}

