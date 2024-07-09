use super::LoaderImpl;
use crate::device::DeviceConfig;
use crate::error::CallmError;
use crate::models::{
    ModelArchitecture, ModelGemma, ModelImpl, ModelLlama, ModelMistral, ModelPhi3, ModelQwen2,
};
use crate::templates::{TemplateDummy, TemplateImpl, TemplateJinja};
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

const USE_FLASH_ATTN: bool = false;
const DEFAULT_MODEL_SAFETENSORS_FILE: &str = "model.safetensors";
const DEFAULT_MODEL_INDEX_JSON: &str = "model.safetensors.index.json";
const DEFAULT_MODEL_CONFIG_JSON: &str = "config.json";
const DEFAULT_MODEL_TOKENIZER_JSON: &str = "tokenizer.json";
const DEFAULT_MODEL_TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

#[derive(Debug, Default)]
pub struct LoaderSafetensors {
    location: PathBuf,
    base_dir: PathBuf,
    model_files: Vec<PathBuf>,
    config_path: PathBuf,
    tokenizer_path: PathBuf,
    config: Value,
    device: Arc<DeviceConfig>,
    architecture: ModelArchitecture,
    bos_token_id: Option<i64>,
    eos_token_id: Option<i64>,
    chat_template: Option<String>,
}

impl LoaderSafetensors {
    pub fn new(location: &str) -> Self {
        Self {
            location: PathBuf::from(location),
            ..Default::default()
        }
    }

    fn validate_location(&mut self) -> Result<(), CallmError> {
        let metadata = fs::metadata(&self.location)?;
        // populate base_dir & model files vec
        match metadata.is_file() {
            true => {
                // location pointing to a single safetensors file
                self.model_files.push(self.location.clone());
                // find base dir
                let mut base = self.location.clone();
                base.pop();
                self.base_dir = base;
            }
            false => {
                // location pointing to a directory
                self.base_dir.clone_from(&self.location);
                // look for safetensors model index
                let mut model_index = self.base_dir.clone();
                model_index.push(DEFAULT_MODEL_INDEX_JSON);
                // TODO: replace with std::fs::try_exists once it stabilizes
                if fs::metadata(&model_index).is_ok() {
                    // read model index
                    self.model_files = read_model_index_json(&model_index)?
                        .iter()
                        .map(|p| {
                            let mut o = PathBuf::from(&self.base_dir);
                            o.push(p);
                            o
                        })
                        .collect();
                } else {
                    // try default safetensors model filename
                    self.location.clone_from(&self.base_dir);
                    self.location.push(DEFAULT_MODEL_SAFETENSORS_FILE);
                    return self.validate_location();
                }
            }
        };

        // check model config
        self.config_path = {
            let mut p = PathBuf::from(&self.base_dir);
            p.push(DEFAULT_MODEL_CONFIG_JSON);
            p
        };
        let metadata = fs::metadata(&self.config_path)?;
        if !metadata.is_file() {
            return Err(CallmError::LoaderFail(
                "Unable to find model config file".to_string(),
            ));
        }

        // check tokenizer config
        self.tokenizer_path = {
            let mut p = PathBuf::from(&self.base_dir);
            p.push(DEFAULT_MODEL_TOKENIZER_JSON);
            p
        };
        let metadata = fs::metadata(&self.tokenizer_path)?;
        if !metadata.is_file() {
            return Err(CallmError::LoaderFail(
                "Unable to find model config file".to_string(),
            ));
        }

        Ok(())
    }

    fn load_config(&mut self) -> Result<(), CallmError> {
        // deserialize model config
        let file = fs::File::open(&self.config_path)?;
        let reader = io::BufReader::new(file);
        self.config = serde_json::from_reader(reader)?;
        let config_map = self.config.as_object().ok_or(CallmError::LoaderFail(
            "Unknown model config format".to_string(),
        ))?;

        // determine BOS token
        self.bos_token_id = Some(
            config_map
                .get("bos_token_id")
                .ok_or(CallmError::LoaderFail(
                    "Missing BOS token ID in model config".to_string(),
                ))?
                .as_i64()
                .ok_or(CallmError::LoaderFail(
                    "Model config BOS token ID is not an integer".to_string(),
                ))?,
        );

        // determine EOS token
        self.eos_token_id = Some(
            config_map
                .get("eos_token_id")
                .ok_or(CallmError::LoaderFail(
                    "Missing EOS token ID in model config".to_string(),
                ))?
                .as_i64()
                .ok_or(CallmError::LoaderFail(
                    "Model config EOS token ID is not an integer".to_string(),
                ))?,
        );

        // determine model architecture
        self.architecture = match config_map
            .get("architectures")
            .ok_or(CallmError::LoaderFail(
                "Missing architecture in model config".to_string(),
            ))?
            .as_array()
            .ok_or(CallmError::LoaderFail(
                "Model config architectures is not an array".to_string(),
            ))?
            .first()
            .ok_or(CallmError::LoaderFail(
                "Empty architectures array in model config".to_string(),
            ))?
            .as_str()
            .ok_or(CallmError::LoaderFail(
                "Model architecture in model config is not a string".to_string(),
            ))? {
            "Gemma2ForCausalLM" => ModelArchitecture::Gemma,
            "LlamaForCausalLM" => {
                if let Some(eos_token_id) = &self.eos_token_id {
                    if *eos_token_id == 128001 {
                        log::debug!("Applying Meta Llama EOS token fix");
                        self.eos_token_id = Some(128009);
                    }
                }
                ModelArchitecture::Llama
            }
            "MistralForCausalLM" => ModelArchitecture::Mistral,
            "Phi3ForCausalLM" => ModelArchitecture::Phi3,
            "Qwen2ForCausalLM" => ModelArchitecture::Qwen2,
            _ => ModelArchitecture::Unsupported,
        };

        // search tokenizer config JSON for chat template
        let tokenizer_config_path = {
            let mut p = PathBuf::from(&self.base_dir);
            p.push(DEFAULT_MODEL_TOKENIZER_CONFIG_JSON);
            p
        };
        if let Ok(f) = fs::File::open(tokenizer_config_path) {
            let mut tokenizer_config_bufreader = io::BufReader::new(f);
            #[derive(Deserialize)]
            pub struct ChatTemplate {
                chat_template: String,
            }
            if let Ok(v) =
                serde_json::from_reader::<_, ChatTemplate>(&mut tokenizer_config_bufreader)
            {
                self.chat_template = Some(v.chat_template);
                log::debug!("Loaded chat template from tokenizer config");
            }
        } else {
            log::debug!("Tokenizer config not found, running without chat template");
        }

        Ok(())
    }

    fn load_model(&mut self) -> Result<Arc<Mutex<dyn ModelImpl>>, CallmError> {
        let model: Arc<Mutex<dyn ModelImpl>> = match self.architecture {
            ModelArchitecture::Gemma => {
                use candle_transformers::models::gemma::Config;
                let config: Config = serde_json::from_value(self.config.clone())?;
                Arc::new(Mutex::new(ModelGemma::from_paths(
                    &self.model_files,
                    &config,
                    Arc::clone(&self.device),
                    USE_FLASH_ATTN,
                )?))
            }
            ModelArchitecture::Llama => {
                use candle_transformers::models::llama::LlamaConfig;
                let config: LlamaConfig = serde_json::from_value(self.config.clone())?;
                Arc::new(Mutex::new(ModelLlama::from_paths(
                    &self.model_files,
                    &config.into_config(USE_FLASH_ATTN),
                    Arc::clone(&self.device),
                )?))
            }
            ModelArchitecture::Mistral => {
                use candle_transformers::models::mistral::Config;
                let config: Config = serde_json::from_value(self.config.clone())?;
                Arc::new(Mutex::new(ModelMistral::from_paths(
                    &self.model_files,
                    &config,
                    Arc::clone(&self.device),
                )?))
            }
            ModelArchitecture::Phi3 => {
                use candle_transformers::models::phi3::Config;
                let config: Config = serde_json::from_value(self.config.clone())?;
                Arc::new(Mutex::new(ModelPhi3::from_paths(
                    &self.model_files,
                    &config,
                    Arc::clone(&self.device),
                )?))
            }
            ModelArchitecture::Qwen2 => {
                use candle_transformers::models::qwen2::Config;
                let config: Config = serde_json::from_value(self.config.clone())?;
                Arc::new(Mutex::new(ModelQwen2::from_paths(
                    &self.model_files,
                    &config,
                    Arc::clone(&self.device),
                )?))
            }
            _ => {
                return Err(CallmError::UnsupportedModel);
            }
        };

        Ok(model)
    }
}

impl LoaderImpl for LoaderSafetensors {
    fn set_device(&mut self, device: Arc<DeviceConfig>) {
        self.device = device;
    }

    fn load(&mut self) -> Result<Arc<Mutex<dyn ModelImpl>>, CallmError> {
        self.validate_location()?;
        self.load_config()?;
        self.load_model()
    }

    fn tokenizer(&mut self) -> Result<Tokenizer, CallmError> {
        let file_str = fs::read_to_string(&self.tokenizer_path)?;
        Tokenizer::from_bytes(file_str.as_bytes())
            .map_err(|e| CallmError::TokenizerError { msg: e.to_string() })

        // Tokenizer::from_file(&self.tokenizer_path)
        //     .map_err(|e| CallmError::TokenizerError { msg: e.to_string() })
    }

    fn template(&mut self) -> Result<Box<dyn TemplateImpl>, CallmError> {
        let mut boxed_template: Box<dyn TemplateImpl> =
            if let Some(template_string) = &self.chat_template {
                Box::new(TemplateJinja::new(template_string))
            } else {
                Box::new(TemplateDummy::new())
            };

        let tokenizer = self.tokenizer()?;
        if let Some(tkn_id) = &self.bos_token_id {
            boxed_template.set_bos_token(tokenizer.id_to_token(*tkn_id as u32));
        }
        if let Some(tkn_id) = &self.eos_token_id {
            boxed_template.set_eos_token(tokenizer.id_to_token(*tkn_id as u32));
        }

        Ok(boxed_template)
    }
}

// read Safetensors model index pointed by 'path' and return vector of model filenames
fn read_model_index_json<P: AsRef<Path>>(path: P) -> Result<Vec<String>, CallmError> {
    use serde_json::Value;

    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let file_values: Value = serde_json::from_reader(reader)?;

    if let Some(obj) = file_values.as_object() {
        if let Some(weight_map) = obj.get("weight_map") {
            if let Some(tensor_map) = weight_map.as_object() {
                let mut files = Vec::new();
                for (_, v) in tensor_map.iter() {
                    if let Some(model_filename) = v.as_str() {
                        let model_filename = model_filename.to_string();
                        if !files.contains(&model_filename) {
                            files.push(model_filename);
                        }
                    }
                }
                return Ok(files);
            }
        }
    }

    Err(CallmError::LoaderFail(
        "Model index deserialization failure".to_string(),
    ))
}
