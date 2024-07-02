use crate::device::DeviceConfig;
use crate::error::CallmError;
use crate::loaders::LoaderImpl;
use crate::models::ModelImpl;
use crate::templates::MessageRole;
use crate::utils::autodetect_loader;
use std::sync::{Arc, Mutex};

/// Pipeline for text generation
pub struct PipelineText {
    model: Option<Box<dyn ModelImpl + Send>>,
    loader: Arc<Mutex<dyn LoaderImpl>>,
    device: Arc<DeviceConfig>,
    // inference parameters
    seed: Option<u64>,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
}

impl PipelineText {
    pub fn builder() -> PipelineTextBuilder {
        PipelineTextBuilder::new()
    }

    pub fn new(loader: Arc<Mutex<dyn LoaderImpl>>) -> Self {
        Self {
            loader,
            model: None,
            device: Arc::new(DeviceConfig::autodetect()),
            seed: None,
            temperature: 0.7,
            top_k: None,
            top_p: None,
        }
    }

    pub fn from_path(path: &str) -> Result<Self, CallmError> {
        Ok(Self::new(autodetect_loader(path)?))
    }

    pub fn load(&mut self) -> Result<(), CallmError> {
        let mut loader = self.loader.lock().unwrap();
        // propagate device to loader
        loader.set_device(Arc::clone(&self.device));
        // loader
        let mut model = loader.load()?;
        // model
        model.load()?;
        // store model trait object
        self.model = Some(model);

        Ok(())
    }

    pub fn run(&mut self, text: &str) -> Result<String, CallmError> {
        use candle_core::Tensor;
        use candle_transformers::generation::{LogitsProcessor, Sampling};

        let model = self.model.as_mut().ok_or(CallmError::GenericError(
            "Cannot run inference, model not loaded".to_string(),
        ))?;

        let mut loader = self.loader.lock().unwrap();

        // spawn logits processor
        // TODO: custom seed / random seed support
        let sampling = {
            if self.temperature <= 0.0 {
                Sampling::ArgMax
            } else {
                match (self.top_k, self.top_p) {
                    (None, None) => Sampling::All {
                        temperature: self.temperature,
                    },
                    (Some(k), None) => Sampling::TopK {
                        k,
                        temperature: self.temperature,
                    },
                    (None, Some(p)) => Sampling::TopP {
                        p,
                        temperature: self.temperature,
                    },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP {
                        k,
                        p,
                        temperature: self.temperature,
                    },
                }
            }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(self.seed.unwrap_or(0), sampling);

        // spawn tokenizer
        let tokenizer = loader.tokenizer()?;

        // spawn template and get EOS token
        let template = loader.template()?;
        let eos_token_str = template.get_eos_token().expect("Missing EOS token");
        let eos_token = tokenizer
            .token_to_id(eos_token_str)
            .expect("EOS token missing in the tokenizer");

        // tokenize user input
        let mut tokens = tokenizer
            .encode(text, false)
            .map_err(|e| CallmError::TokenizerError { msg: e.to_string() })?
            .get_ids()
            .to_vec();

        let num_tokens_at_start = tokens.len();
        log::trace!("EOS token: {} '{}'", eos_token, eos_token_str);
        log::trace!("Tokens: {:?}", tokens);
        log::trace!("Tokens count: {}", num_tokens_at_start);
        // TODO: calculate real max number of tokens by subtracting num_tokens_at_start from
        // context size
        for index in 0..1000 {
            let ctxt_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(ctxt_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, self.device.candle_device())?.unsqueeze(0)?;

            let logits = model.forward(&input, start_pos)?.squeeze(0)?.squeeze(0)?;

            let new_token = logits_processor.sample(&logits)?;
            tokens.push(new_token);

            log::trace!("New token generated: {}", new_token);
            if new_token == eos_token {
                break;
            }
        }

        // clear KV cache
        model.clear_kv_cache()?;

        // decode newly added tokens
        let new_text = tokenizer
            .decode(&tokens[num_tokens_at_start..], true)
            .map_err(|e| CallmError::TokenizerError { msg: e.to_string() })?;

        Ok(new_text)
    }

    pub fn set_device(&mut self, device: DeviceConfig) {
        self.device = Arc::new(device);
    }

    pub fn run_chat(&mut self, messages: &[(MessageRole, String)]) -> Result<String, CallmError> {
        if self.model.is_none() {
            return Err(CallmError::GenericError(
                "Cannot run inference, model not loaded".to_string(),
            ));
        }

        let prompt = {
            let mut loader = self.loader.lock().unwrap();

            let template = loader.template()?;
            template.apply(messages)?
        };

        self.run(&prompt)
    }

    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.seed = seed;
    }

    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    pub fn set_top_k(&mut self, top_k: Option<usize>) {
        self.top_k = top_k;
    }

    pub fn set_top_p(&mut self, top_p: Option<f64>) {
        self.top_p = top_p;
    }
}

/// PipelineText Builder
#[derive(Default)]
pub struct PipelineTextBuilder {
    location: Option<String>,
    loader: Option<Arc<Mutex<dyn LoaderImpl>>>,
    device: Option<DeviceConfig>,
    autoload: bool,
    temperature: f64,
    seed: Option<u64>,
    top_k: Option<usize>,
    top_p: Option<f64>,
}

impl PipelineTextBuilder {
    pub fn new() -> Self {
        Self {
            temperature: 0.7,
            autoload: true,
            ..Default::default()
        }
    }

    pub fn with_location(mut self, location: &str) -> Self {
        self.location = Some(location.to_string());
        self
    }

    pub fn with_loader(mut self, loader: Arc<Mutex<dyn LoaderImpl>>) -> Self {
        self.loader = Some(loader);
        self
    }

    pub fn with_device(mut self, device: DeviceConfig) -> Self {
        self.device = Some(device);
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn autoload(mut self, autoload: bool) -> Self {
        self.autoload = autoload;
        self
    }

    pub fn build(self) -> Result<PipelineText, CallmError> {
        let mut pipeline = match self.loader {
            Some(loader) => PipelineText::new(loader),
            None => match self.location {
                Some(location) => PipelineText::from_path(&location)?,
                None => {
                    return Err(CallmError::GenericError(
                        "No location or loader specified. Use `with_location` or `with_loader`"
                            .to_string(),
                    ));
                }
            },
        };

        pipeline.temperature = self.temperature;
        pipeline.seed = self.seed;
        pipeline.top_k = self.top_k;
        pipeline.top_p = self.top_p;

        if let Some(device) = self.device {
            pipeline.device = Arc::new(device);
        }

        if self.autoload {
            pipeline.load()?;
        }

        Ok(pipeline)
    }
}
