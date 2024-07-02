//! GGUF loader
//!
//! file format specification: `<https://github.com/ggerganov/ggml/blob/8d6b7038871fada44fbaa61dd5eabe5fccab1cbb/docs/gguf.md>`

pub mod llama;

use super::LoaderImpl;
use crate::device::DeviceConfig;
use crate::error::CallmError;
use crate::models::{ModelImpl, ModelLlamaQuantized};
use crate::templates::{TemplateDummy, TemplateImpl, TemplateJinja};
use candle_core::quantized::gguf_file::{Content, Value};
use llama::{parse_llama_kv, LoaderGgufInfoModelLlama};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;

/// GGUF general metadata
#[derive(Clone, Debug, Default)]
pub struct LoaderGgufInfo {
    /// General metadata - required
    pub architecture: String,
    pub quantization_version: u32,
    pub alignment: u32,
    // General metadata
    pub name: Option<String>,
    pub author: Option<String>,
    pub url: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub file_type: Option<u32>,
    pub source: LoaderGgufInfoSource,
    // Varies by model
    pub model: LoaderGgufInfoModel,
    // Tokenizer config
    pub tokenizer: LoaderGgufInfoTokenizer,
}

/// GGUF general.source metadata
#[derive(Clone, Debug, Default)]
pub struct LoaderGgufInfoSource {
    pub url: Option<String>,
    pub huggingface_repository: Option<String>,
}

/// GGUF model enum
#[derive(Clone, Debug, Default)]
pub enum LoaderGgufInfoModel {
    #[default]
    None,
    Llama(LoaderGgufInfoModelLlama),
}

/// GGUF tokenizer metadata
#[derive(Clone, Debug, Default)]
pub struct LoaderGgufInfoTokenizer {
    // required
    model: String,
    tokens: Vec<String>,
    // optional
    pre: Option<String>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unknown_token_id: Option<u32>,
    separator_token_id: Option<u32>,
    padding_token_id: Option<u32>,
    chat_template: Option<String>,
    // optional arrays
    scores: Option<Vec<f32>>,
    token_type: Option<Vec<i32>>,
    merges: Option<Vec<String>>,
    added_tokens: Option<Vec<String>>,
}

/// GGUF loader
#[derive(Clone, Debug, Default)]
pub struct LoaderGguf {
    location: PathBuf,
    file_size: u64,
    info: LoaderGgufInfo,
    device: Arc<DeviceConfig>,
}

impl LoaderGguf {
    pub fn new(location: &str) -> Self {
        Self {
            location: PathBuf::from(location),
            ..Default::default()
        }
    }
}

impl LoaderImpl for LoaderGguf {
    fn set_device(&mut self, device: Arc<DeviceConfig>) {
        self.device = device;
    }

    fn load(&mut self) -> Result<Arc<Mutex<dyn ModelImpl>>, CallmError> {
        let timer = Instant::now();
        // check if location points to a file
        let file_metadata = fs::metadata(&self.location)?;
        if !file_metadata.is_file() {
            return Err(CallmError::LoaderFail(
                "Location is not pointing to GGUF file".to_string(),
            ));
        }
        self.file_size = file_metadata.len();

        // open file and read GGUF header
        let mut file = fs::File::open(&self.location).expect("Error opening GGUF file");
        let gguf_header = Content::read(&mut file).expect("Error reading GGUF header");

        // parse general kv
        let mut gguf_info = parse_general_kv(&gguf_header)?;

        // parse tokenizer kv
        gguf_info.tokenizer = parse_tokenizer_kv(&gguf_header)?;

        // parse model specific kv pairs
        log::debug!("Model architecture '{}'", gguf_info.architecture.as_str());

        let model = match gguf_info.architecture.as_str() {
            "llama" => {
                // parse Llama kv (for future use)
                gguf_info.model = LoaderGgufInfoModel::Llama(
                    parse_llama_kv(&gguf_header).expect("Error parsing model metadata"),
                );

                // apply fix for wrong EOS token in Meta-Llama3
                // NOTE: model defines token 128001 as EOS (<|end_of_text|>)
                // NOTE: however during inference the model appear to be trained
                // NOTE: with EOS 128009 (<|eot_id|>)
                if let Some(defined_eos) = &gguf_info.tokenizer.eos_token_id {
                    if let Some(defined_eos_str) =
                        &gguf_info.tokenizer.tokens.get(*defined_eos as usize)
                    {
                        if *defined_eos == 128001 && defined_eos_str.as_str() == "<|end_of_text|>" {
                            log::info!("Workaround for wrong Llama EOS token [128001 -> 128009]");
                            gguf_info.tokenizer.eos_token_id = Some(128009);
                        }
                    }
                }
                // load model
                let mut m = ModelLlamaQuantized::from_gguf(
                    gguf_header,
                    &mut file,
                    Arc::clone(&self.device),
                )?;
                m.load()?;

                m
            }
            _ => return Err(CallmError::UnsupportedModel),
        };

        // store GGUF info
        self.info = gguf_info;

        log::info!("Loaded in {:.2?}", Instant::now() - timer);

        Ok(Arc::new(Mutex::new(model)))
    }

    fn tokenizer(&mut self) -> Result<Tokenizer, CallmError> {
        use tokenizers::models::bpe::{Merges, Vocab, BPE};
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::pre_tokenizers::sequence::Sequence;
        use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
        use tokenizers::{
            AddedToken, AddedVocabulary, DecoderWrapper, ModelWrapper, NormalizerWrapper,
            PaddingParams, PostProcessorWrapper, PreTokenizerWrapper, SplitDelimiterBehavior,
            TokenizerBuilder, TruncationParams,
        };

        // tokenizer building blocks
        let normalizer: Option<NormalizerWrapper> = None;
        let mut pre_tokenizer: Option<PreTokenizerWrapper> = None;
        #[allow(unused_assignments)]
        let mut post_processor: Option<PostProcessorWrapper> = None;
        #[allow(unused_assignments)]
        let mut decoder: Option<DecoderWrapper> = None;
        let truncation: Option<TruncationParams> = None;
        let padding: Option<PaddingParams> = None;
        let mut added_vocabulary = AddedVocabulary::new();

        // pre-tokenizer
        if let Some(pre) = &self.info.tokenizer.pre {
            match pre.as_str() {
                "llama-bpe" => {
                    let wrappers = vec![
                        PreTokenizerWrapper::Split(Split::new(SplitPattern::Regex(String::from("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")), SplitDelimiterBehavior::Isolated, false).map_err(|e| CallmError::TokenizerError { msg: e.to_string() })?),
                        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
                    ];
                    pre_tokenizer = Some(PreTokenizerWrapper::Sequence(Sequence::new(wrappers)));
                }
                "deepseek-llm" => todo!(),
                "deepseek-coder" => todo!(),
                "falcon" => todo!(),
                _ => {}
            }
        }
        // tokenizer model
        let model: ModelWrapper = match self.info.tokenizer.model.as_str() {
            "gpt2" => {
                // create post-processor
                post_processor = Some(PostProcessorWrapper::ByteLevel(ByteLevel::new(
                    true, false, true,
                )));
                // create decoder
                decoder = Some(DecoderWrapper::ByteLevel(ByteLevel::new(true, true, true)));
                // create vocabulary
                // TODO: profile with pre-allocated HashMap capacity
                let vocab: Vocab = {
                    let mut tknmap = HashMap::new();
                    for (i, tkn) in (0_u32..).zip(self.info.tokenizer.tokens.iter()) {
                        tknmap.insert(tkn.clone(), i);
                    }
                    tknmap
                };
                // create merges
                // TODO: profile with pre-allocated Vec capacity
                let merges: Merges = self
                    .info
                    .tokenizer
                    .merges
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        let split = v.as_str().split_once(' ').unwrap();
                        (String::from(split.0), String::from(split.1))
                    })
                    .collect();

                // create model
                let bpe = BPE::builder()
                    .vocab_and_merges(vocab, merges)
                    .ignore_merges(true)
                    .build()
                    .map_err(|e| CallmError::TokenizerError { msg: e.to_string() })?;

                let model = ModelWrapper::BPE(bpe);

                // create added vocabulary from control tokens
                if let Some(token_type) = &self.info.tokenizer.token_type {
                    let mut added_tokens = vec![];
                    for (i, tkn) in (0_u32..).zip(token_type) {
                        if *tkn == 3 {
                            added_tokens.push(AddedToken::from(
                                &self.info.tokenizer.tokens[i as usize],
                                true,
                            ));
                        }
                    }
                    added_vocabulary.add_special_tokens(
                        added_tokens.as_slice(),
                        &model,
                        None::<&tokenizers::normalizers::strip::Strip>,
                    );
                }

                model
            }
            "llama" => todo!(),
            _ => unimplemented!(),
        };

        let tokenizer = TokenizerBuilder::new()
            .with_model(model)
            .with_normalizer(normalizer)
            .with_pre_tokenizer(pre_tokenizer)
            .with_post_processor(post_processor)
            .with_decoder(decoder)
            .with_truncation(truncation)
            .with_padding(padding)
            .with_added_vocabulary(added_vocabulary)
            .build()
            .map_err(|e| CallmError::TokenizerError { msg: e.to_string() })?;

        Ok(tokenizer.into())
    }

    fn template(&mut self) -> Result<Box<dyn TemplateImpl>, CallmError> {
        let mut boxed_template: Box<dyn TemplateImpl> =
            if let Some(template_string) = &self.info.tokenizer.chat_template {
                // spawn jinja-style chat template from gguf kv tokenizer.chat_template
                Box::new(TemplateJinja::new(template_string))
            } else {
                // fallback to dummy template
                Box::new(TemplateDummy::new())
            };

        // parse GGUF tokenizer kv for BOS and EOS tokens
        if let Some(tkn_id) = &self.info.tokenizer.bos_token_id {
            boxed_template.set_bos_token(Some(self.info.tokenizer.tokens[*tkn_id as usize].clone()))
        }
        if let Some(tkn_id) = &self.info.tokenizer.eos_token_id {
            boxed_template.set_eos_token(Some(self.info.tokenizer.tokens[*tkn_id as usize].clone()))
        }

        Ok(boxed_template)
    }
}

fn parse_required_kv(ctx: &Content) -> Result<LoaderGgufInfo, CallmError> {
    let architecture = get_metadata(&ctx.metadata, "general.architecture")?
        .to_string()?
        .clone();
    let quantization_version =
        get_metadata(&ctx.metadata, "general.quantization_version")?.to_u32()?;
    let alignment = get_metadata(&ctx.metadata, "general.alignment")
        .unwrap_or(&Value::U32(32))
        .to_u32()?;

    Ok(LoaderGgufInfo {
        architecture,
        quantization_version,
        alignment,
        ..LoaderGgufInfo::default()
    })
}

fn parse_general_kv(ctx: &Content) -> Result<LoaderGgufInfo, CallmError> {
    // parse required metadata
    let mut info = parse_required_kv(ctx)?;

    // parse general metadata
    if let Ok(val) = get_metadata(&ctx.metadata, "general.name") {
        info.name = Some(val.to_string()?.clone());
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "general.author") {
        info.author = Some(val.to_string()?.clone());
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "general.url") {
        info.url = Some(val.to_string()?.clone());
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "general.description") {
        info.description = Some(val.to_string()?.clone());
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "general.file_type") {
        info.file_type = Some(val.to_u32()?);
    }

    // parse source metadata
    if let Ok(val) = get_metadata(&ctx.metadata, "general.source.url") {
        info.source.url = Some(val.to_string()?.clone());
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "general.source.huggingface.repository") {
        info.source.huggingface_repository = Some(val.to_string()?.clone());
    }

    Ok(info)
}

// TODO: proper error handling
fn parse_tokenizer_kv(ctx: &Content) -> Result<LoaderGgufInfoTokenizer, CallmError> {
    let mut info = LoaderGgufInfoTokenizer {
        model: get_metadata(&ctx.metadata, "tokenizer.ggml.model")?
            .to_string()?
            .clone(),
        tokens: get_metadata(&ctx.metadata, "tokenizer.ggml.tokens")?
            .to_vec()?
            .iter()
            .map(|v| v.to_string().unwrap().clone())
            .collect(),
        ..Default::default()
    };

    // optional kv
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.chat_template") {
        info.chat_template = Some(val.to_string()?.clone());
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.ggml.pre") {
        info.pre = Some(val.to_string()?.clone());
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.ggml.bos_token_id") {
        info.bos_token_id = Some(val.to_u32()?);
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.ggml.eos_token_id") {
        info.eos_token_id = Some(val.to_u32()?);
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.ggml.unknown_token_id") {
        info.unknown_token_id = Some(val.to_u32()?);
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.ggml.separator_token_id") {
        info.separator_token_id = Some(val.to_u32()?);
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.ggml.padding_token_id") {
        info.padding_token_id = Some(val.to_u32()?);
    }

    // optional kv arrays
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.ggml.scores") {
        info.scores = Some(val.to_vec()?.iter().map(|v| v.to_f32().unwrap()).collect());
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.ggml.token_type") {
        info.token_type = Some(val.to_vec()?.iter().map(|v| v.to_i32().unwrap()).collect());
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.ggml.merges") {
        info.merges = Some(
            val.to_vec()?
                .iter()
                .map(|v| v.to_string().unwrap().clone())
                .collect(),
        );
    }
    if let Ok(val) = get_metadata(&ctx.metadata, "tokenizer.ggml.added_tokens") {
        info.added_tokens = Some(
            val.to_vec()?
                .iter()
                .map(|v| v.to_string().unwrap().clone())
                .collect(),
        );
    }

    Ok(info)
}

fn get_metadata<'a>(
    metadata: &'a HashMap<String, Value>,
    key: &str,
) -> Result<&'a Value, CallmError> {
    let v = metadata.get(key);

    if let Some(value) = v {
        Ok(value)
    } else {
        Err(CallmError::LoaderFail(format!(
            "Missing GGUF metadata key {}",
            key
        )))
    }
}
