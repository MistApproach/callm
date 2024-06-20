use super::get_metadata;
use crate::error::CallmError;
use candle_core::quantized::gguf_file::Content;

/// GGUF Llama model config
#[derive(Clone, Debug, Default)]
pub struct LoaderGgufInfoModelLlama {
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub feed_forward_length: u32,
    pub rope: LoaderGgufInfoModelLlamaRope,
    pub attention: LoaderGgufInfoModelLlamaAttention,
}

#[derive(Clone, Debug, Default)]
pub struct LoaderGgufInfoModelLlamaRope {
    pub dimension_count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct LoaderGgufInfoModelLlamaAttention {
    pub head_count: u32,
    pub layer_norm_rms_epsilon: f32,
    pub head_count_kv: Option<u32>,
}

pub fn parse_llama_kv(ctx: &Content) -> Result<LoaderGgufInfoModelLlama, CallmError> {
    let modelinfo = LoaderGgufInfoModelLlama {
        context_length: get_metadata(&ctx.metadata, "llama.context_length")?.to_u32()?,
        embedding_length: get_metadata(&ctx.metadata, "llama.embedding_length")?.to_u32()?,
        block_count: get_metadata(&ctx.metadata, "llama.block_count")?.to_u32()?,
        feed_forward_length: get_metadata(&ctx.metadata, "llama.feed_forward_length")?.to_u32()?,
        rope: LoaderGgufInfoModelLlamaRope {
            dimension_count: get_metadata(&ctx.metadata, "llama.rope.dimension_count")?.to_u32()?,
        },
        attention: LoaderGgufInfoModelLlamaAttention {
            head_count: get_metadata(&ctx.metadata, "llama.attention.head_count")?.to_u32()?,
            layer_norm_rms_epsilon: get_metadata(
                &ctx.metadata,
                "llama.attention.layer_norm_rms_epsilon",
            )?
            .to_f32()?,
            head_count_kv: if let Ok(v) =
                get_metadata(&ctx.metadata, "llama.attention.head_count_kv")
            {
                Some(v.to_u32()?)
            } else {
                None
            },
        },
    };

    Ok(modelinfo)
}
