# callm

## About
Callm allows you to easily run Generative AI models (like Large Language Models) directly on your hardware, offline.   
Under the hood callm relies heavily on the [candle](https://github.com/huggingface/candle) crate and is written in pure Rust 🦀

### Supported models

| Model | Safetensors | GGUF (quantized) |
| :--- | :---: | :---: |
| Llama | ✅ | ✅ |
| Mistral | ✅ | ✅ |
| Phi3 | ✅ | ❌ |
| Qwen2 | ✅ | ❌ |

> Callm is still in early development stage and is NOT production ready yet.

## Installation
Add callm to your dependencies:
```
$ cargo add callm
```

### Enabling GPU support
Callm uses features for enabling GPU support.  

#### NVIDIA - CUDA
```
$ cargo add callm -F cuda
```

#### Apple - Metal
```
$ cargo add callm -F metal
```

## Usage
Callm uses builder pattern to create inference pipelines.

```rust
use callm::pipelines::PipelineText;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let mut pipeline = PipelineText::builder()
		.with_location("/path/to/model")
		.build()?;

	let text_completion = pipeline.run("Tell me a joke about x86 instruction set")?;
	println!("{}", text_completion);

	Ok(())
}
```
