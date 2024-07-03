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

### Portability
Currently, callm is known to run and has been tested on Linux and macOS.   
Windows has not been tested but is expected to work out-of-the-box.

> Callm is still in early development stage and is NOT production ready yet.

## Installation
Add callm to your dependencies:
```
$ cargo add callm
```

### Enabling GPU support
Callm uses features to selectively enable support for GPU acceleration.

#### NVIDIA (CUDA)
Enable `cuda` feature to include support for CUDA devices.
```
$ cargo add callm -F cuda
```

#### Apple (Metal)
Enable `metal` feature to include support for Metal devices.
```
$ cargo add callm -F metal
```

## Usage
Callm uses builder pattern to create inference pipelines.

```rust
use callm::pipelines::PipelineText;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build pipeline
    let mut pipeline = PipelineText::builder()
        .with_location("/path/to/model")
        .build()?;

    // Run inference
    let text_completion = pipeline.run("Tell me a joke about x86 instruction set")?;
    println!("{text_completion}");

    Ok(())
}
```

### Sampling parameters
Override default sampling parameters during pipeline build or afterwards.

```rust
use callm::pipelines::PipelineText;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build pipeline with custom sampling parameters
    let mut pipeline = PipelineText::builder()
        .with_location("/path/to/model")
        .with_temperature(0.65)
        .with_top_k(25)
        .build()?;

    // Adjust sampling parameters later on
    pipeline.set_seed(42);
    pipeline.set_top_p(0.3);

    // Run inference
    let text_completion = pipeline.run("Write an article about Pentium F00F bug")?;
    println!("{text_completion}");

    Ok(())
}
```


### Instruction-following and Chat models
If the model you are loading includes a chat template you can use role-message style inference.

```
TODO!
```

## Documentation
Consult the [documentation](https://docs.rs/callm/) for a full API reference.
