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
    let text_completion = pipeline.run("Tell me a joke about Rust borrow checker")?;
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
If the model you are loading includes a chat template you can use conversation-style inference via `run_chat()`.   
It accepts a slice of tuples in the form: `(MessageRole, String)`.

```rust
use callm::pipelines::PipelineText;
use callm::templates::MessageRole;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build pipeline
    let mut pipeline = PipelineText::builder()
        .with_location("/path/to/model")
        .with_temperature(0.1)
        .build()?;

    // Prepare conversation messages
    let messages = vec![
        (
            MessageRole::System,
            "You are impersonating Linus Torvalds.".to_string(),
        ),
        (
            MessageRole::User,
            "What is your opinion on Rust in Linux kernel development?".to_string(),
        ),
    ];

    // Run chat-style inference
    let assistant_response = pipeline.run_chat(&messages)?;
    println!("{assistant_response}");

    Ok(())
}
```

## Documentation
Consult the [documentation](https://docs.rs/callm/) for a full API reference.
