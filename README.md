# callm
[![Latest version on crates.io](https://img.shields.io/crates/v/callm?style=plastic)](https://crates.io/crates/callm)
[![Documentation on docs.rs](https://img.shields.io/docsrs/callm?style=plastic)](https://docs.rs/callm)
![License](https://img.shields.io/crates/l/callm?style=plastic)

## About
`callm` enables you to run Generative AI models (such as Large Language Models) directly on your hardware, offline.   
Under the hood, `callm` heavily relies on the [candle](https://github.com/huggingface/candle) crate and is written in pure Rust 🦀.

### Supported models

| Model | Safetensors | GGUF (quantized) |
| :--- | :---: | :---: |
| Llama | ✅ | ✅ |
| Mistral | ✅ | ✅ |
| Phi3 | ✅ | ❌ |
| Qwen2 | ✅ | ❌ |

### Thread safety
While pipelines are safe to send between threads, `callm` has not undergone extensive testing for thread-safety.   
Caution is advised.

### Portability
`callm` is known to run on Linux and macOS, and has been tested on these platforms. While Windows has not been extensively tested, it is expected to work out-of-the-box without issues.

> `callm` is still in an early development stage and is not production-ready yet.

## Installation
Add `callm` to your dependencies:
```
$ cargo add callm
```

### Enabling GPU Support
`callm` uses features to selectively enable support for GPU acceleration.

#### NVIDIA (CUDA)
Enable the `cuda` feature to include support for CUDA devices.

```
$ cargo add callm --features cuda
```

#### Apple (Metal)
Enable the `metal` feature to include support for Metal devices.

```
$ cargo add callm --features metal
```

## Usage
`callm` uses builder pattern to create inference pipelines.

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

### Customizing sampling parameters
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
If the model you are loading includes a chat template, you can use conversation-style inference via `run_chat()`. It accepts a slice of tuples in the form: `(MessageRole, String)`.

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
            "What is your opinion on Rust for Linux kernel development?".to_string(),
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
Several examples and tools can be found in a separate [callm-demos](https://github.com/MistApproach/callm-demos) repo.

## Contributing
Thank you for your interest in contributing to `callm`!

As this project is still in its early stages, your help is invaluable. Here are some ways you can get involved:

* **Report issues**: If you encounter any bugs or unexpected behavior, please file an issue on GitHub. This will help us track and fix problems.
* **Submit a pull request**: If you'd like to contribute code, please fork the repository, make your changes, and submit a pull request. We'll review and merge your changes as soon as possible.
* **Help with documentation**: If you have expertise in a particular area, please help us improve our documentation.

Thank you for your contributions! 💪
