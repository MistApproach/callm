use callm::templates::{MessageRole, TemplateImpl, TemplateJinja as Template};

// Meta-Llama-3-8B-Instruct
const JINJA_TEMPLATE: &str = r#"{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>

'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>

' }}{% endif %}"#;
const BOS_TOKEN: &str = r#"<|begin_of_text|>"#;
const EOS_TOKEN: &str = r#"<|end_of_text|>"#;

#[test]
fn single_user_message() {
    let msgs = vec![(MessageRole::User, "User message 1".to_string())];
    let mut template = Template::new(JINJA_TEMPLATE);
    template.set_bos_token(Some(BOS_TOKEN.to_string()));
    template.set_eos_token(Some(EOS_TOKEN.to_string()));

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r##"<|begin_of_text|><|start_header_id|>user<|end_header_id|>

User message 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"##
    );
}

#[test]
fn two_messages() {
    let msgs = vec![
        (MessageRole::User, "User message 1".to_string()),
        (MessageRole::Assistant, "Assistant message 1".to_string()),
    ];
    let mut template = Template::new(JINJA_TEMPLATE);
    template.set_bos_token(Some(BOS_TOKEN.to_string()));
    template.set_eos_token(Some(EOS_TOKEN.to_string()));

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r##"<|begin_of_text|><|start_header_id|>user<|end_header_id|>

User message 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Assistant message 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"##
    );
}

#[test]
fn three_messages() {
    let msgs = vec![
        (MessageRole::User, "User message 1".to_string()),
        (MessageRole::Assistant, "Assistant message 1".to_string()),
        (MessageRole::User, "User message 2".to_string()),
    ];
    let mut template = Template::new(JINJA_TEMPLATE);
    template.set_bos_token(Some(BOS_TOKEN.to_string()));
    template.set_eos_token(Some(EOS_TOKEN.to_string()));

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r##"<|begin_of_text|><|start_header_id|>user<|end_header_id|>

User message 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Assistant message 1<|eot_id|><|start_header_id|>user<|end_header_id|>

User message 2<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"##
    );
}

#[test]
fn with_system_message() {
    let msgs = vec![
        (MessageRole::System, "System message".to_string()),
        (MessageRole::User, "User message 1".to_string()),
    ];
    let mut template = Template::new(JINJA_TEMPLATE);
    template.set_bos_token(Some(BOS_TOKEN.to_string()));
    template.set_eos_token(Some(EOS_TOKEN.to_string()));

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r##"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

System message<|eot_id|><|start_header_id|>user<|end_header_id|>

User message 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"##
    );
}
