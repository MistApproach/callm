use callm::templates::{MessageRole, TemplateImpl, TemplateJinja as Template};

// Mistral-7B-Instruct-v0.3
const JINJA_TEMPLATE: &str = r#"{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"#;
const BOS_TOKEN: &str = r#"<s>"#;
const EOS_TOKEN: &str = r#"</s>"#;

#[test]
fn single_user_message() {
    let msgs = vec![(MessageRole::User, "User message 1".to_string())];
    let mut template = Template::new(JINJA_TEMPLATE);
    template.set_bos_token(Some(BOS_TOKEN.to_string()));
    template.set_eos_token(Some(EOS_TOKEN.to_string()));

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r#"<s>[INST] User message 1 [/INST]"#
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
        r#"<s>[INST] User message 1 [/INST]Assistant message 1</s>"#
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
        r#"<s>[INST] User message 1 [/INST]Assistant message 1</s>[INST] User message 2 [/INST]"#
    );
}

#[test]
#[should_panic]
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
