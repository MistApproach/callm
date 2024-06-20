use callm::templates::{MessageRole, TemplateImpl, TemplateJinja as Template};

// Phi-3-mini-4k-instruct
const JINJA_TEMPLATE: &str = r#"{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '
' + message['content'] + '<|end|>' + '
' + '<|assistant|>' + '
'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '
'}}{% endif %}{% endfor %}"#;
const BOS_TOKEN: &str = r#"<s>"#;

#[test]
fn single_user_message() {
    let msgs = vec![(MessageRole::User, "User message 1".to_string())];
    let mut template = Template::new(JINJA_TEMPLATE);
    template.set_bos_token(Some(BOS_TOKEN.to_string()));

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r#"<s><|user|>
User message 1<|end|>
<|assistant|>
"#
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

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r#"<s><|user|>
User message 1<|end|>
<|assistant|>
Assistant message 1<|end|>
"#
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

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r#"<s><|user|>
User message 1<|end|>
<|assistant|>
Assistant message 1<|end|>
<|user|>
User message 2<|end|>
<|assistant|>
"#
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

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r#"<s><|user|>
User message 1<|end|>
<|assistant|>
"#
    );
}
