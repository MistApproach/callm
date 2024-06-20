use callm::templates::{MessageRole, TemplateImpl, TemplateJinja as Template};

// Qwen1.5-4B-Chat
const JINJA_TEMPLATE: &str = r#"{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"#;

#[test]
fn single_user_message() {
    let msgs = vec![(MessageRole::User, "User message 1".to_string())];
    let template = Template::new(JINJA_TEMPLATE);

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r#"<|im_start|>user
User message 1<|im_end|>
<|im_start|>assistant
"#
    );
}

#[test]
fn two_messages() {
    let msgs = vec![
        (MessageRole::User, "User message 1".to_string()),
        (MessageRole::Assistant, "Assistant message 1".to_string()),
    ];
    let template = Template::new(JINJA_TEMPLATE);

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r#"<|im_start|>user
User message 1<|im_end|>
<|im_start|>assistant
Assistant message 1<|im_end|>
<|im_start|>assistant
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
    let template = Template::new(JINJA_TEMPLATE);

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r#"<|im_start|>user
User message 1<|im_end|>
<|im_start|>assistant
Assistant message 1<|im_end|>
<|im_start|>user
User message 2<|im_end|>
<|im_start|>assistant
"#
    );
}

#[test]
fn with_system_message() {
    let msgs = vec![
        (MessageRole::System, "System message".to_string()),
        (MessageRole::User, "User message 1".to_string()),
    ];
    let template = Template::new(JINJA_TEMPLATE);

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r#"<|im_start|>system
System message<|im_end|>
<|im_start|>user
User message 1<|im_end|>
<|im_start|>assistant
"#
    );
}

#[test]
fn two_system_messages() {
    let msgs = vec![
        (MessageRole::System, "System message".to_string()),
        (MessageRole::System, "Another system message".to_string()),
    ];
    let template = Template::new(JINJA_TEMPLATE);

    assert_eq!(
        template.apply(msgs.as_slice()).unwrap(),
        r#"<|im_start|>system
System message<|im_end|>
<|im_start|>system
Another system message<|im_end|>
<|im_start|>assistant
"#
    );
}
