use serde::{Deserialize, Serialize};

use super::Message;

#[derive(Debug, Deserialize, Clone, Default, PartialEq, Serialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

#[cfg(test)]
mod tests {
    use crate::messages::{ContentBlock, ImageSource, MessageContent, Role};

    use super::*;
    #[test]
    fn serde() {
        let tests = vec![
            (
                "simple",
                r#"{
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Hello, world"}
                ]
            }"#,
                Request {
                    model: "claude-3-opus-20240229".to_string(),
                    max_tokens: 1024,
                    messages: vec![Message {
                        role: Role::User,
                        content: MessageContent::Text("Hello, world".to_string()),
                    }],
                    ..Default::default()
                },
            ),
            (
                "multiple conversation",
                r#"{
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "messages": [
                        {"role": "user", "content": "Hello there."},
                        {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"},
                        {"role": "user", "content": "Can you explain LLMs in plain English?"}
                ]
            }"#,
                Request {
                    model: "claude-3-opus-20240229".to_string(),
                    max_tokens: 1024,
                    messages: vec![
                        Message {
                            role: Role::User,
                            content: MessageContent::Text("Hello there.".to_string()),
                        },
                        Message {
                            role: Role::Assistant,
                            content: MessageContent::Text(
                                "Hi, I'm Claude. How can I help you?".to_string(),
                            ),
                        },
                        Message {
                            role: Role::User,
                            content: MessageContent::Text(
                                "Can you explain LLMs in plain English?".to_string(),
                            ),
                        },
                    ],
                    ..Default::default()
                },
            ),
            (
                "image content",
                r#"{
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": [
                        {
                          "type": "image",
                          "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "/9j/4AAQSkZJRg..."
                          }
                        },
                        {"type": "text", "text": "What is in this image?"}
                      ]}
                ]
            }"#,
                Request {
                    model: "claude-3-opus-20240229".to_string(),
                    max_tokens: 1024,
                    messages: vec![Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Image {
                                source: ImageSource::Base64 {
                                    media_type: "image/jpeg".to_string(),
                                    data: "/9j/4AAQSkZJRg...".to_string(),
                                },
                            },
                            ContentBlock::Text {
                                text: "What is in this image?".to_string(),
                            },
                        ]),
                    }],
                    ..Default::default()
                },
            ),
        ];
        for (name, json, expected) in tests {
            //test deserialize
            let actual: Request = serde_json::from_str(json).unwrap();
            assert_eq!(actual, expected, "deserialize test failed: {}", name);
            //test serialize
            let serialized = serde_json::to_string(&expected).unwrap();
            let actual: Request = serde_json::from_str(&serialized).unwrap();
            assert_eq!(actual, expected, "serialize test failed: {}", name);
        }
    }
}
