use serde::{Deserialize, Serialize};

use super::{BaseContentBlock, ContentBlock, Message, MessageContent, Role};

#[derive(Debug, Deserialize, Clone, Default, PartialEq, Serialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<System>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
}

#[derive(Debug, Deserialize, Clone, Default, PartialEq, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: String,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Thinking {
    #[serde(rename = "enabled")]
    Enabled { budget_tokens: u32 },
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum System {
    Text(String),
    Blocks(Vec<SystemMessage>),
}

#[derive(Debug, Deserialize, Clone, Default, PartialEq, Serialize)]
pub struct SystemMessage {
    pub r#type: SystemMessageType,
    pub text: String,
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Deserialize, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SystemMessageType {
    #[default]
    Text,
}

#[derive(Debug, Deserialize, Clone, Default, PartialEq, Serialize)]
pub struct CacheControl {
    pub r#type: CacheControlType,
}

#[derive(Debug, Deserialize, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheControlType {
    #[default]
    Ephemeral,
}

/// process_messages take arbitrary user input messages and process them to ensure them conform to Anthropic API requirements.
/// the requirements are:
/// 1. start with user message
/// 2. alternate between user and assistant message
/// 3. the last assistant message cannot have trailing empty space
///
/// This function will:
/// 1. drop any empty message
/// 2. concatenate consecutive messages of the same role
/// 3. add a user message to the start of the conversation if the first message is of role assistant
/// 4. trim trailing empty space from the last message if it is of role assistant
pub fn process_messages(messages: &[Message]) -> Vec<Message> {
    let mut filtered = Vec::with_capacity(messages.len());
    if messages.is_empty() {
        return filtered;
    }

    let mut prev_message: Option<Message> = None;
    for message in messages {
        //if content is empty, drop the message
        if message.is_all_empty() {
            continue;
        }
        if let Some(prev_msg) = prev_message.as_ref() {
            if prev_msg.role == message.role {
                let mut combined_message = prev_msg.clone();
                match (&mut combined_message.content, &message.content) {
                    (MessageContent::Text(prev), MessageContent::Text(curr)) => {
                        prev.push('\n');
                        prev.push_str(curr);
                    }
                    (MessageContent::Blocks(prev), MessageContent::Blocks(curr)) => {
                        prev.retain(|b| !b.is_empty());
                        let curr_clone: Vec<_> =
                            curr.clone().into_iter().filter(|v| !v.is_empty()).collect();
                        prev.extend(curr_clone);
                    }
                    (MessageContent::Blocks(prev), MessageContent::Text(curr)) => {
                        prev.retain(|v| !v.is_empty());
                        prev.push(ContentBlock::Base(BaseContentBlock::Text {
                            text: curr.clone(),
                        }));
                    }
                    (MessageContent::Text(prev), MessageContent::Blocks(curr)) => {
                        let mut blocks =
                            vec![ContentBlock::Base(BaseContentBlock::Text { text: prev.clone() })];
                        let curr_clone: Vec<_> =
                            curr.clone().into_iter().filter(|v| !v.is_empty()).collect();
                        blocks.extend(curr_clone);
                        combined_message.content = MessageContent::Blocks(blocks);
                    }
                }
                filtered.pop();
                filtered.push(combined_message.clone());
                prev_message = Some(combined_message);
                continue;
            }
        }
        filtered.push(message.clone());
        prev_message = Some(message.clone());
    }

    //if first message is of role assistant, add a user message to the start of the conversation
    if let Some(first) = messages.first() {
        if first.role == Role::Assistant {
            filtered.insert(
                0,
                Message {
                    role: Role::User,
                    content: MessageContent::Text("Starting the conversation...".to_string()),
                },
            );
        }
    }

    //if last message is of role assistant,
    //trim trailing empty space
    //the previous step guarantees that the last message is not empty
    if let Some(last) = filtered.last_mut() {
        if last.role == Role::Assistant {
            match &mut last.content {
                MessageContent::Text(text) => {
                    *text = text.trim_end().to_string();
                }
                MessageContent::Blocks(blocks) => {
                    for block in blocks {
                        if let ContentBlock::Base(BaseContentBlock::Text { text }) = block {
                            *text = text.trim_end().to_string();
                        }
                    }
                }
            }
        }
    }

    filtered
}

#[cfg(test)]
mod tests {
    use crate::messages::{
        ContentBlock, ImageSource, MessageContent, RequestOnlyContentBlock, Role,
        ToolUseContentBlock,
    };

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
                "with thinking enabled",
                r#"{
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 5000
                },
                "messages": [
                    {"role": "user", "content": "Solve this complex math problem step by step"}
                ]
            }"#,
                Request {
                    model: "claude-3-opus-20240229".to_string(),
                    max_tokens: 1024,
                    thinking: Some(Thinking::Enabled {
                        budget_tokens: 5000,
                    }),
                    messages: vec![Message {
                        role: Role::User,
                        content: MessageContent::Text(
                            "Solve this complex math problem step by step".to_string(),
                        ),
                    }],
                    ..Default::default()
                },
            ),
            (
                "system messages",
                r#"{
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "system":[
                    {"type":"text","text":"You are a helpful assistant."},
                    {"type":"text","text":"You are a really helpful assistant."}
                ],
                "messages": [
                    {"role": "user", "content": "Hello, world"}
                ]
            }"#,
                Request {
                    model: "claude-3-opus-20240229".to_string(),
                    max_tokens: 1024,
                    system: Some(System::Blocks(vec![
                        SystemMessage {
                            r#type: SystemMessageType::Text,
                            text: "You are a helpful assistant.".to_string(),
                            cache_control: None,
                        },
                        SystemMessage {
                            r#type: SystemMessageType::Text,
                            text: "You are a really helpful assistant.".to_string(),
                            cache_control: None,
                        },
                    ])),
                    messages: vec![Message {
                        role: Role::User,
                        content: MessageContent::Text("Hello, world".to_string()),
                    }],
                    ..Default::default()
                },
            ),
            (
                "system message with cache control",
                r#"{
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "system":[
                    {"type":"text","text":"You are a helpful assistant."},
                    {"type":"text","text":"You are a really helpful assistant.", "cache_control": {"type":"ephemeral"}}
                ],
                "messages": [
                    {"role": "user", "content": "Hello, world"}
                ]
            }"#,
                Request {
                    model: "claude-3-opus-20240229".to_string(),
                    max_tokens: 1024,
                    system: Some(System::Blocks(vec![
                        SystemMessage {
                            r#type: SystemMessageType::Text,
                            text: "You are a helpful assistant.".to_string(),
                            cache_control: None,
                        },
                        SystemMessage {
                            r#type: SystemMessageType::Text,
                            text: "You are a really helpful assistant.".to_string(),
                            cache_control: Some(CacheControl {
                                r#type: CacheControlType::Ephemeral,
                            }),
                        },
                    ])),
                    messages: vec![Message {
                        role: Role::User,
                        content: MessageContent::Text("Hello, world".to_string()),
                    }],
                    ..Default::default()
                },
            ),
            (
                "system message string",
                r#"{
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "system":"You are a helpful assistant.",
                "messages": [
                    {"role": "user", "content": "Hello, world"}
                ]
            }"#,
                Request {
                    model: "claude-3-opus-20240229".to_string(),
                    max_tokens: 1024,
                    system: Some(System::Text("You are a helpful assistant.".to_string())),
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
                            ContentBlock::RequestOnly(RequestOnlyContentBlock::Image {
                                source: ImageSource::Base64 {
                                    media_type: "image/jpeg".to_string(),
                                    data: "/9j/4AAQSkZJRg...".to_string(),
                                },
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "What is in this image?".to_string(),
                            }),
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

    #[test]
    fn process() {
        let tests = vec![
            (
                "[(assistant, text)]",
                vec![Message {
                    role: Role::Assistant,
                    content: MessageContent::Text("hi".to_string()),
                }],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Starting the conversation...".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("hi".to_string()),
                    },
                ],
            ),
            (
                "[(assistant, blocks)]",
                vec![Message {
                    role: Role::Assistant,
                    content: MessageContent::Blocks(vec![
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "hi".to_string(),
                        }),
                    ]),
                }],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Starting the conversation...".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "hi".to_string(),
                            }),
                        ]),
                    },
                ],
            ),
            (
                "[(assistant, blocks)]-2",
                vec![Message {
                    role: Role::Assistant,
                    content: MessageContent::Blocks(vec![
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "hi".to_string(),
                        }),
                        ContentBlock::RequestOnly(RequestOnlyContentBlock::Image {
                            source: ImageSource::Base64 {
                                media_type: "img/png".to_string(),
                                data: "abcs".to_string(),
                            },
                        }),
                    ]),
                }],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Starting the conversation...".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "hi".to_string(),
                            }),
                            ContentBlock::RequestOnly(RequestOnlyContentBlock::Image {
                                source: ImageSource::Base64 {
                                    media_type: "img/png".to_string(),
                                    data: "abcs".to_string(),
                                },
                            }),
                        ]),
                    },
                ],
            ),
            (
                "[(assistant, text), (user, text)]",
                vec![
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("hi".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("hi".to_string()),
                    },
                ],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Starting the conversation...".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("hi".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("hi".to_string()),
                    },
                ],
            ),
            (
                "[(assistant, text), (user, blocks)]",
                vec![
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("hi".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "hi".to_string(),
                            }),
                            ContentBlock::RequestOnly(RequestOnlyContentBlock::Image {
                                source: ImageSource::Base64 {
                                    media_type: "img/png".to_string(),
                                    data: "abcs".to_string(),
                                },
                            }),
                        ]),
                    },
                ],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Starting the conversation...".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("hi".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "hi".to_string(),
                            }),
                            ContentBlock::RequestOnly(RequestOnlyContentBlock::Image {
                                source: ImageSource::Base64 {
                                    media_type: "img/png".to_string(),
                                    data: "abcs".to_string(),
                                },
                            }),
                        ]),
                    },
                ],
            ),
            (
                "[(user, text), (user, text)]",
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("how are you".to_string()),
                    },
                ],
                vec![Message {
                    role: Role::User,
                    content: MessageContent::Text("Hi,\nhow are you".to_string()),
                }],
            ),
            (
                "[(user, text), (user, blocks)]",
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                ],
                vec![Message {
                    role: Role::User,
                    content: MessageContent::Blocks(vec![
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "Hi,".to_string(),
                        }),
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "how are you".to_string(),
                        }),
                    ]),
                }],
            ),
            (
                "[(user, blocks), (user, text)]",
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                ],
                vec![Message {
                    role: Role::User,
                    content: MessageContent::Blocks(vec![
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "how are you".to_string(),
                        }),
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "Hi,".to_string(),
                        }),
                    ]),
                }],
            ),
            (
                "[(assistant, text), (assistant, text)]",
                vec![
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("how are you".to_string()),
                    },
                ],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Starting the conversation...".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("Hi,\nhow are you".to_string()),
                    },
                ],
            ),
            (
                //this may not happen
                "[(assistant, blocks), (assistant, text)]",
                vec![
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                ],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Starting the conversation...".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "Hi,".to_string(),
                            }),
                        ]),
                    },
                ],
            ),
            (
                "[(user, blocks), (user, text), (user, blocks)]",
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                        ]),
                    },
                ],
                vec![Message {
                    role: Role::User,
                    content: MessageContent::Blocks(vec![
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "how are you".to_string(),
                        }),
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "Hi,".to_string(),
                        }),
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "who are you".to_string(),
                        }),
                    ]),
                }],
            ),
            (
                "[(user, blocks), (user, text), (user, blocks), (user, text)]",
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("ho".to_string()),
                    },
                ],
                vec![Message {
                    role: Role::User,
                    content: MessageContent::Blocks(vec![
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "how are you".to_string(),
                        }),
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "Hi,".to_string(),
                        }),
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "who are you".to_string(),
                        }),
                        ContentBlock::Base(BaseContentBlock::Text {
                            text: "ho".to_string(),
                        }),
                    ]),
                }],
            ),
            (
                "[(user, blocks), (user, text), (user, blocks), (assistant, text)]",
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("ho".to_string()),
                    },
                ],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "Hi,".to_string(),
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("ho".to_string()),
                    },
                ],
            ),
            (
                "[(user, blocks), (user, text), (assistant, blocks), (user, text)]",
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("ho".to_string()),
                    },
                ],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "Hi,".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("ho".to_string()),
                    },
                ],
            ),
            (
                "[(user, blocks), (assistant, text), (user, blocks), (user, text)]",
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("ho".to_string()),
                    },
                ],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "ho".to_string(),
                            }),
                        ]),
                    },
                ],
            ),
            (
                "[(assistant, blocks), (user, text), (user, blocks), (user, text)]",
                vec![
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Hi,".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("ho".to_string()),
                    },
                ],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Starting the conversation...".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "Hi,".to_string(),
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "ho".to_string(),
                            }),
                        ]),
                    },
                ],
            ),
            (
                "[(user, text), empty, (assistant, text with trailing space), empty]",
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("hi".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("hi    ".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("".to_string()),
                    },
                ],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("hi".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("hi".to_string()),
                    },
                ],
            ),
            (
                "last one",
                vec![
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("hi".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "   ".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "     ".to_string(),
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "hi".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("how are you".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("hi   ".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you    ".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("         ".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("".to_string()),
                    },
                ],
                vec![
                    Message {
                        role: Role::User,
                        content: MessageContent::Text("Starting the conversation...".to_string()),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("hi".to_string()),
                    },
                    Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "hi".to_string(),
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "how are you".to_string(),
                            }),
                        ]),
                    },
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "hi".to_string(),
                            }),
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "who are you".to_string(),
                            }),
                        ]),
                    },
                ],
            ),
        ];
        for (name, messages, expected) in tests {
            let got = process_messages(&messages);
            assert_eq!(got, expected, "test failed: {}", name);
        }
    }

    #[test]
    fn tool_use() {
        let tests = vec![
            (
                "simple tool",
                r#"{
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "tools": [{
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "input_schema": "{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The city and state, e.g. San Francisco, CA\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"],\"description\":\"The unit of temperature, either \\\"celsius\\\" or \\\"fahrenheit\\\"\"}},\"required\":[\"location\"]}"
                }],
                "messages": [{"role": "user", "content": "What is the weather like in San Francisco?"}]
            }"#,
                Request {
                    model: "claude-3-opus-20240229".to_string(),
                    max_tokens: 1024,
                    messages: vec![Message {
                        role: Role::User,
                        content: MessageContent::Text(
                            "What is the weather like in San Francisco?".to_string(),
                        ),
                    }],
                    tools: Some(vec![Tool {
                        name: "get_weather".to_string(),
                        description: Some(
                            "Get the current weather in a given location".to_string(),
                        ),
                        input_schema: "{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The city and state, e.g. San Francisco, CA\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"],\"description\":\"The unit of temperature, either \\\"celsius\\\" or \\\"fahrenheit\\\"\"}},\"required\":[\"location\"]}".to_string(),
                    }]),
                    ..Default::default()
                },
            ),
            (
                "sequencial",
                r#"{
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "input_schema": "{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The city and state, e.g. San Francisco, CA\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"],\"description\":\"The unit of temperature, either \\\"celsius\\\" or \\\"fahrenheit\\\"\"}},\"required\":[\"location\"]}"
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather like in San Francisco?"
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "<thinking>I need to use get_weather, and the user wants SF, which is likely San Francisco, CA.</thinking>"
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_01A09q90qw90lq917835lq9",
                                "name": "get_weather",
                                "input": {
                                    "location": "San Francisco, CA",
                                    "unit": "celsius"
                                }
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                                "content": "15 degrees"
                            }
                        ]
                    }
                ]
            }"#,
                Request {
                    model: "claude-3-opus-20240229".to_string(),
                    max_tokens: 1024,
                    tools: Some(vec![Tool {
                        name: "get_weather".to_string(),
                        description: Some(
                            "Get the current weather in a given location".to_string(),
                        ),
                        input_schema: "{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The city and state, e.g. San Francisco, CA\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"],\"description\":\"The unit of temperature, either \\\"celsius\\\" or \\\"fahrenheit\\\"\"}},\"required\":[\"location\"]}".to_string(),
                    }]),
                    messages: vec![
                        Message {
                            role: Role::User,
                            content: MessageContent::Text(
                                "What is the weather like in San Francisco?".to_string(),
                            ),
                        },
                        Message {
                            role: Role::Assistant,
                            content: MessageContent::Blocks(vec![
                                ContentBlock::Base(BaseContentBlock::Text {
                                    text: "<thinking>I need to use get_weather, and the user wants SF, which is likely San Francisco, CA.</thinking>".to_string(),
                                }),
                                ContentBlock::Base(BaseContentBlock::ToolUse(ToolUseContentBlock {
                                    id: "toolu_01A09q90qw90lq917835lq9".to_string(),
                                    name: "get_weather".to_string(),
                                    input: serde_json::json!({
                                        "location": "San Francisco, CA",
                                        "unit": "celsius"
                                    }),
                                })),
                            ]),
                        },
                        Message {
                            role: Role::User,
                            content: MessageContent::Blocks(vec![
                                ContentBlock::RequestOnly(RequestOnlyContentBlock::ToolResult {
                                    tool_use_id: "toolu_01A09q90qw90lq917835lq9".to_string(),
                                    content: "15 degrees".to_string(),
                                }),
                            ]),
                        },
                    ],
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
