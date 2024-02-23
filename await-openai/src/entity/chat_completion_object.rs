use std::fmt;

use serde::{Deserialize, Serialize};

use super::create_chat_completion::{FinishReason, ToolCall};

#[derive(Debug, Deserialize, Clone, Default, PartialEq, Serialize)]
pub struct Response {
    /// A unique identifier for the completion.
    pub id: String,
    pub choices: Vec<Choice>,
    /// The Unix timestamp (in seconds) of when the completion was created.
    pub created: u32,

    /// The model used for completion.
    pub model: String,
    /// This fingerprint represents the backend configuration that the model runs with.
    ///
    /// Can be used in conjunction with the `seed` request parameter to understand when backend changes have been
    /// made that might impact determinism.
    pub system_fingerprint: Option<String>,

    /// The object type, which is always "text_completion"
    pub object: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone, PartialEq)]
pub struct Message {
    /// The contents of the message.
    pub content: Option<String>,

    /// The tool calls generated by the model, such as function calls.
    pub tool_calls: Option<Vec<ToolCall>>,

    /// The role of the author of this message.
    pub role: Role,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    #[default]
    User,
    Assistant,
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match *self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence,
    /// `length` if the maximum number of tokens specified in the request was reached,
    /// `content_filter` if content was omitted due to a flag from our content filters,
    /// `tool_calls` if the model called a tool, or `function_call` (deprecated) if the model called a function.
    pub finish_reason: Option<FinishReason>,
    /// Log probability information for the choice.
    pub logprobs: Option<Logprobs>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct Logprobs {
    /// A list of message content tokens with log probability information.
    pub content: Option<Vec<LogprobContent>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct LogprobContent {
    /// The token.
    pub token: String,
    /// The log probability of this token.
    pub logprob: f32,
    /// A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token.
    pub bytes: Option<Vec<u8>>,
    ///  List of the most likely tokens and their log probability, at this token position. In rare cases, there may be fewer than the number of requested `top_logprobs` returned.
    pub top_logprobs: Vec<TopLogprobs>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct TopLogprobs {
    /// The token.
    pub token: String,
    /// The log probability of this token.
    pub logprob: f32,
    /// A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token.
    pub bytes: Option<Vec<u8>>,
}

/// Usage statistics for the completion request.
#[derive(Debug, Deserialize, Serialize, Default, Clone, PartialEq)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the generated completion.
    pub completion_tokens: u32,
    /// Total number of tokens used in the request (prompt + completion).
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use crate::entity::create_chat_completion::{ToolCallFunction, ToolCallFunctionObj};

    use super::*;

    #[test]
    fn serde() {
        let tests = vec![
            (
                "default",
                r#"{
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-0613",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [{
                  "index": 0,
                  "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?"
                  },
                  "logprobs": null,
                  "finish_reason": "stop"
                }],
                "usage": {
                  "prompt_tokens": 9,
                  "completion_tokens": 12,
                  "total_tokens": 21
                }
              }"#,
                Response {
                    id: "chatcmpl-123".to_string(),
                    object: "chat.completion".to_string(),
                    created: 1677652288,
                    model: "gpt-3.5-turbo-0613".to_string(),
                    system_fingerprint: Some("fp_44709d6fcb".to_string()),
                    choices: vec![Choice {
                        index: 0,
                        message: Message {
                            role: Role::Assistant,
                            content: Some(
                                "\n\nHello there, how may I assist you today?".to_string(),
                            ),
                            tool_calls: None,
                        },
                        logprobs: None,
                        finish_reason: Some(FinishReason::Stop),
                    }],
                    usage: Usage {
                        prompt_tokens: 9,
                        completion_tokens: 12,
                        total_tokens: 21,
                    },
                },
            ),
            (
                "function",
                r#"{
                    "id": "chatcmpl-abc123",
                    "object": "chat.completion",
                    "created": 1699896916,
                    "model": "gpt-3.5-turbo-0613",
                    "choices": [
                      {
                        "index": 0,
                        "message": {
                          "role": "assistant",
                          "content": null,
                          "tool_calls": [
                            {
                              "id": "call_abc123",
                              "type": "function",
                              "function": {
                                "name": "get_current_weather",
                                "arguments": "{\n\"location\": \"Boston, MA\"\n}"
                              }
                            }
                          ]
                        },
                        "logprobs": null,
                        "finish_reason": "tool_calls"
                      }
                    ],
                    "usage": {
                      "prompt_tokens": 82,
                      "completion_tokens": 17,
                      "total_tokens": 99
                    }
                  }"#,
                Response {
                    id: "chatcmpl-abc123".to_string(),
                    object: "chat.completion".to_string(),
                    created: 1699896916,
                    model: "gpt-3.5-turbo-0613".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        message: Message {
                            role: Role::Assistant,
                            content: None,
                            tool_calls: Some(vec![ToolCall::Function(ToolCallFunction {
                                id: "call_abc123".to_string(),
                                function: ToolCallFunctionObj {
                                    name: "get_current_weather".to_string(),
                                    arguments: "{\n\"location\": \"Boston, MA\"\n}".to_string(),
                                },
                            })]),
                        },
                        logprobs: None,
                        finish_reason: Some(FinishReason::ToolCalls),
                    }],
                    usage: Usage {
                        prompt_tokens: 82,
                        completion_tokens: 17,
                        total_tokens: 99,
                    },
                    system_fingerprint: None,
                },
            ),
            (
                "logprobs",
                r#"{
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1702685778,
                    "model": "gpt-3.5-turbo-0613",
                    "choices": [
                      {
                        "index": 0,
                        "message": {
                          "role": "assistant",
                          "content": "Hello! How can I assist you today?"
                        },
                        "logprobs": {
                          "content": [
                            {
                              "token": "Hello",
                              "logprob": -0.31725305,
                              "bytes": [72, 101, 108, 108, 111],
                              "top_logprobs": [
                                {
                                  "token": "Hello",
                                  "logprob": -0.31725305,
                                  "bytes": [72, 101, 108, 108, 111]
                                },
                                {
                                  "token": "Hi",
                                  "logprob": -1.3190403,
                                  "bytes": [72, 105]
                                }
                              ]
                            },
                            {
                              "token": "!",
                              "logprob": -0.02380986,
                              "bytes": [
                                33
                              ],
                              "top_logprobs": [
                                {
                                  "token": "!",
                                  "logprob": -0.02380986,
                                  "bytes": [33]
                                },
                                {
                                  "token": " there",
                                  "logprob": -3.787621,
                                  "bytes": [32, 116, 104, 101, 114, 101]
                                }
                              ]
                            },
                            {
                              "token": " How",
                              "logprob": -0.000054669687,
                              "bytes": [32, 72, 111, 119],
                              "top_logprobs": [
                                {
                                  "token": " How",
                                  "logprob": -0.000054669687,
                                  "bytes": [32, 72, 111, 119]
                                },
                                {
                                  "token": "<|end|>",
                                  "logprob": -10.953937,
                                  "bytes": null
                                }
                              ]
                            },
                            {
                              "token": " can",
                              "logprob": -0.015801601,
                              "bytes": [32, 99, 97, 110],
                              "top_logprobs": [
                                {
                                  "token": " can",
                                  "logprob": -0.015801601,
                                  "bytes": [32, 99, 97, 110]
                                },
                                {
                                  "token": " may",
                                  "logprob": -4.161023,
                                  "bytes": [32, 109, 97, 121]
                                }
                              ]
                            },
                            {
                              "token": " I",
                              "logprob": -3.7697225e-6,
                              "bytes": [
                                32,
                                73
                              ],
                              "top_logprobs": [
                                {
                                  "token": " I",
                                  "logprob": -3.7697225e-6,
                                  "bytes": [32, 73]
                                },
                                {
                                  "token": " assist",
                                  "logprob": -13.596657,
                                  "bytes": [32, 97, 115, 115, 105, 115, 116]
                                }
                              ]
                            },
                            {
                              "token": " assist",
                              "logprob": -0.04571125,
                              "bytes": [32, 97, 115, 115, 105, 115, 116],
                              "top_logprobs": [
                                {
                                  "token": " assist",
                                  "logprob": -0.04571125,
                                  "bytes": [32, 97, 115, 115, 105, 115, 116]
                                },
                                {
                                  "token": " help",
                                  "logprob": -3.1089056,
                                  "bytes": [32, 104, 101, 108, 112]
                                }
                              ]
                            },
                            {
                              "token": " you",
                              "logprob": -5.4385737e-6,
                              "bytes": [32, 121, 111, 117],
                              "top_logprobs": [
                                {
                                  "token": " you",
                                  "logprob": -5.4385737e-6,
                                  "bytes": [32, 121, 111, 117]
                                },
                                {
                                  "token": " today",
                                  "logprob": -12.807695,
                                  "bytes": [32, 116, 111, 100, 97, 121]
                                }
                              ]
                            },
                            {
                              "token": " today",
                              "logprob": -0.0040071653,
                              "bytes": [32, 116, 111, 100, 97, 121],
                              "top_logprobs": [
                                {
                                  "token": " today",
                                  "logprob": -0.0040071653,
                                  "bytes": [32, 116, 111, 100, 97, 121]
                                },
                                {
                                  "token": "?",
                                  "logprob": -5.5247097,
                                  "bytes": [63]
                                }
                              ]
                            },
                            {
                              "token": "?",
                              "logprob": -0.0008108172,
                              "bytes": [63],
                              "top_logprobs": [
                                {
                                  "token": "?",
                                  "logprob": -0.0008108172,
                                  "bytes": [63]
                                },
                                {
                                  "token": "?\n",
                                  "logprob": -7.184561,
                                  "bytes": [63, 10]
                                }
                              ]
                            }
                          ]
                        },
                        "finish_reason": "stop"
                      }
                    ],
                    "usage": {
                      "prompt_tokens": 9,
                      "completion_tokens": 9,
                      "total_tokens": 18
                    },
                    "system_fingerprint": null
                  }
                  "#,
                Response {
                    id: "chatcmpl-123".to_string(),
                    object: "chat.completion".to_string(),
                    created: 1702685778,
                    model: "gpt-3.5-turbo-0613".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        message: Message {
                            role: Role::Assistant,
                            content: Some("Hello! How can I assist you today?".to_string()),
                            ..Default::default()
                        },
                        logprobs: Some(Logprobs {
                            content: Some(vec![
                                LogprobContent {
                                    token: "Hello".to_string(),
                                    logprob: -0.31725305,
                                    bytes: Some(vec![72, 101, 108, 108, 111]),
                                    top_logprobs: vec![
                                        TopLogprobs {
                                            token: "Hello".to_string(),
                                            logprob: -0.31725305,
                                            bytes: Some(vec![72, 101, 108, 108, 111]),
                                        },
                                        TopLogprobs {
                                            token: "Hi".to_string(),
                                            logprob: -1.3190403,
                                            bytes: Some(vec![72, 105]),
                                        },
                                    ],
                                },
                                LogprobContent {
                                    token: "!".to_string(),
                                    logprob: -0.02380986,
                                    bytes: Some(vec![33]),
                                    top_logprobs: vec![
                                        TopLogprobs {
                                            token: "!".to_string(),
                                            logprob: -0.02380986,
                                            bytes: Some(vec![33]),
                                        },
                                        TopLogprobs {
                                            token: " there".to_string(),
                                            logprob: -3.787621,
                                            bytes: Some(vec![32, 116, 104, 101, 114, 101]),
                                        },
                                    ],
                                },
                                LogprobContent {
                                    token: " How".to_string(),
                                    logprob: -0.000054669687,
                                    bytes: Some(vec![32, 72, 111, 119]),
                                    top_logprobs: vec![
                                        TopLogprobs {
                                            token: " How".to_string(),
                                            logprob: -0.000054669687,
                                            bytes: Some(vec![32, 72, 111, 119]),
                                        },
                                        TopLogprobs {
                                            token: "<|end|>".to_string(),
                                            logprob: -10.953937,
                                            bytes: None,
                                        },
                                    ],
                                },
                                LogprobContent {
                                    token: " can".to_string(),
                                    logprob: -0.015801601,
                                    bytes: Some(vec![32, 99, 97, 110]),
                                    top_logprobs: vec![
                                        TopLogprobs {
                                            token: " can".to_string(),
                                            logprob: -0.015801601,
                                            bytes: Some(vec![32, 99, 97, 110]),
                                        },
                                        TopLogprobs {
                                            token: " may".to_string(),
                                            logprob: -4.161023,
                                            bytes: Some(vec![32, 109, 97, 121]),
                                        },
                                    ],
                                },
                                LogprobContent {
                                    token: " I".to_string(),
                                    logprob: -3.7697225e-6,
                                    bytes: Some(vec![32, 73]),
                                    top_logprobs: vec![
                                        TopLogprobs {
                                            token: " I".to_string(),
                                            logprob: -3.7697225e-6,
                                            bytes: Some(vec![32, 73]),
                                        },
                                        TopLogprobs {
                                            token: " assist".to_string(),
                                            logprob: -13.596657,
                                            bytes: Some(vec![32, 97, 115, 115, 105, 115, 116]),
                                        },
                                    ],
                                },
                                LogprobContent {
                                    token: " assist".to_string(),
                                    logprob: -0.04571125,
                                    bytes: Some(vec![32, 97, 115, 115, 105, 115, 116]),
                                    top_logprobs: vec![
                                        TopLogprobs {
                                            token: " assist".to_string(),
                                            logprob: -0.04571125,
                                            bytes: Some(vec![32, 97, 115, 115, 105, 115, 116]),
                                        },
                                        TopLogprobs {
                                            token: " help".to_string(),
                                            logprob: -3.1089056,
                                            bytes: Some(vec![32, 104, 101, 108, 112]),
                                        },
                                    ],
                                },
                                LogprobContent {
                                    token: " you".to_string(),
                                    logprob: -5.4385737e-6,
                                    bytes: Some(vec![32, 121, 111, 117]),
                                    top_logprobs: vec![
                                        TopLogprobs {
                                            token: " you".to_string(),
                                            logprob: -5.4385737e-6,
                                            bytes: Some(vec![32, 121, 111, 117]),
                                        },
                                        TopLogprobs {
                                            token: " today".to_string(),
                                            logprob: -12.807695,
                                            bytes: Some(vec![32, 116, 111, 100, 97, 121]),
                                        },
                                    ],
                                },
                                LogprobContent {
                                    token: " today".to_string(),
                                    logprob: -0.0040071653,
                                    bytes: Some(vec![32, 116, 111, 100, 97, 121]),
                                    top_logprobs: vec![
                                        TopLogprobs {
                                            token: " today".to_string(),
                                            logprob: -0.0040071653,
                                            bytes: Some(vec![32, 116, 111, 100, 97, 121]),
                                        },
                                        TopLogprobs {
                                            token: "?".to_string(),
                                            logprob: -5.5247097,
                                            bytes: Some(vec![63]),
                                        },
                                    ],
                                },
                                LogprobContent {
                                    token: "?".to_string(),
                                    logprob: -0.0008108172,
                                    bytes: Some(vec![63]),
                                    top_logprobs: vec![
                                        TopLogprobs {
                                            token: "?".to_string(),
                                            logprob: -0.0008108172,
                                            bytes: Some(vec![63]),
                                        },
                                        TopLogprobs {
                                            token: "?\n".to_string(),
                                            logprob: -7.184561,
                                            bytes: Some(vec![63, 10]),
                                        },
                                    ],
                                },
                            ]),
                        }),
                        finish_reason: Some(FinishReason::Stop),
                    }],
                    usage: Usage {
                        prompt_tokens: 9,
                        completion_tokens: 9,
                        total_tokens: 18,
                    },
                    system_fingerprint: None,
                },
            ),
        ];
        for (name, json, expected) in tests {
            //test deserialize
            let actual: Response = serde_json::from_str(json).unwrap();
            assert_eq!(actual, expected, "deserialize test failed: {}", name);
            //test serialize
            let serialized = serde_json::to_string(&expected).unwrap();
            let actual: Response = serde_json::from_str(&serialized).unwrap();
            assert_eq!(actual, expected, "serialize test failed: {}", name);
        }
    }
}
