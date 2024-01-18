use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::{
    chat_completion_object::{Logprobs, Role},
    create_chat_completion::{FinishReason, ToolCall},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Chunk {
    Done,
    Data(ChunkResponse),
}

impl FromStr for Chunk {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "[DONE]" => Ok(Chunk::Done),
            _ => {
                let response = serde_json::from_str::<ChunkResponse>(s)?;
                Ok(Chunk::Data(response))
            }
        }
    }
}

pub struct ChunkResponseDone;

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct ChunkResponse {
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
}

#[derive(Debug, Default, Deserialize, Serialize, Clone, PartialEq)]
pub struct Choice {
    pub index: u32,
    pub delta: DeltaMessage,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence,
    /// `length` if the maximum number of tokens specified in the request was reached,
    /// `content_filter` if content was omitted due to a flag from our content filters,
    /// `tool_calls` if the model called a tool, or `function_call` (deprecated) if the model called a function.
    pub finish_reason: Option<FinishReason>,
    /// Log probability information for the choice.
    pub logprobs: Option<Logprobs>,
}
#[derive(Debug, Deserialize, Serialize, Default, Clone, PartialEq)]
pub struct DeltaMessage {
    /// The contents of the message.
    pub content: Option<String>,

    /// The tool calls generated by the model, such as function calls.
    pub tool_calls: Option<Vec<ToolCall>>,

    /// The role of the author of this message.
    pub role: Option<Role>,
}

#[cfg(test)]
mod tests {
    use crate::entity::chat_completion_object::Role;

    use super::*;

    #[test]
    fn serde() {
        let tests = vec![
            (
                "start",
                r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}"#,
                ChunkResponse {
                    id: "chatcmpl-123".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 1694268190,
                    model: "gpt-3.5-turbo-0613".to_string(),
                    system_fingerprint: Some("fp_44709d6fcb".to_string()),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: Some(Role::Assistant),
                            content: Some("".to_string()),
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                },
            ),
            (
                "data",
                r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"!"},"logprobs":null,"finish_reason":null}]}"#,
                ChunkResponse {
                    id: "chatcmpl-123".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 1694268190,
                    model: "gpt-3.5-turbo-0613".to_string(),
                    system_fingerprint: Some("fp_44709d6fcb".to_string()),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            content: Some("!".to_string()),
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                },
            ),
            (
                "end",
                r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}"#,
                ChunkResponse {
                    id: "chatcmpl-123".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 1694268190,
                    model: "gpt-3.5-turbo-0613".to_string(),
                    system_fingerprint: Some("fp_44709d6fcb".to_string()),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            ..Default::default()
                        },
                        finish_reason: Some(FinishReason::Stop),
                        ..Default::default()
                    }],
                },
            ),
        ];
        for (name, json, expected) in tests {
            //test deserialize
            let actual: ChunkResponse = serde_json::from_str(json).unwrap();
            assert_eq!(actual, expected, "deserialize test failed: {}", name);
            //test serialize
            let serialized = serde_json::to_string(&expected).unwrap();
            let actual: ChunkResponse = serde_json::from_str(&serialized).unwrap();
            assert_eq!(actual, expected, "serialize test failed: {}", name);

            //test enum
            let got: Chunk = json.parse().unwrap();
            let want = Chunk::Data(expected);
            assert_eq!(got, want, "enum test failed: {}", name)
        }
    }

    #[test]
    fn test_done() {
        let input = "[DONE]";
        let want = Chunk::Done;
        let got: Chunk = input.parse().unwrap();
        assert_eq!(want, got, "test [DONE]");
    }
}
