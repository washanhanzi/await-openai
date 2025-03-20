use std::{
    convert::Infallible,
    fmt::{self, Display, Formatter},
    str::FromStr,
};

use serde::{Deserialize, Serialize};

use super::{response::Response, ContentBlock, StopReason, Usage};

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventName {
    Unspecified,
    Error,
    MessageStart,
    ContentBlockDelta,
    ContentBlockStart,
    Ping,
    ContentBlockStop,
    MessageDelta,
    MessageStop,
}

impl FromStr for EventName {
    type Err = Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "error" => Ok(EventName::Error),
            "message_start" => Ok(EventName::MessageStart),
            "content_block_start" => Ok(EventName::ContentBlockStart),
            "ping" => Ok(EventName::Ping),
            "content_block_delta" => Ok(EventName::ContentBlockDelta),
            "content_block_stop" => Ok(EventName::ContentBlockStop),
            "message_delta" => Ok(EventName::MessageDelta),
            "message_stop" => Ok(EventName::MessageStop),
            _ => Ok(EventName::Unspecified),
        }
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum EventData {
    Error {
        error: ErrorData,
    },
    MessageStart {
        message: Response,
    },
    ContentBlockStart {
        index: u32,
        content_block: ContentBlock,
    },
    Ping,
    ContentBlockDelta {
        index: u32,
        delta: ContentBlock,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: Usage,
    },
    MessageStop,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ErrorData {
    OverloadedError { message: String },
    // Additional error types
    InternalServerError { message: String },
    BadRequestError { message: String },
    UnauthorizedError { message: String },
}

impl Display for ErrorData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ErrorData::OverloadedError { message } => write!(f, "OverloadedError: {}", message),
            ErrorData::InternalServerError { message } => write!(f, "InternalServerError: {}", message),
            ErrorData::BadRequestError { message } => write!(f, "BadRequestError: {}", message),
            ErrorData::UnauthorizedError { message } => write!(f, "UnauthorizedError: {}", message),
        }
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct MessageDelta {
    pub stop_reason: StopReason,
    pub stop_sequence: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::Role;
    #[test]
    fn serde() {
        let tests = vec![
            (
                "error_overloaded",
                "error",
                r#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#,
                EventName::Error,
                EventData::Error {
                    error: ErrorData::OverloadedError {
                        message: "Overloaded".to_string(),
                    },
                },
            ),
            (
                "message_start_empty_content",
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_019LBLYFJ7fG3fuAqzuRQbyi","type":"message","role":"assistant","content":[],"model":"claude-3-opus-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}"#,
                EventName::MessageStart,
                EventData::MessageStart {
                    message: Response {
                        id: "msg_019LBLYFJ7fG3fuAqzuRQbyi".to_string(),
                        r#type: "message".to_string(),
                        role: Role::Assistant,
                        content: vec![],
                        model: "claude-3-opus-20240229".to_string(),
                        stop_reason: None,
                        stop_sequence: None,
                        usage: Usage {
                            input_tokens: Some(10),
                            output_tokens: 1,
                        },
                    },
                },
            ),
            (
                "content_block_start_empty_text",
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                EventName::ContentBlockStart,
                EventData::ContentBlockStart {
                    index: 0,
                    content_block: ContentBlock::Text {
                        text: "".to_string(),
                    },
                },
            ),
            (
                "ping_event",
                "ping",
                r#"{"type": "ping"}"#,
                EventName::Ping,
                EventData::Ping,
            ),
            (
                "content_block_delta_hello",
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
                EventName::ContentBlockDelta,
                EventData::ContentBlockDelta {
                    index: 0,
                    delta: ContentBlock::TextDelta {
                        text: "Hello".to_string(),
                    },
                },
            ),
            (
                "content_block_delta_exclamation",
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}"#,
                EventName::ContentBlockDelta,
                EventData::ContentBlockDelta {
                    index: 0,
                    delta: ContentBlock::TextDelta {
                        text: "!".to_string(),
                    },
                },
            ),
            (
                "content_block_stop_index_0",
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
                EventName::ContentBlockStop,
                EventData::ContentBlockStop { index: 0 },
            ),
            (
                "message_delta_end_turn",
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}"#,
                EventName::MessageDelta,
                EventData::MessageDelta {
                    delta: MessageDelta {
                        stop_reason: StopReason::EndTurn,
                        stop_sequence: None,
                    },
                    usage: Usage {
                        input_tokens: None,
                        output_tokens: 12,
                    },
                },
            ),
            (
                "message_stop_event",
                "message_stop",
                r#"{"type":"message_stop"}"#,
                EventName::MessageStop,
                EventData::MessageStop,
            ),
            // New test cases based on the latest Anthropic API documentation
            (
                "content_block_start_tool_use",
                "content_block_start",
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_01AbCdEfGhIjKlMnOpQrStUv","name":"weather_forecast","input":{}}}"#,
                EventName::ContentBlockStart,
                EventData::ContentBlockStart {
                    index: 1,
                    content_block: ContentBlock::ToolUse {
                        id: "tu_01AbCdEfGhIjKlMnOpQrStUv".to_string(),
                        name: "weather_forecast".to_string(),
                        input: serde_json::json!({}),
                    },
                },
            ),
            (
                "content_block_delta_input_json_start",
                "content_block_delta",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\": \"San Fra\"}"}}"#,
                EventName::ContentBlockDelta,
                EventData::ContentBlockDelta {
                    index: 1,
                    delta: ContentBlock::InputJsonDelta {
                        partial_json: "{\"location\": \"San Fra\"}".to_string(),
                    },
                },
            ),
            (
                "content_block_delta_input_json_continuation",
                "content_block_delta",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"ncisco\"}"}}"#,
                EventName::ContentBlockDelta,
                EventData::ContentBlockDelta {
                    index: 1,
                    delta: ContentBlock::InputJsonDelta {
                        partial_json: "ncisco\"}".to_string(),
                    },
                },
            ),
            (
                "content_block_start_thinking",
                "content_block_start",
                r#"{"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":"","signature":null}}"#,
                EventName::ContentBlockStart,
                EventData::ContentBlockStart {
                    index: 2,
                    content_block: ContentBlock::Thinking {
                        thinking: "".to_string(),
                        signature: None,
                    },
                },
            ),
            (
                "content_block_delta_thinking",
                "content_block_delta",
                r#"{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"Let me solve this step by step:\n\n1. First break down 27 * 453"}}"#,
                EventName::ContentBlockDelta,
                EventData::ContentBlockDelta {
                    index: 2,
                    delta: ContentBlock::ThinkingDelta {
                        thinking: "Let me solve this step by step:\n\n1. First break down 27 * 453".to_string(),
                    },
                },
            ),
            (
                "content_block_delta_signature",
                "content_block_delta",
                r#"{"type":"content_block_delta","index":2,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}"#,
                EventName::ContentBlockDelta,
                EventData::ContentBlockDelta {
                    index: 2,
                    delta: ContentBlock::SignatureDelta {
                        signature: "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds...".to_string(),
                    },
                },
            ),
            (
                "message_delta_max_tokens",
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":1024}}"#,
                EventName::MessageDelta,
                EventData::MessageDelta {
                    delta: MessageDelta {
                        stop_reason: StopReason::MaxTokens,
                        stop_sequence: None,
                    },
                    usage: Usage {
                        input_tokens: None,
                        output_tokens: 1024,
                    },
                },
            ),
            (
                "message_delta_stop_sequence",
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"STOP"},"usage":{"output_tokens":45}}"#,
                EventName::MessageDelta,
                EventData::MessageDelta {
                    delta: MessageDelta {
                        stop_reason: StopReason::StopSequence,
                        stop_sequence: Some("STOP".to_string()),
                    },
                    usage: Usage {
                        input_tokens: None,
                        output_tokens: 45,
                    },
                },
            ),
            (
                "content_block_start_tool_result",
                "content_block_start",
                r#"{"type":"content_block_start","index":3,"content_block":{"type":"tool_result","tool_use_id":"tu_01AbCdEfGhIjKlMnOpQrStUv","content":"{\"temperature\":22,\"conditions\":\"sunny\",\"location\":\"San Francisco, CA\"}"}}"#,
                EventName::ContentBlockStart,
                EventData::ContentBlockStart {
                    index: 3,
                    content_block: ContentBlock::ToolResult {
                        tool_use_id: "tu_01AbCdEfGhIjKlMnOpQrStUv".to_string(),
                        content: "{\"temperature\":22,\"conditions\":\"sunny\",\"location\":\"San Francisco, CA\"}".to_string(),
                    },
                },
            ),
        ];
        for (test_name, name, input, event_name, event_data) in tests {
            let got_event_name = EventName::from_str(name).unwrap();
            assert_eq!(
                got_event_name, event_name,
                "test failed for event name: {} ({})",
                name, test_name
            );

            let got_event_data: EventData = serde_json::from_str(input).unwrap();
            assert_eq!(
                got_event_data, event_data,
                "test failed for event data: {}",
                test_name
            );
        }
    }
}
