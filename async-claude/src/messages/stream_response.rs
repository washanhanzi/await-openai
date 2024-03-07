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
    ContentBlockStart,
    Ping,
    ContentBlockDelta,
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
}

impl Display for ErrorData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ErrorData::OverloadedError { message } => write!(f, "OverloadedError: {}", message),
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
                "ping",
                r#"{"type": "ping"}"#,
                EventName::Ping,
                EventData::Ping,
            ),
            (
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
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
                EventName::ContentBlockStop,
                EventData::ContentBlockStop { index: 0 },
            ),
            (
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
                "message_stop",
                r#"{"type":"message_stop"}"#,
                EventName::MessageStop,
                EventData::MessageStop,
            ),
        ];
        for (name, input, event_name, event_data) in tests {
            let got_event_name = EventName::from_str(name).unwrap();
            assert_eq!(
                got_event_name, event_name,
                "test failed for event name: {}",
                name
            );

            let got: EventData = serde_json::from_str(input).unwrap();
            assert_eq!(got, event_data, "test failed for event data: {}", name);
        }
    }
}
