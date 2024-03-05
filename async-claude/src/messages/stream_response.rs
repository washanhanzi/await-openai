use serde::{Deserialize, Serialize};

use super::{response::Response, ContentBlock, StopReason, Usage};

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventName {
    Unspecified,
    MessageStart,
    ContentBlockStart,
    Ping,
    ContentBlockDelta,
    ContentBlockStop,
    MessageDelta,
    MessageStop,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum StreamResponse {
    Error(ErrorData),
    MessageStart { message: Response },
    ContentBlockStart,
    Ping,
    ContentBlockDelta { index: u32, delta: ContentBlock },
    ContentBlockStop { index: u32 },
    MessageDelta { delta: MessageDelta },
    MessageStop,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ErrorData {
    OverloadedError { message: String },
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct MessageDelta {
    stop_reason: StopReason,
    stop_sequence: Option<String>,
    usage: Usage,
}
