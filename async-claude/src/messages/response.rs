use serde::{Deserialize, Serialize};

use super::{ContentBlock, Role, StopReason, Usage};

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct Response {
    id: String,
    r#type: String,
    role: Role,
    content: Vec<ContentBlock>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<StopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequence: Option<String>,
    usage: Usage,
}
