use serde::{Deserialize, Serialize};
pub mod request;
#[allow(unused_imports)]
pub use request::*;
pub mod response;
#[allow(unused_imports)]
pub use response::*;
pub mod stream_response;
#[allow(unused_imports)]
pub use stream_response::*;

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

impl Message {
    //return true if text or all blocks are empty or only contain white spaces
    pub fn is_all_empty(&self) -> bool {
        self.content.is_all_empty()
    }
}

#[derive(Debug, Deserialize, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    #[default]
    User,
    Assistant,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl MessageContent {
    pub fn is_all_empty(&self) -> bool {
        match self {
            MessageContent::Text(s) => s.trim().is_empty(),
            MessageContent::Blocks(blocks) => {
                if blocks.is_empty() {
                    return true;
                }
                for block in blocks {
                    if !block.is_empty() {
                        return false;
                    }
                }
                true
            }
        }
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

impl ContentBlock {
    pub fn is_empty(&self) -> bool {
        match self {
            ContentBlock::Text { text } => text.trim().is_empty(),
            ContentBlock::Image { source } => match source {
                ImageSource::Base64 { media_type, data } => {
                    media_type.trim().is_empty() || data.trim().is_empty()
                }
            },
            ContentBlock::TextDelta { text } => text.trim().is_empty(),
            ContentBlock::ToolUse { input, .. } => input.is_null(),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
            } => tool_use_id.trim().is_empty() || content.trim().is_empty(),
        }
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ImageSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

#[derive(Debug, Deserialize, Default, Clone, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens: Option<usize>,
    pub output_tokens: usize,
}
