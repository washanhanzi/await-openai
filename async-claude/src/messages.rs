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

    /// Validates that the message contains valid content for the Claude API
    pub fn validate(&self) -> Result<(), String> {
        self.content.validate()
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

    /// Validates that the message content only contains valid content block types for request body
    pub fn validate(&self) -> Result<(), String> {
        match self {
            MessageContent::Text(_) => Ok(()),
            MessageContent::Blocks(blocks) => {
                for (i, block) in blocks.iter().enumerate() {
                    if !matches!(block, ContentBlock::Base(_) | ContentBlock::RequestOnly(_)) {
                        return Err(format!(
                            "Invalid content block type at index {}: {:?}. Only Text, Image, ToolUse, ToolResult, Document, Thinking, and RedactedThinking are allowed in request body.",
                            i, block
                        ));
                    }
                }
                Ok(())
            }
        }
    }
}

// Base content block types that can be used in both request body and streaming
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum BaseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "tool_use")]
    ToolUse(ToolUseContentBlock),
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct ToolUseContentBlock {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

// Additional content block types that can only be used in request body
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum RequestOnlyContentBlock {
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "document")]
    Document {
        #[serde(skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

// Content blocks that can be used in request body (all types)
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ContentBlock {
    Base(BaseContentBlock),
    RequestOnly(RequestOnlyContentBlock),
    RedactedThinking(RedactedThinkingContentBlock),
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ResponseContentBlock {
    Base(BaseContentBlock),
    RedactedThinking(RedactedThinkingContentBlock),
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RedactedThinkingContentBlock {
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
}

impl ContentBlock {
    pub fn is_empty(&self) -> bool {
        match self {
            ContentBlock::Base(base) => match base {
                BaseContentBlock::Text { text } => text.trim().is_empty(),
                BaseContentBlock::ToolUse(tool_use) => {
                    tool_use.id.is_empty()
                        || tool_use.name.is_empty()
                        || !tool_use.input.is_object()
                }
                BaseContentBlock::Thinking { thinking, .. } => thinking.trim().is_empty(),
            },
            ContentBlock::RequestOnly(req_only) => match req_only {
                RequestOnlyContentBlock::Image { source } => match source {
                    ImageSource::Base64 { media_type, data } => {
                        media_type.trim().is_empty() || data.trim().is_empty()
                    }
                },
                RequestOnlyContentBlock::Document { source, id } => {
                    (source.is_none() || id.is_none())
                }
                RequestOnlyContentBlock::ToolResult {
                    tool_use_id,
                    content,
                } => tool_use_id.is_empty() || content.trim().is_empty(),
            },
            ContentBlock::RedactedThinking(redacted_thinking) => match redacted_thinking {
                RedactedThinkingContentBlock::RedactedThinking { data } => data.is_empty(),
            },
        }
    }
}

// Delta content blocks for streaming
// TODO: it should include a redacted_thinking delta, but the docuement dind't include it as example
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum DeltaContentBlock {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

impl DeltaContentBlock {
    pub fn is_empty(&self) -> bool {
        match self {
            DeltaContentBlock::TextDelta { text } => text.trim().is_empty(),
            DeltaContentBlock::InputJsonDelta { partial_json } => partial_json.is_empty(),
            DeltaContentBlock::ThinkingDelta { thinking } => thinking.is_empty(),
            DeltaContentBlock::SignatureDelta { signature } => signature.is_empty(),
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
#[serde(tag = "type")]
pub enum DocumentSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

#[derive(Debug, Deserialize, Default, Clone, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: u32,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}
