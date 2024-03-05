use serde::{Deserialize, Serialize};
mod request;
mod response;
mod stream_response;

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct Message {
    role: Role,
    content: MessageContent,
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

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
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
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct Usage {
    input_tokens: Option<u32>,
    output_tokens: u32,
}
