use serde::{Deserialize, Deserializer, Serialize};

use crate::util::deserialize_obj_or_vec;

pub mod request;
pub mod response;
pub use request::*;
pub use response::*;

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct Content {
    //todo: it can be option?
    #[serde(deserialize_with = "deserialize_role")]
    role: Role,
    #[serde(deserialize_with = "deserialize_obj_or_vec")]
    parts: Vec<Part>,
}

///The role in a conversation associated with the content. Specifying a role is required even in singleturn use cases. Acceptable values include the following:
///USER: Specifies content that's sent by you.
///MODEL: Specifies the model's response.
#[derive(Debug, Serialize, Deserialize, Clone, Default, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    #[default]
    User,
    Model,
}

fn deserialize_role<'de, D>(deserializer: D) -> Result<Role, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    let ss = s.to_lowercase();
    match ss.as_str() {
        "user" => Ok(Role::User),
        "model" => Ok(Role::Model),
        _ => Err(serde::de::Error::custom("Invalid value for Role")),
    }
}

/// Ordered parts that make up the input. Parts may have different MIME types.
/// For gemini-1.0-pro, only the text field is valid. The token limit is 32k.
/// For gemini-1.0-pro-vision, you may specify either text only, text and up to 16 images, or text and 1 video. The token limit is 16k.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum Part {
    /// The text instructions or chat dialogue to include in the prompt.
    #[serde(rename = "text")]
    Text(String),
    /// Serialized bytes data of the image or video. You can specify at most 1 image with inlineData. To specify up to 16 images, use fileData.
    #[serde(rename = "inlineData")]
    Inline(InlineData),
    #[serde(rename = "fileData")]
    File(FileData),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    /// The media type of the image or video specified in the data or fileUri fields. Acceptable values include the following:
    ///
    /// image/png
    /// image/jpeg
    /// video/mov
    /// video/mpeg
    /// video/mp4
    /// video/mpg
    /// video/avi
    /// video/wmv
    /// video/mpegps
    /// video/flv
    ///
    ///
    /// Maximum video length: 2 minutes.
    ///
    /// No limit on image resolution.
    mime_type: String,
    /// The base64 encoding of the image or video to include inline in the prompt. When including media inline, you must also specify MIMETYPE.
    /// size limit: 20MB
    data: String,
    video_metadata: Option<VideoMetadata>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    mime_type: String,
    ///The Cloud Storage URI of the image or video to include in the prompt. The bucket that stores the file must be in the same Google Cloud project that's sending the request. You must also specify MIMETYPE.
    ///size limit: 20MB
    file_uri: String,
    video_metadata: Option<VideoMetadata>,
}

/// Optional. For video input, the start and end offset of the video in Duration format. For example, to specify a 10 second clip starting at 1:00, set "start_offset": { "seconds": 60 } and "end_offset": { "seconds": 70 }.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct VideoMetadata {
    start_offset: VideoOffset,
    end_offset: VideoOffset,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct VideoOffset {
    seconds: i64,
    nanos: i32,
}

/// The category of a rating.
/// These categories cover various kinds of harms that developers may wish to adjust.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum HarmCategory {
    /// Sexually explicit content.
    #[serde(rename = "HARM_CATEGORY_SEXUALLY_EXPLICIT")]
    SexuallyExplicit,

    /// Hate speech and content.
    #[serde(rename = "HARM_CATEGORY_HATE_SPEECH")]
    HateSpeech,

    /// Harassment content.
    #[serde(rename = "HARM_CATEGORY_HARASSMENT")]
    Harassment,

    /// Dangerous content.
    #[serde(rename = "HARM_CATEGORY_DANGEROUS_CONTENT")]
    DangerousContent,
}
