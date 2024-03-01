use std::fmt;

use super::{deserialize_obj_or_arr, deserialize_option_obj_or_arr};

use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct RequestBody {
    #[serde(deserialize_with = "deserialize_obj_or_arr")]
    contents: Vec<Content>,
    /// A piece of code that enables the system to interact with external systems to perform an action, or set of actions, outside of knowledge and scope of the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_settings: Option<Vec<SafetySetting>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerateionConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Content {
    role: Role,
    #[serde(deserialize_with = "deserialize_obj_or_arr")]
    parts: Vec<ContentPart>,
}

///The role in a conversation associated with the content. Specifying a role is required even in singleturn use cases. Acceptable values include the following:
///USER: Specifies content that's sent by you.
///MODEL: Specifies the model's response.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]

pub enum Role {
    User,
    Model,
}

/// Ordered parts that make up the input. Parts may have different MIME types.
/// For gemini-1.0-pro, only the text field is valid. The token limit is 32k.
/// For gemini-1.0-pro-vision, you may specify either text only, text and up to 16 images, or text and 1 video. The token limit is 16k.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ContentPart {
    /// The text instructions or chat dialogue to include in the prompt.
    Text(TextData),
    /// Serialized bytes data of the image or video. You can specify at most 1 image with inlineData. To specify up to 16 images, use fileData.
    Inline(InlineData),
    File(FileData),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct TextData {
    text: String,
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// One or more function declarations. Each function declaration contains information about one function that includes the following:
    /// name The name of the function to call. Must start with a letter or an underscore. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    /// description (optional). The description and purpose of the function. The model uses this to decide how and whether to call the function. For the best results, we recommend that you include a description.
    /// parameters The parameters of this function in a format that's compatible with the OpenAPI schema format.
    /// For more information, see Function calling.
    function_declarations: Vec<FunctionTool>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FunctionTool {
    name: String,
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SafetySetting {
    category: SafetySettingCategory,
    threshhold: SafetySettingThreshold,
}

/// The safety category to configure a threshold for. Acceptable values include the following:
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetySettingCategory {
    HarmCategorySexuallyExplicit,
    HarmCategoryHateSpeech,
    HarmCategoryHarassment,
    HarmCategoryDangerousContent,
}

/// The threshold for blocking responses that could belong to the specified safety category based on probability.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetySettingThreshold {
    BlockNone,
    BlockLowAndAbove,
    BlockMedAndAbove,
    BlockOnlyHigh,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateionConfig {
    /// The temperature is used for sampling during the response generation, which occurs when topP and topK are applied. Temperature controls the degree of randomness in token selection. Lower temperatures are good for prompts that require a more deterministic and less open-ended or creative response, while higher temperatures can lead to more diverse or creative results. A temperature of 0 is deterministic: the highest probability response is always selected.
    /// Range: 0.0 - 1.0
    /// Default for gemini-1.0-pro: 0.9
    /// Default for gemini-1.0-pro-vision: 0.4
    temperature: f32,
    /// Top-P changes how the model selects tokens for output. Tokens are selected from the most (see top-K) to least probable until the sum of their probabilities equals the top-P value. For example, if tokens A, B, and C have a probability of 0.3, 0.2, and 0.1 and the top-P value is 0.5, then the model will select either A or B as the next token by using temperature and excludes C as a candidate.
    /// Specify a lower value for less random responses and a higher value for more random responses.
    /// Range: 0.0 - 1.0
    /// Default: 1.0
    top_p: f32,
    /// Top-K changes how the model selects tokens for output. A top-K of 1 means the next selected token is the most probable among all tokens in the model's vocabulary (also called greedy decoding), while a top-K of 3 means that the next token is selected from among the three most probable tokens by using temperature.
    /// For each token selection step, the top-K tokens with the highest probabilities are sampled. Then tokens are further filtered based on top-P with the final token selected using temperature sampling.
    /// Specify a lower value for less random responses and a higher value for more random responses.
    /// Range: 1-40
    /// Default for gemini-1.0-pro-vision: 32
    /// Default for gemini-1.0-pro: none
    top_k: Option<u32>,
    /// The number of response variations to return.
    /// This value must be 1.
    candidate_count: u32,
    /// Maximum number of tokens that can be generated in the response. A token is approximately four characters. 100 tokens correspond to roughly 60-80 words.
    /// Specify a lower value for shorter responses and a higher value for potentially longer responses.
    /// Range for gemini-1.0-pro: 1-8192 (default: 8192)
    /// Range for gemini-1.0-pro-vision: 1-2048 (default: 2048)
    max_output_tokens: u64,
    /// Specifies a list of strings that tells the model to stop generating text if one of the strings is encountered in the response. If a string appears multiple times in the response, then the response truncates where it's first encountered. The strings are case-sensitive.
    /// For example, if the following is the returned response when stopSequences isn't specified:
    /// public static string reverse(string myString)
    /// Then the returned response with stopSequences set to ["Str","reverse"] is:
    /// public static string
    /// Maximum 5 items in the list.
    stop_sequences: Option<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serde() {
        let tests = vec![(
            "simple",
            r#"{"contents": {"role": "user","parts": {"text": "Give me a recipe for banana bread."}}}"#,
            RequestBody {
                contents: vec![Content {
                    role: Role::User,
                    parts: vec![ContentPart::Text(TextData {
                        text: "Give me a recipe for banana bread.".to_string(),
                    })],
                }],
                ..Default::default()
            },
        )];
        for (name, json, expected) in tests {
            //test deserialize
            let actual: RequestBody = serde_json::from_str(json).unwrap();
            assert_eq!(actual, expected, "deserialize test failed: {}", name);
            //test serialize
            let serialized = serde_json::to_string(&expected).unwrap();
            let actual: RequestBody = serde_json::from_str(&serialized).unwrap();
            assert_eq!(actual, expected, "serialize test failed: {}", name);
        }
    }
}
