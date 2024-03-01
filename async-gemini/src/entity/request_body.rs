use super::{deserialize_obj_or_vec, deserialize_option_obj_or_vec};

use serde::{de, Deserialize, Deserializer, Serialize};

/// when deserilization:
/// - google api support both camelCase and snake_case key, but we only support camel case.
/// - google api allow trailling comma, but not here
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RequestBody {
    #[serde(deserialize_with = "deserialize_obj_or_vec")]
    contents: Vec<Content>,
    /// A piece of code that enables the system to interact with external systems to perform an action, or set of actions, outside of knowledge and scope of the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_option_obj_or_vec", default)]
    safety_settings: Option<Vec<SafetySetting>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerateionConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Content {
    #[serde(deserialize_with = "deserialize_role")]
    role: Role,
    #[serde(deserialize_with = "deserialize_obj_or_vec")]
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

fn deserialize_role<'de, D>(deserializer: D) -> Result<Role, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    let ss = s.to_lowercase();
    match ss.as_str() {
        "user" => Ok(Role::User),
        "model" => Ok(Role::Model),
        _ => Err(de::Error::custom("Invalid value for Role")),
    }
}
// impl<'de> Deserialize<'de> for Role {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: Deserializer<'de>,
//     {
//         let s = String::deserialize(deserializer)?;
//         match s.to_lowercase().as_str() {
//             "user" => Ok(Role::User),
//             "model" => Ok(Role::Model),
//             _ => Err(de::Error::custom("Invalid value for Role")),
//         }
//     }
// }

/// Ordered parts that make up the input. Parts may have different MIME types.
/// For gemini-1.0-pro, only the text field is valid. The token limit is 32k.
/// For gemini-1.0-pro-vision, you may specify either text only, text and up to 16 images, or text and 1 video. The token limit is 16k.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum ContentPart {
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
    threshold: SafetySettingThreshold,
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
    temperature: Option<f32>,
    /// Top-P changes how the model selects tokens for output. Tokens are selected from the most (see top-K) to least probable until the sum of their probabilities equals the top-P value. For example, if tokens A, B, and C have a probability of 0.3, 0.2, and 0.1 and the top-P value is 0.5, then the model will select either A or B as the next token by using temperature and excludes C as a candidate.
    /// Specify a lower value for less random responses and a higher value for more random responses.
    /// Range: 0.0 - 1.0
    /// Default: 1.0
    top_p: Option<f32>,
    /// Top-K changes how the model selects tokens for output. A top-K of 1 means the next selected token is the most probable among all tokens in the model's vocabulary (also called greedy decoding), while a top-K of 3 means that the next token is selected from among the three most probable tokens by using temperature.
    /// For each token selection step, the top-K tokens with the highest probabilities are sampled. Then tokens are further filtered based on top-P with the final token selected using temperature sampling.
    /// Specify a lower value for less random responses and a higher value for more random responses.
    /// Range: 1-40
    /// Default for gemini-1.0-pro-vision: 32
    /// Default for gemini-1.0-pro: none
    top_k: Option<u32>,
    /// The number of response variations to return.
    /// This value must be 1.
    candidate_count: Option<u32>,
    /// Maximum number of tokens that can be generated in the response. A token is approximately four characters. 100 tokens correspond to roughly 60-80 words.
    /// Specify a lower value for shorter responses and a higher value for potentially longer responses.
    /// Range for gemini-1.0-pro: 1-8192 (default: 8192)
    /// Range for gemini-1.0-pro-vision: 1-2048 (default: 2048)
    max_output_tokens: Option<u64>,
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
    use serde_json::json;

    use super::*;
    #[test]
    fn serde() {
        let tests = vec![
            (
                "simple",
                r#"{"contents": {"role": "user","parts": {"text": "Give me a recipe for banana bread."}}}"#,
                RequestBody {
                    contents: vec![Content {
                        role: Role::User,
                        parts: vec![ContentPart::Text(
                             "Give me a recipe for banana bread.".to_string(),
                        )],
                    }],
                    ..Default::default()
                },
            ),
            (
                "text",
                r#"{
                    "contents":
                    {
                        "role": "user",
                        "parts":
                        {
                            "text": "Give me a recipe for banana bread."
                        }
                    },
                    "safetySettings":
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_LOW_AND_ABOVE"
                    },
                    "generationConfig":
                    {
                        "temperature": 0.2,
                        "topP": 0.8,
                        "topK": 40
                    }
                }"#,
                RequestBody {
                    contents: vec![Content {
                        role: Role::User,
                        parts: vec![ContentPart::Text(
                             "Give me a recipe for banana bread.".to_string(),
                        )],
                    }],
                    safety_settings: Some(vec![SafetySetting {
                        category: SafetySettingCategory::HarmCategorySexuallyExplicit,
                        threshold: SafetySettingThreshold::BlockLowAndAbove,
                    }]),
                    generation_config: Some(GenerateionConfig {
                        temperature: Some(0.2),
                        top_p: Some(0.8),
                        top_k: Some(40),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ),
            (
                "chat",
                r#"{
                    "contents": [
                      {
                        "role": "USER",
                        "parts": { "text": "Hello!" }
                      },
                      {
                        "role": "MODEL",
                        "parts": { "text": "Argh! What brings ye to my ship?" }
                      },
                      {
                        "role": "USER",
                        "parts": { "text": "Wow! You are a real-life priate!" }
                      }
                    ],
                    "safetySettings": {
                      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                      "threshold": "BLOCK_LOW_AND_ABOVE"
                    },
                    "generationConfig": {
                      "temperature": 0.2,
                      "topP": 0.8,
                      "topK": 40,
                      "maxOutputTokens": 200
                    }
                  }"#,
                RequestBody {
                    contents: vec![
                        Content {
                            role: Role::User,
                            parts: vec![ContentPart::Text(
                                 "Hello!".to_string(),
                            )],
                        },
                        Content {
                            role: Role::Model,
                            parts: vec![ContentPart::Text(
                                 "Argh! What brings ye to my ship?".to_string(),
                            )],
                        },
                        Content {
                            role: Role::User,
                            parts: vec![ContentPart::Text(
                                 "Wow! You are a real-life priate!".to_string(),
                            )],
                        },
                    ],
                    safety_settings: Some(vec![SafetySetting {
                        category: SafetySettingCategory::HarmCategorySexuallyExplicit,
                        threshold: SafetySettingThreshold::BlockLowAndAbove,
                    }]),
                    generation_config: Some(GenerateionConfig {
                        temperature: Some(0.2),
                        top_p: Some(0.8),
                        top_k: Some(40),
                        max_output_tokens: Some(200),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ),
            (
                "multimodal",
                r#"{
                    "contents": {
                      "role": "user",
                      "parts": [
                        {
                          "fileData": {
                            "mimeType": "image/jpeg",
                            "fileUri": "gs://cloud-samples-data/ai-platform/flowers/daisy/10559679065_50d2b16f6d.jpg"
                          }
                        },
                        {
                          "text": "Describe this picture."
                        }
                      ]
                    },
                    "safetySettings": {
                      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                      "threshold": "BLOCK_LOW_AND_ABOVE"
                    },
                    "generationConfig": {
                      "temperature": 0.4,
                      "topP": 1.0,
                      "topK": 32,
                      "maxOutputTokens": 2048
                    }
                  }"#,
                RequestBody {
                    contents: vec![Content {
                        role: Role::User,
                        parts: vec![
                            ContentPart::File(FileData {
                                mime_type: "image/jpeg".to_string(),
                                file_uri: "gs://cloud-samples-data/ai-platform/flowers/daisy/10559679065_50d2b16f6d.jpg".to_string(),
                                video_metadata: None,
                            }),
                            ContentPart::Text(
                                 "Describe this picture.".to_string(),
                            ),
                        ],
                    }],
                    safety_settings: Some(vec![SafetySetting {
                        category: SafetySettingCategory::HarmCategorySexuallyExplicit,
                        threshold: SafetySettingThreshold::BlockLowAndAbove,
                    }]),
                    generation_config: Some(GenerateionConfig {
                        temperature: Some(0.4),
                        top_p: Some(1.0),
                        top_k: Some(32),
                        max_output_tokens: Some(2048),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ),
            (
                "function",
                r#"{
                    "contents": {
                      "role": "user",
                      "parts": {
                        "text": "Which theaters in Mountain View show Barbie movie?"
                      }
                    },
                    "tools": [
                      {
                        "functionDeclarations": [
                          {
                            "name": "find_movies",
                            "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
                            "parameters": {
                              "type": "object",
                              "properties": {
                                "location": {
                                  "type": "string",
                                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                                },
                                "description": {
                                  "type": "string",
                                  "description": "Any kind of description including category or genre, title words, attributes, etc."
                                }
                              },
                              "required": [
                                "description"
                              ]
                            }
                          },
                          {
                            "name": "find_theaters",
                            "description": "find theaters based on location and optionally movie title which are is currently playing in theaters",
                            "parameters": {
                              "type": "object",
                              "properties": {
                                "location": {
                                  "type": "string",
                                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                                },
                                "movie": {
                                  "type": "string",
                                  "description": "Any movie title"
                                }
                              },
                              "required": [
                                "location"
                              ]
                            }
                          },
                          {
                            "name": "get_showtimes",
                            "description": "Find the start times for movies playing in a specific theater",
                            "parameters": {
                              "type": "object",
                              "properties": {
                                "location": {
                                  "type": "string",
                                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                                },
                                "movie": {
                                  "type": "string",
                                  "description": "Any movie title"
                                },
                                "theater": {
                                  "type": "string",
                                  "description": "Name of the theater"
                                },
                                "date": {
                                  "type": "string",
                                  "description": "Date for requested showtime"
                                }
                              },
                              "required": [
                                "location",
                                "movie",
                                "theater",
                                "date"
                              ]
                            }
                          }
                        ]
                      }
                    ]
                  }"#,
                RequestBody {
                    contents: vec![Content {
                        role: Role::User,
                        parts: vec![
                            ContentPart::Text(
                                 "Which theaters in Mountain View show Barbie movie?".to_string(),
                            ),
                        ],
                    }],
                    tools:Some(vec![Tool{
                        function_declarations:vec![
                            FunctionTool{
                                name:"find_movies".to_string(),
                                description:Some("find movie titles currently playing in theaters based on any description, genre, title words, etc.".to_string()),
                                parameters:json!({
                              "type": "object",
                              "properties": {
                                "location": {
                                  "type": "string",
                                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                                },
                                "description": {
                                  "type": "string",
                                  "description": "Any kind of description including category or genre, title words, attributes, etc."
                                }
                              },
                              "required": [
                                "description"
                              ]
                            })
                            },
                            FunctionTool{
                                name:"find_theaters".to_string(),
                                description:Some("find theaters based on location and optionally movie title which are is currently playing in theaters".to_string()),
                                parameters:json!({
                              "type": "object",
                              "properties": {
                                "location": {
                                  "type": "string",
                                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                                },
                                "movie": {
                                  "type": "string",
                                  "description": "Any movie title"
                                }
                              },
                              "required": [
                                "location"
                              ]
                            })
                            },
                            FunctionTool{
                                name:"get_showtimes".to_string(),
                                description:Some("Find the start times for movies playing in a specific theater".to_string()),
                                parameters:json!({
                              "type": "object",
                              "properties": {
                                "location": {
                                  "type": "string",
                                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                                },
                                "movie": {
                                  "type": "string",
                                  "description": "Any movie title"
                                },
                                "theater": {
                                  "type": "string",
                                  "description": "Name of the theater"
                                },
                                "date": {
                                  "type": "string",
                                  "description": "Date for requested showtime"
                                }
                              },
                              "required": [
                                "location",
                                "movie",
                                "theater",
                                "date"
                              ]
                            })
                            }
                        ]
                    }]),
                    ..Default::default()
                },
            ),
        ];
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
