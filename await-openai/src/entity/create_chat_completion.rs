use std::{borrow::Cow, collections::HashMap};

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct RequestBody {
    /// A list of messages comprising the conversation so far. [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).
    pub messages: Vec<Message>, // min: 1

    /// ID of the model to use.
    /// See the [model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility) table for details on which models work with the Chat API.
    pub model: String,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    ///
    /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>, // min: -2.0, max: 2.0, default: 0

    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100.
    /// Mathematically, the bias is added to the logits generated by the model prior to sampling.
    /// The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection;
    /// values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, serde_json::Value>>, // default: null

    /// Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the `content` of `message`. This option is currently not available on the `gpt-4-vision-preview` model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. `logprobs` must be set to `true` if this parameter is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>, // min: 0, max: 20

    /// The maximum number of [tokens](https://platform.openai.com/tokenizer) that can be generated in the chat completion.
    ///
    /// The total length of input tokens and generated tokens is limited by the model's context length.
    /// [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for counting tokens.
    ///
    /// This value is now deprecated in favor of max_completion_tokens.
    #[deprecated(note = "Use max_completion_tokens instead")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// **o-series models only**
    ///
    /// Constrains effort on reasoning for reasoning models. Currently supported values are `low`, `medium`, and `high`.
    /// Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,

    /// How many chat completion choices to generate for each input message. Note that you will be charged based on the number of generated tokens across all of the choices. Keep `n` as `1` to minimize costs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u8>, // min:1, max: 128, default: 1

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    ///
    /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>, // min: -2.0, max: 2.0, default 0

    /// An object specifying the format that the model must output. Compatible with `gpt-4-1106-preview` and `gpt-3.5-turbo-1106`.
    ///
    /// Setting to `{ "type": "json_object" }` enables JSON mode, which guarantees the message the model generates is valid JSON.
    ///
    /// **Important:** when using JSON mode, you **must** also instruct the model to produce JSON yourself via a system or user message. Without this, the model may generate an unending stream of whitespace until the generation reaches the token limit, resulting in a long-running and seemingly "stuck" request. Also note that the message content may be partially cut off if `finish_reason="length"`, which indicates the generation exceeded `max_tokens` or the conversation exceeded the max context length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    ///  This feature is in Beta.
    /// If specified, our system will make a best effort to sample deterministically, such that repeated requests
    /// with the same `seed` and parameters should return the same result.
    /// Determinism is not guaranteed, and you should refer to the `system_fingerprint` response parameter to monitor changes in the backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Up to 4 sequences where the API will stop generating further tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,

    /// If set, partial message deltas will be sent, like in ChatGPT.
    /// Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
    /// as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or `top_p` but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>, // min: 0, max: 2, default: 1,

    /// An alternative to sampling with temperature, called nucleus sampling,
    /// where the model considers the results of the tokens with top_p probability mass.
    /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    ///
    ///  We generally recommend altering this or `temperature` but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>, // min: 0, max: 1, default: 1

    /// A list of tools the model may call. Currently, only functions are supported as a tool.
    /// Use this to provide a list of functions the model may generate JSON inputs for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Whether or not to store the output of this chat completion request for use in our model distillation or evals products.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>, // default: false

    /// Developer-defined tags and values used for filtering completions in the dashboard.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// Whether to enable parallel function calling during tool use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>, // default: true

    /// Output types that you would like the model to generate for this request.
    /// Most models are capable of generating text, which is the default: ["text"]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,

    /// Configuration for a Predicted Output, which can greatly improve response times
    /// when large parts of the model response are known ahead of time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<PredictionConfig>,

    /// Parameters for audio output. Required when audio output is requested with modalities: ["audio"]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioConfig>,

    /// Specifies the latency tier to use for processing the request.
    /// This parameter is relevant for customers subscribed to the scale tier service.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>, // default: "auto"

    /// Options for streaming response. Only set this when you set stream: true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// This tool searches the web for relevant results to use in a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,

    /// Open router compatible field
    /// https://openrouter.ai/announcements/reasoning-tokens-for-thinking-models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<OpenRouterReasoning>,
}

impl RequestBody {
    pub fn first_user_message(&self) -> Option<&Message> {
        self.messages
            .iter()
            .find(|message| matches!(message, Message::User(_)))
    }

    pub fn first_user_message_text(&self) -> Option<String> {
        self.messages
            .iter()
            .find(|message| matches!(message, Message::User(_)))
            .and_then(|message| match message {
                Message::User(user_message) => match &user_message.content {
                    Content::Text(text) => Some(text.clone()),
                    Content::Array(array) => Some(
                        array
                            .iter()
                            .filter_map(|content_part| match content_part {
                                ContentPart::Text(text_part) => Some(text_part.text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<String>>()
                            .join(""),
                    ),
                },
                _ => None,
            })
    }

    pub fn last_user_message(&self) -> Option<&Message> {
        self.messages
            .iter()
            .rev()
            .find(|message| matches!(message, Message::User(_)))
    }

    pub fn last_user_message_text(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .find(|message| matches!(message, Message::User(_)))
            .and_then(|message| match message {
                Message::User(user_message) => match &user_message.content {
                    Content::Text(text) => Some(text.clone()),
                    Content::Array(array) => Some(
                        array
                            .iter()
                            .filter_map(|content_part| match content_part {
                                ContentPart::Text(text_part) => Some(text_part.text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<String>>()
                            .join(" "),
                    ),
                },
                _ => None,
            })
    }
}

pub struct RequestBodyBuilder {
    inner: RequestBody,
}

impl RequestBodyBuilder {
    pub fn new() -> Self {
        RequestBodyBuilder {
            inner: RequestBody::default(),
        }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.inner.model = model.into();
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.inner.messages = messages;
        self
    }

    pub fn push_user_message(mut self, message: impl Into<String>) -> Self {
        self.inner.messages.push(Message::User(UserMessage {
            content: Content::Text(message.into()),
            name: None,
        }));
        self
    }

    pub fn push_system_message(mut self, message: impl Into<String>) -> Self {
        self.inner.messages.push(Message::System(SystemMessage {
            content: message.into(),
            ..Default::default()
        }));
        self
    }

    pub fn prepend_system_message(mut self, message: impl Into<String>) -> Self {
        self.inner.messages.insert(
            0,
            Message::System(SystemMessage {
                content: message.into(),
                ..Default::default()
            }),
        );
        self
    }

    pub fn frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.inner.frequency_penalty = Some(frequency_penalty);
        self
    }

    pub fn logit_bias(mut self, logit_bias: HashMap<String, serde_json::Value>) -> Self {
        self.inner.logit_bias = Some(logit_bias);
        self
    }

    pub fn logprobs(mut self, logprobs: bool) -> Self {
        self.inner.logprobs = Some(logprobs);
        self
    }

    pub fn top_logprobs(mut self, top_logprobs: u8) -> Self {
        self.inner.top_logprobs = Some(top_logprobs);
        self
    }

    pub fn max_completion_tokens(mut self, max_completion_tokens: u32) -> Self {
        self.inner.max_completion_tokens = Some(max_completion_tokens);
        self
    }

    pub fn reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.inner.reasoning_effort = Some(reasoning_effort);
        self
    }

    pub fn n(mut self, n: u8) -> Self {
        self.inner.n = Some(n);
        self
    }

    pub fn presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.inner.presence_penalty = Some(presence_penalty);
        self
    }

    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.inner.response_format = Some(response_format);
        self
    }

    pub fn seed(mut self, seed: i64) -> Self {
        self.inner.seed = Some(seed);
        self
    }

    pub fn stop(mut self, stop: Stop) -> Self {
        self.inner.stop = Some(stop);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.inner.stream = Some(stream);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.inner.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.inner.top_p = Some(top_p);
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.inner.tools = Some(tools);
        self
    }

    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.inner.tool_choice = Some(tool_choice);
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.inner.user = Some(user.into());
        self
    }

    pub fn store(mut self, store: bool) -> Self {
        self.inner.store = Some(store);
        self
    }

    pub fn metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.inner.metadata = Some(metadata);
        self
    }

    pub fn parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.inner.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }

    pub fn modalities(mut self, modalities: Vec<String>) -> Self {
        self.inner.modalities = Some(modalities);
        self
    }

    pub fn prediction(mut self, prediction: PredictionConfig) -> Self {
        self.inner.prediction = Some(prediction);
        self
    }

    pub fn audio(mut self, audio: AudioConfig) -> Self {
        self.inner.audio = Some(audio);
        self
    }

    pub fn service_tier(mut self, service_tier: String) -> Self {
        self.inner.service_tier = Some(service_tier);
        self
    }

    pub fn stream_options(mut self, stream_options: StreamOptions) -> Self {
        self.inner.stream_options = Some(stream_options);
        self
    }

    pub fn web_search_options(mut self, web_search_options: WebSearchOptions) -> Self {
        self.inner.web_search_options = Some(web_search_options);
        self
    }

    pub fn build(self) -> RequestBody {
        self.inner
    }
}

impl Default for RequestBodyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Stop {
    String(String),
    Array(Vec<String>), // minItems: 1; maxItems: 4
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct Tool {
    pub r#type: ToolType,
    pub function: FunctionTool,
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub struct FunctionTool {
    /// The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    pub name: Cow<'static, str>,
    /// A description of what the function does, used by the model to choose when and how to call the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Cow<'static, str>>,
    /// The parameters the functions accepts, described as a JSON Schema object. See the [guide](https://platform.openai.com/docs/guides/text-generation/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.
    ///
    /// Omitting `parameters` defines a function with an empty parameter list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    #[default]
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct TopLogprobs {
    /// The token.
    pub token: String,
    /// The log probability of this token.
    pub logprob: f32,
    /// A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token.
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System(SystemMessage),
    #[serde(rename = "user")]
    User(UserMessage),
    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),
    #[serde(rename = "tool")]
    Tool(ToolMessage),
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct SystemMessage {
    /// The contents of the system message.
    pub content: String,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct UserMessage {
    /// The contents of the user message.
    pub content: Content,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Content {
    /// The text contents of the message.
    Text(String),
    ///  An array of content parts with a defined type, each can be of type `text` or `image_url`
    /// when passing in images. You can pass multiple images by adding multiple `image_url` content parts.
    ///  Image input is only supported when using the `gpt-4-vision-preview` model.
    Array(Vec<ContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text(TextContentPart),
    #[serde(rename = "image_url")]
    Image(ImageContentPart),
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct TextContentPart {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageUrlDetail {
    #[default]
    Auto,
    Low,
    High,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ImageUrl {
    /// Either a URL of the image or the base64 encoded image data.
    pub url: String,
    /// Specifies the detail level of the image. Learn more in the [Vision guide](https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding).
    pub detail: Option<ImageUrlDetail>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ImageContentPart {
    pub image_url: ImageUrl,
    /// The witdth and height of the image in pixels. the field is used for tokens calculation, it won't serilized, or deserialized when it's None.
    #[serde(skip_serializing)]
    pub dimensions: Option<(u32, u32)>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct AssistantMessage {
    /// The contents of the assistant message.
    pub content: Option<String>,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type")]
pub enum ToolCall {
    #[serde(rename = "function")]
    Function(ToolCallFunction),
}

impl ToolCall {
    pub fn id(&self) -> &str {
        match self {
            ToolCall::Function(f) => &f.id,
        }
    }
    pub fn name(&self) -> &str {
        match self {
            ToolCall::Function(f) => &f.function.name,
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize, Clone, PartialEq)]
pub struct ToolCallFunction {
    /// The ID of the tool call.
    pub id: String,
    /// The function that the model called.
    pub function: ToolCallFunctionObj,
}

#[derive(Debug, Deserialize, Default, Serialize, Clone, PartialEq)]
pub struct ToolCallFunctionObj {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ToolMessage {
    /// The contents of the tool message.
    pub content: String,
    /// Tool call that this message is responding to.
    pub tool_call_id: String,
}

/// Controls which (if any) function is called by the model.
/// `none` means the model will not call a function and instead generates a message.
/// `auto` means the model can pick between generating a message or calling a function.
/// Specifying a particular function via `{"type: "function", "function": {"name": "my_function"}}` forces the model to call that function.

/// `none` is the default when no functions are present. `auto` is the default if functions are present.
#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    #[default]
    None,
    Auto,
    #[serde(untagged)]
    Function(ToolChoiceFunction),
}

/// Specifies a tool the model should use. Use to force the model to call a specific function.
#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct ToolChoiceFunction {
    /// The type of the tool. Currently, only `function` is supported.
    pub r#type: ToolType,
    pub function: FunctionName,
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct FunctionName {
    /// The name of the function to call.
    pub name: String,
}

#[derive(Debug, Deserialize, Default, Serialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        description: Option<String>,
        properties: Option<serde_json::Value>,
        name: String,
        strict: Option<bool>,
    },
}

#[derive(Debug, Deserialize, Default, Serialize, Clone, PartialEq)]
pub struct StreamOptions {
    /// If set, an additional chunk will be streamed before the data: [DONE] message.
    /// The usage field on this chunk shows the token usage statistics for the entire request,
    /// and the choices field will always be an empty array. All other chunks will also include
    /// a usage field, but with a null value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Deserialize, Default, Serialize, Clone, PartialEq)]
pub struct PredictionConfig {
    /// The predicted output text that you expect the model to generate.
    pub text: String,
    /// Optional list of predicted logprobs for each token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize, Default, Serialize, Clone, PartialEq)]
pub struct AudioConfig {
    /// The voice to use for text-to-speech.
    /// Supported voices are: alloy, echo, fable, onyx, nova, and shimmer
    pub voice: String,
    /// The audio output format.
    /// Supported formats are: mp3, opus, aac, and flac
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// The speed of the generated audio.
    /// Select a value from 0.25 to 4.0. 1.0 is the default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

#[derive(Debug, Deserialize, Default, Serialize, Clone, PartialEq)]
pub struct WebSearchOptions {
    /// High level guidance for the amount of context window space to use for the search.
    /// One of `low`, `medium`, or `high`. `medium` is the default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,

    /// Approximate location parameters for the search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<WebSearchUserLocation>,
}

#[derive(Debug, Deserialize, Default, Serialize, Clone, PartialEq)]
pub struct WebSearchUserLocation {
    /// The type of location approximation. Always `approximate`.
    pub r#type: String,

    /// Approximate location parameters for the search.
    pub approximate: WebSearchUserLocationApproximate,
}

#[derive(Debug, Deserialize, Default, Serialize, Clone, PartialEq)]
pub struct WebSearchUserLocationApproximate {
    /// Free text input for the city of the user, e.g. `San Francisco`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,

    /// The two-letter ISO country code of the user, e.g. `US`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,

    /// Free text input for the region of the user, e.g. `California`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,

    /// The IANA timezone of the user, e.g. `America/Los_Angeles`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    High,
    Medium,
    Low,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct OpenRouterReasoning {
    effort: ReasoningEffort,
    exclude: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde() {
        let tests = vec![(
            "default",
            r#"{"model":"gpt-3.5-turbo","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello!"}]}"#,
            RequestBody {
                model: "gpt-3.5-turbo".to_string(),
                messages: vec![
                    Message::System(SystemMessage {
                        content: "You are a helpful assistant.".to_string(),
                        ..Default::default()
                    }),
                    Message::User(UserMessage {
                        content: Content::Text("Hello!".to_string()),
                        name: None,
                    }),
                ],
                ..Default::default()
            },
        ),
        (
            "image input",
            r#"{"model": "gpt-4-vision-preview","messages": [{"role": "user","content": [{"type": "text","text": "What's in this image?"},{"type": "image_url","image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}]}],"max_completion_tokens": 300}"#,
            RequestBody{
                model: "gpt-4-vision-preview".to_string(),
                messages:vec![
                    Message::User(UserMessage{
                        content: Content::Array(vec![
                            ContentPart::Text(TextContentPart{
                                text: "What's in this image?".to_string()
                            }),
                            ContentPart::Image(ImageContentPart{
                                dimensions: None,
                                image_url: ImageUrl{
                                    url: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_string(),
                                    detail: None
                                }
                            })
                        ]),
                        name: None
                    })
                ],
                max_completion_tokens: Some(300),
                ..Default::default()
            }
        ),
        (
            "streaming",
            r#"{"model": "gpt-3.5-turbo","messages": [{"role": "system","content": "You are a helpful assistant."},{"role": "user","content": "Hello!"}],"stream": true}"#,
            RequestBody{
                model: "gpt-3.5-turbo".to_string(),
                messages:vec![
                    Message::System(SystemMessage {
                        content: "You are a helpful assistant.".to_string(),
                        ..Default::default()
                    }),
                    Message::User(UserMessage {
                        content: Content::Text("Hello!".to_string()),
                        name: None,
                    }),
                ],
                stream: Some(true),
                ..Default::default()
            }
        ),
        (
            "functions",
            r#"{"model": "gpt-3.5-turbo","messages": [{"role": "user","content": "What is the weather like in Boston?"}],"tools": [{"type": "function","function": {"name": "get_current_weather","description": "Get the current weather in a given location","parameters": {"type": "object","properties": {"location": {"type": "string","description": "The city and state, e.g. San Francisco, CA"},"unit": {"type": "string","enum": ["celsius", "fahrenheit"]}},"required": ["location"]}}}],"tool_choice": "auto"}"#,
            RequestBody{
                model: "gpt-3.5-turbo".to_string(),
                messages:vec![
                    Message::User(UserMessage {
                        content: Content::Text("What is the weather like in Boston?".to_string()),
                        name: None,
                    }),
                ],
                tools:Some(vec![Tool{
                    r#type:ToolType::Function,
                    function: FunctionTool{
                    name:"get_current_weather".to_string().into(),
                    description: Some("Get the current weather in a given location".to_string().into()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }))
                }
                }]),
                tool_choice: Some(ToolChoice::Auto),
                ..Default::default()
            }
        ),
        (
            "logprobs",
            r#"{"model": "gpt-3.5-turbo","messages": [{"role": "user","content": "Hello!"}],"logprobs": true,"top_logprobs": 2}"#,
            RequestBody{
                model: "gpt-3.5-turbo".to_string(),
                messages:vec![
                    Message::User(UserMessage {
                        content: Content::Text("Hello!".to_string()),
                        name: None,
                    }),
                ],
                logprobs: Some(true),
                top_logprobs: Some(2),
                ..Default::default()
            }
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

    #[test]
    fn test_first_user_message_text() {
        // Test with a simple text content
        let request_body = RequestBody {
            model: "gpt-3.5-turbo".to_string(),
            messages: vec![
                Message::System(SystemMessage {
                    content: "You are a helpful assistant.".to_string(),
                    ..Default::default()
                }),
                Message::User(UserMessage {
                    content: Content::Text("Hello, how are you?".to_string()),
                    name: None,
                }),
            ],
            ..Default::default()
        };
        assert_eq!(
            request_body.first_user_message_text(),
            Some("Hello, how are you?".to_string())
        );

        // Test with array content
        let request_body_with_array = RequestBody {
            model: "gpt-4-vision-preview".to_string(),
            messages: vec![
                Message::User(UserMessage {
                    content: Content::Array(vec![
                        ContentPart::Text(TextContentPart {
                            text: "What's in this image?".to_string(),
                        }),
                        ContentPart::Text(TextContentPart {
                            text: " Please describe it.".to_string(),
                        }),
                        ContentPart::Image(ImageContentPart {
                            dimensions: None,
                            image_url: ImageUrl {
                                url: "https://example.com/image.jpg".to_string(),
                                detail: None,
                            },
                        }),
                    ]),
                    name: None,
                }),
            ],
            ..Default::default()
        };
        assert_eq!(
            request_body_with_array.first_user_message_text(),
            Some("What's in this image? Please describe it.".to_string())
        );

        // Test with no user message
        let request_body_no_user = RequestBody {
            model: "gpt-3.5-turbo".to_string(),
            messages: vec![
                Message::System(SystemMessage {
                    content: "You are a helpful assistant.".to_string(),
                    ..Default::default()
                }),
            ],
            ..Default::default()
        };
        assert_eq!(request_body_no_user.first_user_message_text(), None);
    }
}
