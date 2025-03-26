use std::{
    str::FromStr,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;

use crate::{
    entity::{
        chat_completion_chunk::{
            Choice, Chunk, ChunkResponse, DeltaMessage, OpenaiEventDataParser, ToolCallChunk,
            ToolCallFunctionObjChunk,
        },
        chat_completion_object::{
            Response as OpenaiResponse, Role as OpenaiRole, Usage as OpenaiUsage,
        },
        create_chat_completion::{
            Content, ContentPart, FinishReason, Message as OpenaiMessage,
            RequestBody as OpenaiRequestBody, Stop, ToolCall, ToolCallFunction,
            ToolCallFunctionObj,
        },
    },
    magi::EventDataParser,
};

pub use async_claude::messages::*;

impl From<OpenaiRequestBody> for Request {
    fn from(body: OpenaiRequestBody) -> Self {
        let mut res = Request {
            model: body.model,
            stream: body.stream,
            temperature: body.temperature,
            top_p: body.top_p,
            max_tokens: body.max_tokens.unwrap_or(4000),
            ..Default::default()
        };
        let mut messages = Vec::with_capacity(body.messages.len());
        let mut system_message = None;
        for message in body.messages {
            match message {
                OpenaiMessage::System(system) => {
                    system_message.replace(system.content);
                }
                OpenaiMessage::User(user) => match user.content {
                    Content::Text(text) => messages.push(Message {
                        role: Role::User,
                        content: MessageContent::Text(text),
                    }),
                    Content::Array(parts) => {
                        let mut blocks = vec![];
                        for p in parts {
                            match p {
                                ContentPart::Text(text_part) => {
                                    blocks.push(ContentBlock::Base(BaseContentBlock::Text {
                                        text: text_part.text,
                                    }))
                                }
                                ContentPart::Image(image_part) => {
                                    if !image_part.image_url.url.starts_with("http") {
                                        if let Some(mime) =
                                            parse_mime_from_base64(&image_part.image_url.url)
                                        {
                                            blocks.push(ContentBlock::RequestOnly(
                                                RequestOnlyContentBlock::Image {
                                                    source: ImageSource::Base64 {
                                                        media_type: mime,
                                                        data: image_part.image_url.url,
                                                    },
                                                },
                                            ))
                                        }
                                    }
                                    tracing::warn!("Image URL is not supported in Claude yet");
                                }
                            }
                        }
                        messages.push(Message {
                            role: Role::User,
                            content: MessageContent::Blocks(blocks),
                        });
                    }
                },
                OpenaiMessage::Assistant(assistant) => {
                    if let Some(text) = assistant.content {
                        messages.push(Message {
                            role: Role::Assistant,
                            content: MessageContent::Text(text),
                        })
                    }
                }
                _ => {}
            }
        }
        res.system = system_message.map(System::Text);
        res.messages = messages;
        if let Some(stop) = body.stop {
            match stop {
                Stop::String(s) => res.stop_sequences = Some(vec![s]),
                Stop::Array(ss) => res.stop_sequences = Some(ss),
            }
        }
        res
    }
}

fn parse_mime_from_base64(s: &str) -> Option<String> {
    let arr: Vec<&str> = s.split(',').collect();
    if arr.len() < 2 {
        return None;
    }
    match arr[0] {
        "data:image/jpeg;base64" => Some("image/jpeg".to_string()),
        "data:image/png;base64" => Some("image/png".to_string()),
        "data:image/gif;base64" => Some("image/gif".to_string()),
        "data:image/webp;base64" => Some("image/webp".to_string()),
        _ => None,
    }
}

/// ClaudeEventDataParser can convert event data from Claude API to Openai API.
/// It stores the intermidiate state of the parsing result and can be used to generate Openai's unary response.
/// It provide two methods to parse the event data, `parse_str` and `parse_value`.
/// If you want to parse from a source I'm not aware of, use `parse_to_openai_event_data` which accepts a reference to `EventData`.
/// The parsed results may return `None`, if you want a 1:1 map of Claude's stream response data, map `None` to `get_default_chunk`.
///
/// # Example
///
/// ````
/// use await_openai::claude::ClaudeEventDataParser;
///
/// let mut parser = ClaudeEventDataParser::default();
/// let data = r#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#;
/// let event_data = parser.parse_str(data);
/// assert_eq!(event_data.unwrap_err().to_string(), "Error from Claude API: OverloadedError: Overloaded");
/// ````
#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeEventDataParser {
    usage: OpenaiUsage,
    parser: OpenaiEventDataParser,
    stop_reason: Option<StopReason>,
    stop_sequence: Option<String>,
    tool_call: Option<ToolCall>,
    claude_tool_calls: Vec<ToolUseContentBlock>,
    signature: Option<String>,
}

impl Default for ClaudeEventDataParser {
    fn default() -> Self {
        let created_at = {
            match SystemTime::now().duration_since(UNIX_EPOCH) {
                Ok(n) => n.as_secs(),
                Err(_) => 0,
            }
        };
        let mut parser = OpenaiEventDataParser::default();
        parser.created = created_at;
        Self {
            usage: OpenaiUsage::default(),
            parser: OpenaiEventDataParser::default(),
            stop_reason: None,
            stop_sequence: None,
            tool_call: None,
            claude_tool_calls: vec![],
            signature: None,
        }
    }
}

impl EventDataParser<EventData> for ClaudeEventDataParser {
    type Error = anyhow::Error;
    type Output = (Option<Chunk>, Option<ToolCall>);
    type UnarayResponse = OpenaiResponse;

    fn parse(
        &mut self,
        data: &EventData,
    ) -> Result<(Option<Chunk>, Option<ToolCall>), anyhow::Error> {
        match data {
            EventData::Error { error } => {
                anyhow::bail!("Error from Claude API: {}", error);
            }
            EventData::MessageStart { message } => {
                self.parser.update_id_if_empty(&message.id);
                self.parser.update_model_if_empty(&message.model);
                self.usage.prompt_tokens = message.usage.input_tokens.unwrap_or_default();
                self.usage.completion_tokens = message.usage.output_tokens;
                Ok((
                    Some(self.chunk_with_choice(0, None, None, Some(OpenaiRole::Assistant), None)),
                    None,
                ))
            }
            EventData::ContentBlockStart {
                index: _,
                content_block,
            } => match content_block {
                BaseContentBlock::ToolUse(tool_use) => {
                    self.tool_call = Some(ToolCall::Function(ToolCallFunction {
                        id: tool_use.id.to_string(),
                        function: ToolCallFunctionObj {
                            name: tool_use.name.to_string(),
                            arguments: String::new(),
                        },
                    }));
                    Ok((None, None))
                }
                BaseContentBlock::Text { text: _ } => Ok((None, None)),
                BaseContentBlock::Thinking {
                    thinking: _,
                    signature: _,
                } => Ok((None, None)),
            },
            EventData::Ping => Ok((None, None)),
            EventData::ContentBlockDelta { index, delta } => match delta {
                DeltaContentBlock::TextDelta { text } => {
                    self.parser.push_content(text);
                    Ok((
                        Some(self.chunk_with_choice(*index as usize, Some(text), None, None, None)),
                        None,
                    ))
                }
                DeltaContentBlock::InputJsonDelta { partial_json } => {
                    let prev_tool_call = self.tool_call.take();
                    if let Some(ToolCall::Function(function)) = prev_tool_call {
                        self.tool_call = Some(ToolCall::Function(ToolCallFunction {
                            id: function.id,
                            function: ToolCallFunctionObj {
                                name: function.function.name,
                                arguments: function.function.arguments + partial_json,
                            },
                        }));
                    }
                    Ok((None, None))
                }
                DeltaContentBlock::ThinkingDelta { thinking } => {
                    self.parser.push_thinking(thinking);
                    Ok((
                        Some(self.chunk_with_choice(
                            *index as usize,
                            None,
                            Some(thinking),
                            None,
                            None,
                        )),
                        None,
                    ))
                }
                DeltaContentBlock::SignatureDelta { signature } => {
                    self.signature = Some(signature.to_string());
                    Ok((None, None))
                }
            },
            EventData::ContentBlockStop { index: _ } => {
                if let Some(ToolCall::Function(function)) = self.tool_call.take() {
                    let tool_call = ToolCall::Function(ToolCallFunction {
                        id: function.id.to_string(),
                        function: function.function.clone(),
                    });
                    self.parser.push_tool_call(tool_call.clone());
                    if let Ok(obj) =
                        serde_json::from_str::<serde_json::Value>(&function.function.arguments)
                    {
                        self.claude_tool_calls.push(ToolUseContentBlock {
                            id: function.id,
                            name: function.function.name,
                            input: obj,
                        });
                    }
                    return Ok((None, Some(tool_call)));
                }
                Ok((None, None))
            }
            EventData::MessageDelta { delta, usage } => {
                self.usage.completion_tokens += usage.output_tokens;
                self.parser
                    .set_finish_reason(Some(delta.stop_reason.clone().into()));
                self.stop_reason = Some(delta.stop_reason.clone());
                Ok((
                    Some(self.chunk_with_choice(
                        0,
                        None,
                        None,
                        None,
                        Some(delta.stop_reason.clone().into()),
                    )),
                    None,
                ))
            }
            EventData::MessageStop => Ok((Some(Chunk::Done), None)),
        }
    }

    fn response(mut self) -> OpenaiResponse {
        self.parser.object = "chat.completion".to_string();
        let mut res = self.parser.response();
        res.usage = OpenaiUsage {
            prompt_tokens: self.usage.prompt_tokens,
            completion_tokens: self.usage.completion_tokens,
            total_tokens: self.usage.prompt_tokens + self.usage.completion_tokens,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };
        res
    }
}

impl ClaudeEventDataParser {
    pub fn claude_response(&self) -> async_claude::messages::Response {
        let mut content = vec![];
        if !self.parser.think_content.is_empty() {
            content.push(ResponseContentBlock::Base(BaseContentBlock::Thinking {
                thinking: self.parser.think_content.clone(),
                signature: self.signature.clone(),
            }));
        }
        if !self.parser.content.is_empty() {
            content.push(ResponseContentBlock::Base(BaseContentBlock::Text {
                text: self.parser.content.clone(),
            }));
        }
        for tool_call in self.claude_tool_calls.iter() {
            content.push(ResponseContentBlock::Base(BaseContentBlock::ToolUse(
                ToolUseContentBlock {
                    id: tool_call.id.to_string(),
                    name: tool_call.name.to_string(),
                    input: tool_call.input.clone(),
                },
            )));
        }

        async_claude::messages::Response {
            id: self.parser.id.to_string(),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content,
            model: self.parser.model.to_string(),
            stop_reason: self.stop_reason.clone(),
            stop_sequence: None,
            usage: Usage {
                input_tokens: Some(self.usage.prompt_tokens),
                output_tokens: self.usage.completion_tokens,
            },
        }
    }
    pub fn default_chunk(&self) -> Chunk {
        Chunk::Data(ChunkResponse {
            id: self.parser.id.to_string(),
            choices: vec![],
            created: self.parser.created,
            model: self.parser.model.to_string(),
            system_fingerprint: String::new(),
            service_tier: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        })
    }

    pub fn chunk_with_choice(
        &self,
        index: usize,
        text: Option<&str>,
        reasoning: Option<&str>,
        role: Option<OpenaiRole>,
        finish_reason: Option<FinishReason>,
    ) -> Chunk {
        Chunk::Data(ChunkResponse {
            id: self.parser.id.to_string(),
            choices: vec![Choice {
                index,
                delta: DeltaMessage {
                    role,
                    content: text.map(|s| s.to_string()),
                    reasoning: reasoning.map(|s| s.to_string()),
                    ..Default::default()
                },
                finish_reason,
                ..Default::default()
            }],
            created: self.parser.created,
            model: self.parser.model.to_string(),
            system_fingerprint: String::new(),
            service_tier: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        })
    }

    pub fn parse_str(&mut self, d: &str) -> Result<(Option<Chunk>, Option<ToolCall>)> {
        let payload = serde_json::from_str::<EventData>(d)?;
        self.parse(&payload)
    }

    pub fn parse_value(
        &mut self,
        d: serde_json::Value,
    ) -> Result<(Option<Chunk>, Option<ToolCall>)> {
        let payload = serde_json::from_value::<EventData>(d)?;
        self.parse(&payload)
    }
}

impl From<StopReason> for FinishReason {
    fn from(reason: StopReason) -> Self {
        match reason {
            StopReason::EndTurn => FinishReason::Stop,
            StopReason::MaxTokens => FinishReason::Length,
            StopReason::StopSequence => FinishReason::Stop,
            StopReason::ToolUse => FinishReason::ToolCalls,
        }
    }
}

#[cfg(feature = "claude-price")]
pub fn price(model: &str, usage: &OpenaiUsage) -> f32 {
    let claude_usage = Usage {
        input_tokens: Some(usage.prompt_tokens),
        output_tokens: usage.completion_tokens,
    };
    async_claude::price(model, &claude_usage)
}

#[cfg(test)]
mod tests {
    use crate::{
        entity::{
            chat_completion_chunk::{Choice, Chunk, ChunkResponse, DeltaMessage},
            chat_completion_object::{
                Choice as OpenaiResponseChoice, Message as OpenaiMessage,
                Response as OpenaiResponse, Role as OpenaiRole, Usage,
            },
            create_chat_completion::{
                FinishReason, RequestBody, ToolCall, ToolCallFunction, ToolCallFunctionObj,
            },
        },
        magi::EventDataParser,
    };

    use anyhow::anyhow;
    use async_claude::messages::{
        BaseContentBlock, ContentBlock, ImageSource, Message, MessageContent,
        RequestOnlyContentBlock, Role, StopReason, System, request::Request,
    };

    use super::ClaudeEventDataParser;

    #[test]
    fn convert_request() {
        let tests = vec![
            (
                "default",
                r#"{"model":"gpt-3.5-turbo","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello!"}]}"#,
                Request {
                    model: "gpt-3.5-turbo".to_string(),
                    system: Some(System::Text("You are a helpful assistant.".to_string())),
                    messages: vec![Message {
                        role: Role::User,
                        content: MessageContent::Text("Hello!".to_string()),
                    }],
                    max_tokens: 4000,
                    ..Default::default()
                },
            ),
            (
                "image",
                r#"{"model": "gpt-4-vision-preview","messages": [{"role": "user","content": [{"type": "text","text": "What's in this image?"},{"type": "image_url","image_url": {"url": "data:image/png;base64,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"}}]}],"max_tokens": 300}"#,
                Request {
                    model: "gpt-4-vision-preview".to_string(),
                    messages: vec![Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Base(BaseContentBlock::Text {
                                text: "What's in this image?".to_string(),
                            }),
                            ContentBlock::RequestOnly(RequestOnlyContentBlock::Image {
                                source: ImageSource::Base64 {
                                    media_type: "image/png".to_string(),
                                    data: "data:image/png;base64,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".to_string(),
                                },
                            }),
                        ]),
                    }],
                    max_tokens: 300,
                    ..Default::default()
                },
            ),
        ];
        for (name, json, want) in tests {
            //test deserialize
            let parsed: RequestBody = serde_json::from_str(json).unwrap();
            let got: Request = parsed.into();
            assert_eq!(got, want, "deserialize test failed: {}", name);
        }
    }

    #[test]
    fn test_process_stream_events() {
        let events = vec![
            (
                "data1",
                r#"{"type": "message_start", "message": {"id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY", "type": "message", "role": "assistant", "content": [], "model": "claude-3-7-sonnet-20250219", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 1}}}"#,
                Some(Chunk::Data(ChunkResponse {
                    id: "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: Some(OpenaiRole::Assistant),
                            content: None,
                            tool_calls: None,
                            ..Default::default()
                        },
                        finish_reason: None,
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-7-sonnet-20250219".to_string(),
                    system_fingerprint: "".to_string(),
                    service_tier: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                })),
            ),
            (
                "data2",
                r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}"#,
                None,
            ),
            ("data3", r#"{"type": "ping"}"#, None),
            (
                "data4",
                r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}"#,
                Some(Chunk::Data(ChunkResponse {
                    id: "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: Some("Hello".to_string()),
                            tool_calls: None,
                            ..Default::default()
                        },
                        finish_reason: None,
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-7-sonnet-20250219".to_string(),
                    system_fingerprint: "".to_string(),
                    service_tier: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                })),
            ),
            (
                "data5",
                r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}"#,
                Some(Chunk::Data(ChunkResponse {
                    id: "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: Some("!".to_string()),
                            tool_calls: None,
                            ..Default::default()
                        },
                        finish_reason: None,
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-7-sonnet-20250219".to_string(),
                    system_fingerprint: "".to_string(),
                    service_tier: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                })),
            ),
            (
                "data6",
                r#"{"type": "content_block_stop", "index": 0}"#,
                None,
            ),
            (
                "data7",
                r#"{"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence":null}, "usage": {"output_tokens": 15}}"#,
                Some(Chunk::Data(ChunkResponse {
                    id: "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: None,
                            tool_calls: None,
                            ..Default::default()
                        },
                        finish_reason: Some(FinishReason::Stop),
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-7-sonnet-20250219".to_string(),
                    system_fingerprint: "".to_string(),
                    service_tier: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                })),
            ),
            ("data8", r#"{"type": "message_stop"}"#, Some(Chunk::Done)),
        ];

        let mut parser = ClaudeEventDataParser::default();

        for (name, event_str, want) in events {
            match parser.parse_str(event_str) {
                Ok((got, tool_call)) => {
                    assert_eq!(got, want, "openai event data not match: {}", name);
                    assert_eq!(tool_call, None, "tool call should be None")
                }
                Err(e) => panic!("Error parsing event: {}", e),
            }
        }

        //claude unaray response
        let claude_response = parser.claude_response();
        let want_claude_response = async_claude::messages::Response {
            id: "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY".to_string(),
            r#type: "message".to_string(),
            role: async_claude::messages::Role::Assistant,
            content: vec![
                async_claude::messages::ResponseContentBlock::Base(
                    async_claude::messages::BaseContentBlock::Text {
                        text: "Hello!".to_string(),
                    },
                ),
            ],
            model: "claude-3-7-sonnet-20250219".to_string(),
            stop_reason: None,
            stop_sequence: None,
            usage: async_claude::messages::Usage {
                input_tokens: Some(25),
                output_tokens: 16,
            },
        };
        assert_eq!(
            claude_response, want_claude_response,
            "Claude unary response doesn't match expected value"
        );

        // Store the created timestamp from the parser before consuming it
        let created_timestamp = parser.parser.created;

        //openai unary response
        let openai_response = parser.response();
        let want_openai_response = crate::entity::chat_completion_object::Response {
            id: "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY".to_string(),
            object: "chat.completion".to_string(),
            created: created_timestamp,
            model: "claude-3-7-sonnet-20250219".to_string(),
            system_fingerprint: String::new(),
            choices: vec![
                crate::entity::chat_completion_object::Choice {
                    index: 0,
                    message: crate::entity::chat_completion_object::Message {
                        role: crate::entity::chat_completion_object::Role::Assistant,
                        content: Some("Hello!".to_string()),
                        reasoning: None,
                        tool_calls: None,
                        refusal: None,
                        annotations: None,
                        audio: None,
                    },
                    finish_reason: Some(crate::entity::create_chat_completion::FinishReason::Stop),
                    logprobs: None,
                },
            ],
            usage: crate::entity::chat_completion_object::Usage {
                prompt_tokens: 25,
                completion_tokens: 16,
                total_tokens: 41,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            },
            service_tier: None,
        };
        assert_eq!(
            openai_response, want_openai_response,
            "OpenAI unary response doesn't match expected value"
        );
    }

    #[test]
    fn test_process_tool_use_events() {
        // Define events and expected outcomes
        let test_events = vec![
            (
                "data1",
                r#"{"type":"message_start","message":{"id":"msg_014p7gG3wDgGV9EUtLvnow3U","type":"message","role":"assistant","model":"claude-3-haiku-20240307","stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2},"content":[],"stop_reason":null}}"#,
                Some(Chunk::Data(ChunkResponse {
                    id: "msg_014p7gG3wDgGV9EUtLvnow3U".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: Some(OpenaiRole::Assistant),
                            content: None,
                            tool_calls: None,
                            ..Default::default()
                        },
                        finish_reason: None,
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-haiku-20240307".to_string(),
                    system_fingerprint: "".to_string(),
                    service_tier: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                })),
                None,
            ),
            (
                "data2",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                None,
                None,
            ),
            ("data3", r#"{"type":"ping"}"#, None, None),
            (
                "data4",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Okay"}}"#,
                Some(Chunk::Data(ChunkResponse {
                    id: "msg_014p7gG3wDgGV9EUtLvnow3U".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: Some("Okay".to_string()),
                            tool_calls: None,
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-haiku-20240307".to_string(),
                    system_fingerprint: "".to_string(),
                    service_tier: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                })),
                None,
            ),
            (
                "data5",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":","}}"#,
                Some(Chunk::Data(ChunkResponse {
                    id: "msg_014p7gG3wDgGV9EUtLvnow3U".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: Some(",".to_string()),
                            tool_calls: None,
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-haiku-20240307".to_string(),
                    system_fingerprint: "".to_string(),
                    service_tier: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                })),
                None,
            ),
            (
                "data6",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" let"}}"#,
                Some(Chunk::Data(ChunkResponse {
                    id: "msg_014p7gG3wDgGV9EUtLvnow3U".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: Some(" let".to_string()),
                            tool_calls: None,
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-haiku-20240307".to_string(),
                    system_fingerprint: "".to_string(),
                    service_tier: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                })),
                None,
            ),
            (
                "data7",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'s check the weather for San Francisco, CA:"}}"#,
                Some(Chunk::Data(ChunkResponse {
                    id: "msg_014p7gG3wDgGV9EUtLvnow3U".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: Some(
                                "'s check the weather for San Francisco, CA:".to_string(),
                            ),
                            tool_calls: None,
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-haiku-20240307".to_string(),
                    system_fingerprint: "".to_string(),
                    service_tier: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                })),
                None,
            ),
            (
                "data9",
                r#"{"type":"content_block_stop","index":0}"#,
                None,
                None,
            ),
            (
                "data10",
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}}"#,
                None,
                None,
            ),
            (
                "data11",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}"#,
                None,
                None,
            ),
            (
                "data12",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}"#,
                None,
                None,
            ),
            (
                "data13",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \"San Francisco, CA\", \"unit\": \"fahrenheit\"}"}}"#,
                None,
                None,
            ),
            (
                "data14",
                r#"{"type":"content_block_stop","index":1}"#,
                None,
                Some(ToolCall::Function(ToolCallFunction {
                    id: "toolu_01T1x1fJ34qAmk2tNTrN7Up6".to_string(),
                    function: ToolCallFunctionObj {
                        name: "get_weather".to_string(),
                        arguments:
                            "{\"location\": \"San Francisco, CA\", \"unit\": \"fahrenheit\"}"
                                .to_string(),
                    },
                })),
            ),
            (
                "data15",
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}"#,
                Some(Chunk::Data(ChunkResponse {
                    id: "msg_014p7gG3wDgGV9EUtLvnow3U".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: None,
                            content: None,
                            tool_calls: None,
                            ..Default::default()
                        },
                        finish_reason: Some(FinishReason::ToolCalls),
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-haiku-20240307".to_string(),
                    system_fingerprint: "".to_string(),
                    service_tier: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                })),
                None,
            ),
            (
                "data16",
                r#"{"type":"message_stop"}"#,
                Some(Chunk::Done),
                None,
            ),
        ];

        let mut parser = ClaudeEventDataParser::default();
        for (name, event_str, want, got_tool_call) in test_events {
            match parser.parse_str(event_str) {
                Ok((got, want_tool_call)) => {
                    assert_eq!(got, want, "openai event data not match: {}", name);
                    assert_eq!(
                        got_tool_call, want_tool_call,
                        "openai event tool call not match: {}",
                        name
                    );
                }
                Err(e) => panic!("Error parsing event: {}", e),
            }
        }

        // Claude unary response
        let claude_response = parser.claude_response();
        let want_claude_response = async_claude::messages::Response {
            id: "msg_014p7gG3wDgGV9EUtLvnow3U".to_string(),
            r#type: "message".to_string(),
            role: async_claude::messages::Role::Assistant,
            content: vec![
                async_claude::messages::ResponseContentBlock::Base(
                    async_claude::messages::BaseContentBlock::Text {
                        text: "Okay, let's check the weather for San Francisco, CA:".to_string(),
                    },
                ),
                async_claude::messages::ResponseContentBlock::Base(
                    async_claude::messages::BaseContentBlock::ToolUse(
                        async_claude::messages::ToolUseContentBlock {
                            id: "toolu_01T1x1fJ34qAmk2tNTrN7Up6".to_string(),
                            name: "get_weather".to_string(),
                            input: serde_json::from_str(
                                r#"{"location": "San Francisco, CA", "unit": "fahrenheit"}"#,
                            )
                            .unwrap(),
                        },
                    ),
                ),
            ],
            model: "claude-3-haiku-20240307".to_string(),
            stop_reason: Some(StopReason::ToolUse),
            stop_sequence: None,
            usage: async_claude::messages::Usage {
                input_tokens: Some(472),
                output_tokens: 91,
            },
        };
        assert_eq!(
            claude_response, want_claude_response,
            "Claude unary response doesn't match expected value"
        );

        // Store the created timestamp from the parser before consuming it
        let created_timestamp = parser.parser.created;

        // OpenAI unary response
        let openai_response = parser.response();
        let want_openai_response = crate::entity::chat_completion_object::Response {
            id: "msg_014p7gG3wDgGV9EUtLvnow3U".to_string(),
            object: "chat.completion".to_string(),
            created: created_timestamp,
            model: "claude-3-haiku-20240307".to_string(),
            system_fingerprint: String::new(),
            choices: vec![
                crate::entity::chat_completion_object::Choice {
                    index: 0,
                    message: crate::entity::chat_completion_object::Message {
                        role: crate::entity::chat_completion_object::Role::Assistant,
                        content: Some("Okay, let's check the weather for San Francisco, CA:".to_string()),
                        reasoning: None,
                        tool_calls: Some(vec![
                            crate::entity::create_chat_completion::ToolCall::Function(
                                crate::entity::create_chat_completion::ToolCallFunction {
                                    id: "toolu_01T1x1fJ34qAmk2tNTrN7Up6".to_string(),
                                    function: crate::entity::create_chat_completion::ToolCallFunctionObj {
                                        name: "get_weather".to_string(),
                                        arguments: r#"{"location": "San Francisco, CA", "unit": "fahrenheit"}"#.to_string(),
                                    },
                                },
                            ),
                        ]),
                        refusal: None,
                        annotations: None,
                        audio: None,
                    },
                    finish_reason: Some(crate::entity::create_chat_completion::FinishReason::ToolCalls),
                    logprobs: None,
                },
            ],
            usage: crate::entity::chat_completion_object::Usage {
                prompt_tokens: 472,
                completion_tokens: 91,
                total_tokens: 563,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            },
            service_tier: None,
        };
        assert_eq!(
            openai_response, want_openai_response,
            "OpenAI unary response doesn't match expected value"
        );
    }
}
