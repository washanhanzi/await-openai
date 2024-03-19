use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;

use crate::{
    entity::{
        chat_completion_chunk::{
            Choice, Chunk, ChunkResponse, DeltaMessage, OpenaiEventDataParser,
        },
        chat_completion_object::{
            Response as OpenaiResponse, Role as OpenaiRole, Usage as OpenaiUsage,
        },
        create_chat_completion::{
            Content, ContentPart, FinishReason, Message as OpenaiMessage,
            RequestBody as OpenaiRequestBody, Stop,
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
            ..Default::default()
        };
        let mut messages = vec![];
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
                                ContentPart::Text(text_part) => blocks.push(ContentBlock::Text {
                                    text: text_part.text,
                                }),
                                ContentPart::Image(image_part) => {
                                    if !image_part.image_url.url.starts_with("http") {
                                        if let Some(mime) =
                                            parse_mime_from_base64(&image_part.image_url.url)
                                        {
                                            blocks.push(ContentBlock::Image {
                                                source: ImageSource::Base64 {
                                                    media_type: mime,
                                                    data: image_part.image_url.url,
                                                },
                                            })
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
        res.system = system_message;
        res.messages = messages;
        res.max_tokens = body.max_tokens.unwrap_or(4000);
        if let Some(stop) = body.stop {
            match stop {
                Stop::String(s) => res.stop_sequences = Some(vec![s]),
                Stop::StringArray(ss) => res.stop_sequences = Some(ss),
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
        }
    }
}

impl EventDataParser<EventData, Chunk, OpenaiResponse> for ClaudeEventDataParser {
    //claude api won't return tool_call infomation for now, parse_data always return none
    fn parse_data(&mut self, data: &EventData) -> Option<Chunk> {
        let data = self.parse_to_openai_event_data(data);
        match data {
            Ok(Some(_)) => None,
            Ok(None) => None,
            Err(_) => None,
        }
    }

    fn get_response(mut self) -> OpenaiResponse {
        self.parser.object = "chat.completion".to_string();
        let mut res = self.parser.get_response();
        res.usage = OpenaiUsage {
            prompt_tokens: self.usage.prompt_tokens,
            completion_tokens: self.usage.completion_tokens,
            total_tokens: self.usage.prompt_tokens + self.usage.completion_tokens,
        };
        res
    }
}

impl ClaudeEventDataParser {
    pub fn get_default_chunk(&self) -> Chunk {
        Chunk::Data(ChunkResponse {
            id: self.parser.id.to_string(),
            choices: vec![],
            created: self.parser.created,
            model: self.parser.model.to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
        })
    }

    pub fn get_chunk_with_choice(
        &self,
        index: usize,
        text: &str,
        role: Option<OpenaiRole>,
        finish_reason: Option<FinishReason>,
    ) -> Chunk {
        Chunk::Data(ChunkResponse {
            id: self.parser.id.to_string(),
            choices: vec![Choice {
                index,
                delta: DeltaMessage {
                    role,
                    content: Some(text.to_string()),
                    ..Default::default()
                },
                finish_reason,
                ..Default::default()
            }],
            created: self.parser.created,
            model: self.parser.model.to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
        })
    }

    pub fn get_event_data_from_str(d: &str) -> Result<EventData, serde_json::Error> {
        serde_json::from_str::<EventData>(d)
    }

    pub fn get_event_data_from_value(d: serde_json::Value) -> Result<EventData, serde_json::Error> {
        serde_json::from_value::<EventData>(d)
    }

    pub fn parse_str(&mut self, d: &str) -> Result<Option<Chunk>> {
        let payload = serde_json::from_str::<EventData>(d)?;
        self.parse_to_openai_event_data(&payload)
    }

    pub fn parse_value(&mut self, d: serde_json::Value) -> Result<Option<Chunk>> {
        let payload = serde_json::from_value::<EventData>(d)?;
        self.parse_to_openai_event_data(&payload)
    }

    pub fn parse_to_openai_event_data(&mut self, data: &EventData) -> Result<Option<Chunk>> {
        match data {
            EventData::Error { error: e } => {
                anyhow::bail!("Error from Claude API: {}", e);
            }
            EventData::MessageStart { message } => {
                self.parser.update_id_if_empty(&message.id);
                self.parser.update_model_if_empty(&message.model);
                self.usage.prompt_tokens = message.usage.input_tokens.unwrap_or_default();
                self.usage.completion_tokens = message.usage.output_tokens;
                Ok(Some(self.get_chunk_with_choice(
                    0,
                    "",
                    Some(OpenaiRole::Assistant),
                    None,
                )))
            }
            EventData::Ping => Ok(None),
            EventData::ContentBlockStart {
                index,
                content_block,
            } => {
                let s = match content_block {
                    ContentBlock::Text { text } => text,
                    _ => "",
                };
                Ok(Some(self.get_chunk_with_choice(
                    *index as usize,
                    s,
                    None,
                    None,
                )))
            }
            EventData::ContentBlockDelta { index, delta } => {
                let s = match delta {
                    ContentBlock::TextDelta { text } => text,
                    _ => "",
                };
                self.parser.push_content(s);
                Ok(Some(self.get_chunk_with_choice(
                    *index as usize,
                    s,
                    None,
                    None,
                )))
            }
            EventData::ContentBlockStop { index: _ } => Ok(None),
            EventData::MessageDelta { delta: _, usage } => {
                self.usage.completion_tokens += usage.output_tokens;
                Ok(None)
            }
            EventData::MessageStop => Ok(Some(Chunk::Done)),
        }
    }
}

impl From<StopReason> for FinishReason {
    fn from(reason: StopReason) -> Self {
        match reason {
            StopReason::EndTurn => FinishReason::Stop,
            StopReason::MaxTokens => FinishReason::Length,
            StopReason::StopSequence => FinishReason::Stop,
        }
    }
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
            create_chat_completion::RequestBody,
        },
        magi::EventDataParser,
    };

    use anyhow::anyhow;
    use async_claude::messages::{
        request::Request, ContentBlock, ImageSource, Message, MessageContent, Role,
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
                    system: Some("You are a helpful assistant.".to_string()),
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
                            ContentBlock::Text {
                                text: "What's in this image?".to_string(),
                            },
                            ContentBlock::Image {
                                source: ImageSource::Base64 {
                                    media_type: "image/png".to_string(),
                                    data: "data:image/png;base64,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".to_string(),
                                },
                            },
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
    fn convert_stream_response() {
        let tests = vec![
            (
                "error",
                r#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#,
                Chunk::Data(ChunkResponse {
                    id: "msg_019LBLYFJ7fG3fuAqzuRQbyi".to_string(),
                    choices: vec![],
                    created: 0,
                    model: "claude-3-opus-20240229".to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                }),
                Some(anyhow!("an error")),
            ),
            (
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_019LBLYFJ7fG3fuAqzuRQbyi","type":"message","role":"assistant","content":[],"model":"claude-3-opus-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}"#,
                Chunk::Data(ChunkResponse {
                    id: "msg_019LBLYFJ7fG3fuAqzuRQbyi".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: Some(OpenaiRole::Assistant),
                            content: Some("".to_string()),
                            ..Default::default()
                        },
                        finish_reason: None,
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-opus-20240229".to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                }),
                None,
            ),
            (
                "content_block_start",
                r#"{"type": "content_block_start", "index":0, "content_block": {"type": "text", "text": ""}}"#,
                Chunk::Data(ChunkResponse {
                    id: "msg_019LBLYFJ7fG3fuAqzuRQbyi".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: Some(OpenaiRole::Assistant),
                            content: Some("".to_string()),
                            ..Default::default()
                        },
                        finish_reason: None,
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-opus-20240229".to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                }),
                None,
            ),
            (
                "ping",
                r#"{"type": "ping"}"#,
                Chunk::Data(ChunkResponse {
                    id: "msg_019LBLYFJ7fG3fuAqzuRQbyi".to_string(),
                    choices: vec![],
                    created: 0,
                    model: "claude-3-opus-20240229".to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                }),
                None,
            ),
            (
                "content_block_delta",
                r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}"#,
                Chunk::Data(ChunkResponse {
                    id: "msg_019LBLYFJ7fG3fuAqzuRQbyi".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: Some(OpenaiRole::Assistant),
                            content: Some("Hello".to_string()),
                            ..Default::default()
                        },
                        finish_reason: None,
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-opus-20240229".to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                }),
                None,
            ),
            (
                "content_block_delta",
                r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}"#,
                Chunk::Data(ChunkResponse {
                    id: "msg_019LBLYFJ7fG3fuAqzuRQbyi".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        delta: DeltaMessage {
                            role: Some(OpenaiRole::Assistant),
                            content: Some("!".to_string()),
                            ..Default::default()
                        },
                        finish_reason: None,
                        ..Default::default()
                    }],
                    created: 0,
                    model: "claude-3-opus-20240229".to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                }),
                None,
            ),
            (
                "content_block_stop",
                r#"{"type": "content_block_stop", "index": 0}"#,
                Chunk::Data(ChunkResponse {
                    id: "msg_019LBLYFJ7fG3fuAqzuRQbyi".to_string(),
                    choices: vec![],
                    created: 0,
                    model: "claude-3-opus-20240229".to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                }),
                None,
            ),
            (
                "message_delta",
                r#"{"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence":null, "usage":{"output_tokens": 15}}}"#,
                Chunk::Data(ChunkResponse {
                    id: "msg_019LBLYFJ7fG3fuAqzuRQbyi".to_string(),
                    choices: vec![],
                    created: 0,
                    model: "claude-3-opus-20240229".to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                }),
                None,
            ),
            (
                "message_stop",
                r#"{"type": "message_stop"}"#,
                Chunk::Done,
                None,
            ),
        ];
        let mut parser = ClaudeEventDataParser::default();
        for (name, input, want, err) in tests {
            let got = parser.parse_str(input);
            if got.is_err() && err.is_none() {
                panic!("unexpected error: {}", got.unwrap_err());
            }
            if got.is_err() && err.is_some() {
                return;
            }
            match (got.unwrap(), want) {
                (Some(Chunk::Data(g)), Chunk::Data(w)) => {
                    assert_eq!(g.id, w.id, "id mismatch: {}", name);
                    assert_eq!(g.choices, w.choices, "choices mismatch: {}", name);
                    assert_eq!(g.model, w.model, "model mismatch: {}", name);
                    assert_eq!(
                        g.system_fingerprint, w.system_fingerprint,
                        "fingerprint mismatch: {}",
                        name
                    );
                    assert_eq!(g.object, w.object, "object mismatch: {}", name);
                }
                (Some(Chunk::Done), Chunk::Done) => {}
                (_, _) => {
                    assert_eq!("test failed: {}", name);
                }
            }

            //parser test
            let event_data = ClaudeEventDataParser::get_event_data_from_str(input);
            assert_eq!(parser.parse_data(event_data.as_ref().unwrap()), None);
        }
        let got_res = parser.get_response();
        let want_res = OpenaiResponse {
            id: "msg_019LBLYFJ7fG3fuAqzuRQbyi".to_string(),
            choices: vec![
                OpenaiResponseChoice {
                    index: 0,
                    message: OpenaiMessage {
                        role: OpenaiRole::Assistant,
                        content: Some("Hello!".to_string()),
                        ..Default::default()
                    },
                    finish_reason: None,
                    ..Default::default()
                },
            ],
            created: 0,
            model: "claude-3-opus-20240229".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 16,
                total_tokens: 26,
            },
        };
        assert_eq!(got_res.id, want_res.id, "id mismatch");
        assert_eq!(got_res.choices, want_res.choices, "choices mismatch");
        assert_eq!(got_res.model, want_res.model, "model mismatch");
        assert_eq!(
            got_res.system_fingerprint, want_res.system_fingerprint,
            "fingerprint mismatch"
        );
        assert_eq!(got_res.object, want_res.object, "object mismatch");
        assert_eq!(got_res.usage, want_res.usage, "usage mismatch");
    }
}
