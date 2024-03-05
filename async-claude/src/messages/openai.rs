use std::time::{SystemTime, UNIX_EPOCH};

use await_openai::entity::{
    chat_completion_chunk::{Choice, Chunk, ChunkResponse, DeltaMessage},
    chat_completion_object::Role as OpenaiRole,
    create_chat_completion::{Content, ContentPart, FinishReason, Message as OpenaiMessage, Stop},
};

use super::{
    request::Request, stream_response::EventData, ContentBlock, ImageSource, Message,
    MessageContent, Role, StopReason,
};

impl From<await_openai::entity::create_chat_completion::RequestBody> for Request {
    fn from(body: await_openai::entity::create_chat_completion::RequestBody) -> Self {
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

impl From<super::stream_response::EventData>
    for Option<await_openai::entity::chat_completion_chunk::Chunk>
{
    fn from(value: super::stream_response::EventData) -> Self {
        //TODO option?
        //TODO u32 or usize?
        //TODO how to get usage out of claude
        let mut id = String::new();
        let mut model = String::new();
        let created_at = {
            match SystemTime::now().duration_since(UNIX_EPOCH) {
                Ok(n) => n.as_secs(),
                Err(_) => 0,
            }
        };
        match value {
            EventData::Error(e) => {
                tracing::error!("Error from Claude: {}", e);
                Some(Chunk::Done)
            }
            EventData::MessageStart { message } => {
                id = message.id;
                model = message.model;
                Some(Chunk::Data(ChunkResponse {
                    id,
                    choices: get_choices(0, "", None),
                    created: created_at,
                    model,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                }))
            }
            EventData::Ping => None,
            EventData::ContentBlockStart {
                index,
                content_block,
            } => {
                let s = match content_block {
                    ContentBlock::Text { text } => text,
                    _ => "".to_string(),
                };
                Some(Chunk::Data(ChunkResponse {
                    id,
                    choices: get_choices(index as usize, &s, None),
                    created: created_at,
                    model,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                }))
            }
            EventData::ContentBlockDelta { index, delta } => {
                let s = match delta {
                    ContentBlock::TextDelta { text } => text,
                    _ => "".to_string(),
                };
                Some(Chunk::Data(ChunkResponse {
                    id,
                    choices: get_choices(index as usize, &s, None),
                    created: created_at,
                    model,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                }))
            }
            EventData::ContentBlockStop { index: _ } => None,
            EventData::MessageDelta { delta, usage: _ } => Some(Chunk::Data(ChunkResponse {
                id,
                choices: get_choices(0, "", Some(delta.stop_reason.into())),
                created: created_at,
                model,
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
            })),
            EventData::MessageStop => Some(Chunk::Done),
        }
    }
}

fn get_choices(index: usize, text: &str, finish_reason: Option<FinishReason>) -> Vec<Choice> {
    vec![Choice {
        index,
        delta: DeltaMessage {
            role: Some(OpenaiRole::Assistant),
            content: Some(text.to_string()),
            ..Default::default()
        },
        finish_reason,
        ..Default::default()
    }]
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
