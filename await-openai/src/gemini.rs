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
pub use async_gemini::models::*;

//TODO this are serious problems in gemini function call
impl From<OpenaiRequestBody> for GenerateContentRequest {
    fn from(body: OpenaiRequestBody) -> Self {
        let mut stops = Option::None;
        if let Some(ss) = body.stop {
            match ss {
                Stop::String(s) => stops = Some(vec![s]),
                Stop::Array(a) => stops = Some(a),
            }
        }

        let contents: Vec<Content> = Vec::with_capacity(body.messages.len());

        GenerateContentRequest {
            contents: vec![],
            tools: None,
            safety_settings: None,
            generation_config: Some(GenerateionConfig {
                temperature: body.temperature,
                top_p: body.top_p,
                top_k: None,
                candidate_count: None,
                max_output_tokens: body.max_completion_tokens,
                stop_sequences: stops,
            }),
        }
    }
}
