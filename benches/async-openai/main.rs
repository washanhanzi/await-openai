use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use await_openai::entity::create_chat_completion::{
    Content, Message, RequestBody, SystemMessage, UserMessage,
};
use criterion::{criterion_group, criterion_main, Criterion};
use serde::de::DeserializeOwned;

static DEFAULT: &str = r#"{"model":"gpt-3.5-turbo","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello!"}]}"#;

static IMAGE_INPUT: &str = r#"{"model": "gpt-4-vision-preview","messages": [{"role": "user","content": [{"type": "text","text": "What’s in this image?"},{"type": "image_url","image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}]}],"max_tokens": 300}"#;

fn de_bench<I: DeserializeOwned>(s: &str) {
    let _request: I = serde_json::from_str(s).unwrap();
}

fn de_bench_with_async_openai<I: DeserializeOwned>(s: &str) {
    let _request: I = serde_json::from_str(s).unwrap();
}

fn se_default_request() {
    let request = RequestBody {
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
    };
    let _result = serde_json::to_string(&request).unwrap();
}

fn se_default_request_with_async_openai() {
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-3.5-turbo")
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Who won the world series in 2020?")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("The Los Angeles Dodgers won the World Series in 2020.")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Where was it played?")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();
    let _result = serde_json::to_string(&request).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("deserialize default request", |b| {
        b.iter(|| de_bench::<RequestBody>(DEFAULT))
    });
    c.bench_function("deserialize default request with async-openai", |b| {
        b.iter(|| {
            de_bench_with_async_openai::<async_openai::types::CreateChatCompletionRequest>(DEFAULT)
        })
    });
    c.bench_function("deserialize image input request", |b| {
        b.iter(|| de_bench::<RequestBody>(IMAGE_INPUT))
    });
    //?? async-openai failed on IMAGE_INPUT
    c.bench_function("serialize default request", |b| b.iter(se_default_request));
    c.bench_function("serialize default request with async-openai", |b| {
        b.iter(se_default_request_with_async_openai)
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
