use std::{
    collections::VecDeque,
    sync::{Arc, OnceLock, RwLock},
};

use crate::entity::{
    chat_completion_object::{Choice, Response, Usage},
    create_chat_completion::{
        Content, ContentPart, Message, RequestBody, Tool, ToolCall, ToolType,
    },
};
use anyhow::{anyhow, Result};
use tiktoken_rs::{
    cl100k_base, get_bpe_from_tokenizer,
    tokenizer::{get_tokenizer, Tokenizer},
    CoreBPE,
};

mod image_token;
pub use image_token::get_image_tokens;

pub trait TokenCounter {
    fn count(&self, content: &str) -> usize;
}

pub struct BpeTokenCounter {
    bpe: Arc<RwLock<CoreBPE>>,
}

static CL100K_BASE_TOKENIZER: OnceLock<Arc<RwLock<CoreBPE>>> = OnceLock::new();

pub fn cl100k_base_tokenizer() -> Arc<RwLock<CoreBPE>> {
    CL100K_BASE_TOKENIZER
        .get_or_init(|| Arc::new(RwLock::new(cl100k_base().unwrap())))
        .clone()
}

impl BpeTokenCounter {
    pub fn new(_model: &str) -> Self {
        let bpe = cl100k_base_tokenizer();
        BpeTokenCounter { bpe }
    }
}

impl TokenCounter for BpeTokenCounter {
    fn count(&self, content: &str) -> usize {
        let bpe = self.bpe.read().unwrap();
        bpe.encode_with_special_tokens(content).len()
    }
}

pub struct OpenaiTokens {
    tokens_per_name: i32,
    tokens_per_message: i32,
}

impl OpenaiTokens {
    pub fn new(tokens_per_message: Option<i32>, tokens_per_name: Option<i32>) -> Self {
        // now gpt-3.5-turbo and gpt-4-turbo both use 4 as token per message
        // it seems the following logic is deprecated for now
        // let (tokens_per_message, tokens_per_name) = if model.starts_with("gpt-3.5") {
        //     (
        //         4,  // every message follows <im_start>{role/name}\n{content}<im_end>\n
        //         -1, // if there's a name, the role is omitted
        //     )
        // } else {
        //     (3, 1)
        // };
        let (tokens_per_message, tokens_per_name) = (
            tokens_per_message.unwrap_or(4), // every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name.unwrap_or(-1),   // if there's a name, the role is omitted
        );
        OpenaiTokens {
            // req_contents: VecDeque::new(),
            tokens_per_name,
            tokens_per_message,
        }
    }

    // pub fn push(&mut self, content: &str) {
    //     if content.is_empty() {
    //         return;
    //     }
    //     self.req_contents.push_back(content.to_string());
    // }

    fn parse_prompt_message<'a>(
        &self,
        contents: &mut VecDeque<&'a str>,
        message: &'a Message,
    ) -> i32 {
        let mut temp_counter = 0;
        match message {
            Message::System(m) => {
                if let Some(name) = m.name.as_deref() {
                    temp_counter += self.tokens_per_name;
                    contents.push_back(name);
                }
                contents.push_back(&m.content);
            }
            Message::User(m) => {
                if let Some(name) = m.name.as_deref() {
                    temp_counter += self.tokens_per_name;
                    contents.push_back(name);
                }
                match &m.content {
                    Content::Text(text) => {
                        contents.push_back(text);
                    }
                    Content::Array(array) => {
                        for part in array {
                            match part {
                                ContentPart::Text(t) => {
                                    contents.push_back(&t.text);
                                }
                                ContentPart::Image(image) => {
                                    if let Some((w, h)) = image.dimensions {
                                        temp_counter +=
                                            get_image_tokens((w, h), &image.image_url.detail)
                                                as i32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Message::Assistant(m) => {
                if let Some(name) = m.name.as_deref() {
                    temp_counter += self.tokens_per_name;
                    contents.push_back(name);
                }
                if let Some(content) = m.content.as_deref() {
                    contents.push_back(content);
                }
                if let Some(tools) = &m.tool_calls {
                    for tool in tools {
                        match tool {
                            ToolCall::Function(function_call) => {
                                contents.push_back(&function_call.function.name);
                                contents.push_back(&function_call.function.arguments);
                            }
                        }
                    }
                }
            }
            Message::Tool(m) => {
                contents.push_back(&m.content);
            }
        }
        temp_counter
    }

    pub fn request_count(
        &mut self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        counter: &impl TokenCounter,
    ) -> usize {
        let mut req_contents: VecDeque<&str> = VecDeque::new();
        let mut tool_msgs = String::new();
        let mut num_tokens: i32 = 0;
        for message in messages {
            num_tokens += self.tokens_per_message;
            let temp = self.parse_prompt_message(&mut req_contents, message);
            num_tokens += temp;
        }
        // every reply is primed with <|start|>assistant<|message|>
        num_tokens += 3;
        //calculate tools tokens
        if let Some(tools) = tools {
            for tool in tools {
                match tool.r#type {
                    ToolType::Function => {
                        if let Some(desc) = tool.function.description.as_deref() {
                            req_contents.push_back("// ");
                            req_contents.push_back(desc);
                            req_contents.push_back("\n");
                        }
                        req_contents.push_back("namespace functions\n type ");
                        req_contents.push_back(&tool.function.name);
                        req_contents.push_back("=>\n");
                        // tool.function.parameters is a serde_json::Value
                        if let Some(parameters_json) = tool.function.parameters.as_ref() {
                            tool_msgs.push_str(&parameters_json.to_string());
                        }
                    }
                }
                tool_msgs.push('\n');
            }
        }
        let mut num_tokens: usize = {
            if num_tokens < 0 {
                0
            } else {
                num_tokens as usize
            }
        };
        let concat_contents = req_contents.drain(..).collect::<Vec<&str>>().join(" ");
        num_tokens += counter.count(&concat_contents);
        num_tokens += counter.count(&tool_msgs);
        num_tokens
    }

    pub fn response_count(&mut self, choices: &[Choice], counter: &impl TokenCounter) -> usize {
        let mut content = String::new();
        for choice in choices {
            if let Some(c) = choice.message.content.as_deref() {
                content.push_str(c);
            }
            if let Some(tools) = choice.message.tool_calls.as_deref() {
                for tool in tools {
                    // tool.function.parameters is a serde_json::Value
                    match tool {
                        ToolCall::Function(function_call) => {
                            content.push_str("namespace functions\n type ");
                            content.push_str(&function_call.function.name);
                            content.push_str("=>\n");
                            content.push_str(&function_call.function.arguments);
                        }
                    };
                    content.push('\n');
                }
            }
        }
        counter.count(&content)
    }
}

/// prompt_tokens calculates the token usage for prompts.
/// This function provides an estimated count when the prompt includes [`Tool`] and [`AssistantMessage`]'s [`ToolCall`].
/// It's important to note that an exact methodology for calculating token usage for [`Tool`] and [`ToolCall`] has not been disclosed by OpenAI.
/// For context, see [generating TypeScript definitions](https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/6) as an example.
/// Given the absence of an official method and unreliable results of the above method got from other than get_current_weather example,
/// this implementation adopts a more efficient approach to estimate token usage, potentially sacrificing some degree of accuracy for improved performance and simplicity.
/// You can check the test cases for the estimated and actual token usage. run `cargo test --features tiktoken`.
///
/// [`AssistantMessage`]: crate::entity::create_chat_completion::AssistantMessage
pub fn prompt_tokens(model: &str, messages: &[Message], tools: Option<&[Tool]>) -> usize {
    let counter = BpeTokenCounter::new(model);
    let mut openai_tokens = OpenaiTokens::new(None, None);
    openai_tokens.request_count(messages, tools, &counter)
}

/// completion_tokens calculates the token usage for completion object.
/// The result is an estimation when response includes [`ToolCall`].
pub fn completion_tokens(model: &str, choices: &[Choice]) -> usize {
    let counter = BpeTokenCounter::new(model);
    let mut openai_tokens = OpenaiTokens::new(None, None);
    openai_tokens.response_count(choices, &counter)
}

pub fn usage(req: &RequestBody, res: &Response) -> Usage {
    let counter = BpeTokenCounter::new(&res.model);
    let mut openai_tokens = OpenaiTokens::new(None, None);
    let prompt_tokens = openai_tokens.request_count(&req.messages, req.tools.as_deref(), &counter);
    let completion_tokens = openai_tokens.response_count(&res.choices, &counter);
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::{
        chat_completion_object::Message as ResponseMsg,
        create_chat_completion::{
            Content, FinishReason, FunctionTool, Message, ToolCallFunction, ToolCallFunctionObj,
        },
    };

    #[test]
    fn messages() {
        let messages = vec![
            Message::System(crate::entity::create_chat_completion::SystemMessage {
                name: None,
                content: "You are a helpful assistant.".to_string(),
            }),
            Message::User(crate::entity::create_chat_completion::UserMessage {
                name: None,
                content: Content::Text("hi, how are you".to_string()),
            }),
        ];
        let num_tokens = prompt_tokens("gpt-3.5-turbo", &messages, None).unwrap();
        assert_eq!(num_tokens, 22);
    }

    #[test]
    fn completion() {
        let choices = vec![Choice {
            index: 0,
            message: ResponseMsg {
                content: Some("I'm just a computer program, so I don't have feelings, but I'm here to help you with anything you need. How can I assist you today?".to_string()),
                tool_calls: None,
                role: crate::entity::chat_completion_object::Role::Assistant,
            },
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        }];
        let num_tokens = completion_tokens("gpt-3.5-turbo", &choices).unwrap();
        assert_eq!(num_tokens, 33);
    }

    #[test]
    fn test_estimate_get_weather_example() {
        let messages = vec![
            Message::User(crate::entity::create_chat_completion::UserMessage {
                name: None,
                content: Content::Text("hi, how is the weather in San Francisco, CA".to_string()),
            }),
        ];
        let tools = vec![Tool {
            r#type: ToolType::Function,
            function: FunctionTool {
                name: "get_current_weather".to_string(),
                description: Some("Get the current weather in a given location".to_string()),
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
                })),
            },
        }];
        let num_tokens = prompt_tokens("gpt-3.5-turbo", &messages, Some(&tools)).unwrap();
        //82 vs 85
        assert_eq!(num_tokens, 85);
    }

    #[test]
    fn test_estimate_tool_with_one_param() {
        let messages = vec![
            Message::User(crate::entity::create_chat_completion::UserMessage {
                name: None,
                content: Content::Text("hi, how is the weather in San Francisco, CA".to_string()),
            }),
        ];
        let tools = vec![Tool {
            r#type: ToolType::Function,
            function: FunctionTool {
                name: "google_search".to_string(),
                description: Some("Search the web for the query provided, the web can provide realtime knowledge, news, and other information".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string",
                            "description": "Query content"
                        },
                    },
                })),
            },
        }];
        let num_tokens = prompt_tokens("gpt-3.5-turbo", &messages, Some(&tools)).unwrap();
        //68 vs 75
        assert_eq!(num_tokens, 75);
    }

    #[test]
    fn test_estimate_tool_with_three_params() {
        let messages = vec![
            Message::User(crate::entity::create_chat_completion::UserMessage {
                name: None,
                content: Content::Text("hi, how is the weather in San Francisco, CA".to_string()),
            }),
        ];
        let tools = vec![Tool {
            r#type: ToolType::Function,
            function: FunctionTool {
                name: "google_search".to_string(),
                description: Some("Search the web for the query provided, the web can provide realtime knowledge, news, and other information".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string",
                            "description": "Query content"
                        },
                        "hl": {
                            "type": "string",
                            "description": "Region"
                        },
						"cl": {
                            "type": "string",
                            "description": "locale"
                        }
                    },
                })),
            },
        }];
        let num_tokens = prompt_tokens("gpt-3.5-turbo", &messages, Some(&tools)).unwrap();
        //88 vs 89
        assert_eq!(num_tokens, 89);
    }

    #[test]
    fn estimate_two_tools() {
        let messages = vec![
            Message::User(crate::entity::create_chat_completion::UserMessage {
                name: None,
                content: Content::Text(
                    "get current traffic and current weather for San Francisco".to_string(),
                ),
            }),
        ];
        let tools = vec![Tool {
            r#type: ToolType::Function,
            function: FunctionTool {
                name: "google_search".to_string(),
                description: Some("Search the web for the query provided, the web can provide realtime knowledge, news, and other information".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string",
                            "description": "Query content"
                        },
                        "hl": {
                            "type": "string",
                            "description": "Region"
                        },
						"cl": {
                            "type": "string",
                            "description": "locale"
                        }
                    },
                    "required": ["q"]
                })),
            },
        },Tool {
            r#type: ToolType::Function,
            function: FunctionTool {
                name: "get_current_weather".to_string(),
                description: Some("Get the current weather in a given location".to_string()),
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
                })),
            },
        }];
        let num_tokens = prompt_tokens("gpt-3.5-turbo", &messages, Some(&tools)).unwrap();
        //157 vs 139
        assert_eq!(num_tokens, 139);
    }

    #[test]
    fn estimate_completion_one_tool() {
        let choices = vec![Choice {
            index: 0,
            message: ResponseMsg {
                content: None,
                tool_calls: Some(vec![
                    ToolCall::Function(ToolCallFunction {
                        id: "call_AwFYCZTRUzaOwnMKlxaqHDwW".to_string(),
                        function: ToolCallFunctionObj {
                            name: "get_current_weather".to_string(),
                            arguments: "{\"location\":\"New York\"}".to_string(),
                        },
                    }),
                ]),
                role: crate::entity::chat_completion_object::Role::Assistant,
            },
            finish_reason: Some(FinishReason::ToolCalls),
            logprobs: None,
        }];
        let num_tokens = completion_tokens("gpt-3.5-turbo", &choices).unwrap();
        //15 vs 16
        assert_eq!(num_tokens, 16);
    }

    #[test]
    fn estimate_completion_two_tools() {
        let choices = vec![Choice {
            index: 0,
            message: ResponseMsg {
                content: None,
                tool_calls: Some(vec![
                    ToolCall::Function(ToolCallFunction {
                        id: "call_2wo1EaYIIMqf7fKInNodsc06".to_string(),
                        function: ToolCallFunctionObj {
                            name: "get_current_weather".to_string(),
                            arguments: "{\"location\": \"New York\", \"unit\": \"celsius\"}"
                                .to_string(),
                        },
                    }),
                    ToolCall::Function(ToolCallFunction {
                        id: "call_Tg37fLaVErWp8xpuG6Z8XWkW".to_string(),
                        function: ToolCallFunctionObj {
                            name: "get_current_weather".to_string(),
                            arguments: "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"
                                .to_string(),
                        },
                    }),
                ]),
                role: crate::entity::chat_completion_object::Role::Assistant,
            },
            finish_reason: Some(FinishReason::ToolCalls),
            logprobs: None,
        }];
        let num_tokens = completion_tokens("gpt-3.5-turbo", &choices).unwrap();
        //46 vs 57
        assert_eq!(num_tokens, 57);
    }

    #[test]
    fn estimate_completion_three_tools() {
        let choices = vec![Choice {
            index: 0,
            message: ResponseMsg {
                content: None,
                tool_calls: Some(vec![
                    ToolCall::Function(ToolCallFunction {
                        id: "call_2wo1EaYIIMqf7fKInNodsc06".to_string(),
                        function: ToolCallFunctionObj {
                            name: "get_current_weather".to_string(),
                            arguments: "{\"location\": \"New York\", \"unit\": \"celsius\"}"
                                .to_string(),
                        },
                    }),
                    ToolCall::Function(ToolCallFunction {
                        id: "call_Tg37fLaVErWp8xpuG6Z8XWkW".to_string(),
                        function: ToolCallFunctionObj {
                            name: "get_current_weather".to_string(),
                            arguments: "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"
                                .to_string(),
                        },
                    }),
                    ToolCall::Function(ToolCallFunction {
                        id: "call_Tg37fLaVErWp8xpuG6Z8XWkW".to_string(),
                        function: ToolCallFunctionObj {
                            name: "get_current_weather".to_string(),
                            arguments: "{\"location\": \"Shanghai\", \"unit\": \"celsius\"}"
                                .to_string(),
                        },
                    }),
                ]),
                role: crate::entity::chat_completion_object::Role::Assistant,
            },
            finish_reason: Some(FinishReason::ToolCalls),
            logprobs: None,
        }];
        let num_tokens = completion_tokens("gpt-3.5-turbo", &choices).unwrap();
        //69 vs 80
        assert_eq!(num_tokens, 80);
    }
}
