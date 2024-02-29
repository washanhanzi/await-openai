use std::collections::VecDeque;

use crate::entity::create_chat_completion::{
    Content, ContentPart, Message, Tool, ToolCall, ToolType,
};
use anyhow::{anyhow, Result};
use tiktoken_rs::{
    get_bpe_from_tokenizer,
    tokenizer::{get_tokenizer, Tokenizer},
    CoreBPE,
};

mod image_token;
pub use image_token::get_image_tokens;

pub struct TokenCounter {
    contents: VecDeque<String>,
    token_per_name: i32,
    is_contain_system_message: bool,
}

impl TokenCounter {
    pub fn new(token_per_name: i32) -> Self {
        TokenCounter {
            contents: VecDeque::new(),
            token_per_name,
            is_contain_system_message: false,
        }
    }

    pub fn is_contain_system_message(&self) -> bool {
        self.is_contain_system_message
    }

    pub fn push(&mut self, content: &str) {
        if content.is_empty() {
            return;
        }
        self.contents.push_back(content.to_string());
    }

    pub fn parse_message(&mut self, message: &Message) -> i32 {
        let mut temp_counter = 0;
        match message {
            Message::System(m) => {
                if let Some(name) = m.name.as_deref() {
                    temp_counter += self.token_per_name;
                    self.push(name);
                }
                self.push(&m.content);
            }
            Message::User(m) => {
                if let Some(name) = m.name.as_deref() {
                    temp_counter += self.token_per_name;
                    self.push(name);
                }
                match &m.content {
                    Content::Text(text) => {
                        self.push(text);
                    }
                    Content::Array(array) => {
                        for part in array {
                            match part {
                                ContentPart::Text(t) => {
                                    self.push(&t.text);
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
                    temp_counter += self.token_per_name;
                    self.push(name);
                }
                if let Some(content) = m.content.as_deref() {
                    self.push(content);
                }
                if let Some(tools) = &m.tool_calls {
                    for tool in tools {
                        match tool {
                            ToolCall::Function(function_call) => {
                                self.push(&function_call.function.name);
                                self.push(&function_call.function.arguments);
                            }
                        }
                    }
                }
            }
            Message::Tool(m) => {
                self.push(&m.content);
            }
        }
        temp_counter
    }

    pub fn get_count(self, bpe: &CoreBPE) -> u32 {
        let concatenated_contents = self.contents.into_iter().collect::<Vec<String>>().join(" ");
        bpe.encode_with_special_tokens(&concatenated_contents).len() as u32
    }
}

/// get_prompt_tokens calculates the token usage for prompts.
/// This function provides an estimated count when the prompt includes [`Tool`] and [`AssistantMessage`]'s [`ToolCall`].
/// It's important to note that an exact methodology for calculating token usage for [`Tool`] and [`ToolCall`] has not been disclosed by OpenAI.
/// For context, see [generating TypeScript definitions](https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/6) as an example.
/// Given the absence of an official method and unreliable results of the above method other than get_current_weather example,
/// this implementation adopts a more efficient approach to estimate token usage, potentially sacrificing some degree of accuracy for improved performance and simplicity.
/// You can check the test cases for the estimated and actual token usage.
///
/// [`AssistantMessage`]: crate::entity::create_chat_completion::AssistantMessage
pub fn get_prompt_tokens(model: &str, messages: &[Message], tools: Option<&[Tool]>) -> Result<u32> {
    let tokenizer =
        get_tokenizer(model).ok_or_else(|| anyhow!("No tokenizer found for model {}", model))?;
    if tokenizer != Tokenizer::Cl100kBase {
        anyhow::bail!("Only Cl100kBase model is supported for now")
    }
    let bpe = get_bpe_from_tokenizer(tokenizer)?;

    //now gpt-3.5-turbo and gpt-4-turbo both use 4 as token per message
    //it seems the following logic is deprecated for now
    // let (tokens_per_message, tokens_per_name) = if model.starts_with("gpt-3.5") {
    //     (
    //         4,  // every message follows <im_start>{role/name}\n{content}<im_end>\n
    //         -1, // if there's a name, the role is omitted
    //     )
    // } else {
    //     (3, 1)
    // };
    let (tokens_per_message, tokens_per_name) = (
        4,  // every message follows <im_start>{role/name}\n{content}<im_end>\n
        -1, // if there's a name, the role is omitted
    );
    let mut counter = TokenCounter::new(tokens_per_name);

    let mut num_tokens: i32 = 0;
    for message in messages {
        num_tokens += tokens_per_message;
        let temp = counter.parse_message(message);
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
                        counter.push("// ");
                        counter.push(desc);
                        counter.push("\n");
                    }
                    counter.push("function ");
                    counter.push(&tool.function.name);
                    counter.push("\n");
                    // tool.function.parameters is a serde_json::Value
                    if let Some(parameters_json) = tool.function.parameters.as_ref() {
                        counter.push(&parameters_json.to_string());
                    }
                }
            }
        }
        if counter.is_contain_system_message() {
            num_tokens -= tokens_per_message;
        }
    }
    num_tokens += counter.get_count(&bpe) as i32;
    Ok(num_tokens as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::create_chat_completion::{Content, FunctionTool, Message};

    #[test]
    fn test_messages() {
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
        let num_tokens = get_prompt_tokens("gpt-3.5-turbo", &messages, None).unwrap();
        assert_eq!(num_tokens, 22);
    }

    #[test]
    fn test_estimate_get_weather_example() {
        let messages = vec![Message::User(
            crate::entity::create_chat_completion::UserMessage {
                name: None,
                content: Content::Text("hi, how is the weather in San Francisco, CA".to_string()),
            },
        )];
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
        let num_tokens = get_prompt_tokens("gpt-3.5-turbo", &messages, Some(&tools)).unwrap();
        //82 vs 85
        assert_eq!(num_tokens, 85);
    }

    #[test]
    fn test_estimate_tool_with_one_param() {
        let messages = vec![Message::User(
            crate::entity::create_chat_completion::UserMessage {
                name: None,
                content: Content::Text("hi, how is the weather in San Francisco, CA".to_string()),
            },
        )];
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
        let num_tokens = get_prompt_tokens("gpt-3.5-turbo", &messages, Some(&tools)).unwrap();
        //65 vs 75
        assert_eq!(num_tokens, 75);
    }

    #[test]
    fn test_estimate_tool_with_three_params() {
        let messages = vec![Message::User(
            crate::entity::create_chat_completion::UserMessage {
                name: None,
                content: Content::Text("hi, how is the weather in San Francisco, CA".to_string()),
            },
        )];
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
        let num_tokens = get_prompt_tokens("gpt-3.5-turbo", &messages, Some(&tools)).unwrap();
        //85 vs 89
        assert_eq!(num_tokens, 89);
    }

    #[test]
    fn test_estimate_two_tools() {
        let messages = vec![Message::User(
            crate::entity::create_chat_completion::UserMessage {
                name: None,
                content: Content::Text(
                    "get current traffic and current weather for San Francisco".to_string(),
                ),
            },
        )];
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
        let num_tokens = get_prompt_tokens("gpt-3.5-turbo", &messages, Some(&tools)).unwrap();
        //151 vs 139
        assert_eq!(num_tokens, 139);
    }
}
