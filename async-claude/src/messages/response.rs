use serde::{Deserialize, Serialize};

use super::{ResponseContentBlock, Role, StopReason, Usage};

#[derive(Debug, Deserialize, Default, Clone, PartialEq, Serialize)]
pub struct Response {
    pub id: String,
    pub r#type: String,
    pub role: Role,
    pub content: Vec<ResponseContentBlock>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[cfg(test)]
mod tests {
    use crate::messages::{
        BaseContentBlock, RedactedThinkingContentBlock, Role, ToolUseContentBlock,
    };

    use super::*;
    #[test]
    fn serde() {
        let tests = vec![
            (
                "simple",
                r#"{
                "content": [
                  {
                    "text": "Hi! My name is Claude.",
                    "type": "text"
                  }
                ],
                "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
                "model": "claude-3-opus-20240229",
                "role": "assistant",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "type": "message",
                "usage": {
                  "input_tokens": 10,
                  "output_tokens": 25
                }
              }"#,
                Response {
                    id: "msg_013Zva2CMHLNnXjNJJKqJ2EF".to_string(),
                    model: "claude-3-opus-20240229".to_string(),
                    content: vec![
                        ResponseContentBlock::Base(BaseContentBlock::Text {
                            text: "Hi! My name is Claude.".to_string(),
                        }),
                    ],
                    role: Role::Assistant,
                    stop_reason: Some(StopReason::EndTurn),
                    stop_sequence: None,
                    r#type: "message".to_string(),
                    usage: Usage {
                        input_tokens: Some(10),
                        output_tokens: 25,
                    },
                },
            ),
            (
                "tool_use",
                r#"{
                    "id": "msg_01Aq9w938a90dw8q",
                    "type": "message",
                    "model": "claude-3-opus-20240229",
                    "stop_reason": "tool_use",
                    "role": "assistant",
                    "stop_sequence": null,
                    "content": [
                      {
                        "type": "text",
                        "text": "<thinking>I need to call the get_weather function, and the user wants SF, which is likely San Francisco, CA.</thinking>"
                      },
                      {
                        "type": "tool_use",
                        "id": "toolu_01A09q90qw90lq917835lq9",
                        "name": "get_weather",
                        "input": {"location": "San Francisco, CA", "unit": "celsius"}
                      }
                    ],
                    "usage": {
                        "input_tokens": 527,
                        "output_tokens": 137
                    }
                  }"#,
                Response {
                    id: "msg_01Aq9w938a90dw8q".to_string(),
                    model: "claude-3-opus-20240229".to_string(),
                    content: vec![
                        ResponseContentBlock::Base(BaseContentBlock::Text {
                            text: "<thinking>I need to call the get_weather function, and the user wants SF, which is likely San Francisco, CA.</thinking>".to_string(),
                        }),
                        ResponseContentBlock::Base(BaseContentBlock::ToolUse(ToolUseContentBlock {
                            id: "toolu_01A09q90qw90lq917835lq9".to_string(),
                            name: "get_weather".to_string(),
                            input: serde_json::json!({
                                "location": "San Francisco, CA",
                                "unit": "celsius"
                            }),
                        })),
                    ],
                    role: Role::Assistant,
                    stop_reason: Some(StopReason::ToolUse),
                    stop_sequence: None,
                    r#type: "message".to_string(),
                    usage: Usage {
                        input_tokens: Some(527),
                        output_tokens: 137,
                    },
                },
            ),
            (
                "thinking_block",
                r#"{
                  "id": "msg_01Bq9w938a90dw8r",
                  "type": "message",
                  "model": "claude-3-opus-20240229",
                  "stop_reason": "end_turn",
                  "role": "assistant",
                  "stop_sequence": null,
                  "content": [
                    {
                      "type": "thinking",
                      "thinking": "Let me analyze this step by step...",
                      "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve/op91XWHOEBLLqIOMfFG/UvLEczmEsUjavL...."
                    },
                    {
                      "type": "redacted_thinking",
                      "data": "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpP..."
                    },
                    {
                      "type": "text",
                      "text": "Based on my analysis..."
                    }
                  ],
                  "usage": {
                    "input_tokens": 320,
                    "output_tokens": 150
                  }
                }"#,
                Response {
                    id: "msg_01Bq9w938a90dw8r".to_string(),
                    model: "claude-3-opus-20240229".to_string(),
                    content: vec![
                        ResponseContentBlock::Base(BaseContentBlock::Thinking {
                            thinking: "Let me analyze this step by step...".to_string(),
                            signature: Some("WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve/op91XWHOEBLLqIOMfFG/UvLEczmEsUjavL....".to_string()),
                        }),
                        ResponseContentBlock::RedactedThinking(RedactedThinkingContentBlock::RedactedThinking {
                            data: "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpP...".to_string(),
                        }),
                        ResponseContentBlock::Base(BaseContentBlock::Text {
                            text: "Based on my analysis...".to_string(),
                        }),
                    ],
                    role: Role::Assistant,
                    stop_reason: Some(StopReason::EndTurn),
                    stop_sequence: None,
                    r#type: "message".to_string(),
                    usage: Usage {
                        input_tokens: Some(320),
                        output_tokens: 150,
                    },
                },
            ),
        ];
        for (name, json, expected) in tests {
            //test deserialize
            let actual: Response = serde_json::from_str(json).unwrap();
            assert_eq!(actual, expected, "deserialize test failed: {}", name);
            //test serialize
            let serialized = serde_json::to_string(&expected).unwrap();
            let actual: Response = serde_json::from_str(&serialized).unwrap();
            assert_eq!(actual, expected, "serialize test failed: {}", name);
        }
    }
}
