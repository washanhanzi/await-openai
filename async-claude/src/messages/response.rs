use serde::{Deserialize, Serialize};

use super::{ContentBlock, Role, StopReason, Usage};

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct Response {
    pub id: String,
    pub r#type: String,
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[cfg(test)]
mod tests {
    use crate::messages::{ContentBlock, Role};

    use super::*;
    #[test]
    fn serde() {
        let tests = vec![(
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
                content: vec![ContentBlock::Text {
                    text: "Hi! My name is Claude.".to_string(),
                }],
                role: Role::Assistant,
                stop_reason: Some(StopReason::EndTurn),
                stop_sequence: None,
                r#type: "message".to_string(),
                usage: Usage {
                    input_tokens: Some(10),
                    output_tokens: 25,
                },
            },
        )];
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
