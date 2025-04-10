use rmcp::model::Tool as RmcpTool;

use crate::entity::create_chat_completion::{FunctionTool, Tool, ToolType};

impl From<RmcpTool> for Tool {
    fn from(rmcp_tool: RmcpTool) -> Self {
        Tool {
            r#type: ToolType::Function,
            function: FunctionTool {
                name: rmcp_tool.name.clone(),
                description: Some(rmcp_tool.description),
                parameters: Some(serde_json::to_value(&*rmcp_tool.input_schema).unwrap()),
            },
        }
    }
}

impl From<Tool> for RmcpTool {
    fn from(tool: Tool) -> Self {
        RmcpTool {
            name: tool.function.name.clone(),
            description: tool
                .function
                .description
                .clone()
                .unwrap_or(String::new().into()),
            input_schema: match tool.function.parameters {
                Some(params) => serde_json::from_value(params).unwrap_or_default(),
                None => Default::default(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::borrow::Cow;

    #[test]
    fn test_convert_rmcp_tool_to_openai_tool() {
        // Create a sample RmcpTool
        let rmcp_tool = RmcpTool {
            name: "get_weather".into(),
            description: "Get the current weather in a location".into(),
            input_schema: serde_json::from_value(json!({
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
            .unwrap(),
        };

        // Convert to OpenAI Tool
        let openai_tool: Tool = rmcp_tool.into();

        // Verify conversion
        assert_eq!(openai_tool.r#type, ToolType::Function);
        assert_eq!(openai_tool.function.name, "get_weather");
        assert_eq!(
            openai_tool.function.description,
            Some(Cow::from("Get the current weather in a location"))
        );

        let params = openai_tool.function.parameters.unwrap();
        assert_eq!(params["type"], "object");
        assert!(params["properties"].is_object());
        assert!(params["properties"]["location"].is_object());
        assert_eq!(params["properties"]["location"]["type"], "string");
        assert!(params["required"].is_array());
        assert_eq!(params["required"][0], "location");
    }

    #[test]
    fn test_convert_openai_tool_to_rmcp_tool() {
        // Create a sample OpenAI Tool
        let openai_tool = Tool {
            r#type: ToolType::Function,
            function: FunctionTool {
                name: "get_weather".into(),
                description: Some("Get the current weather in a location".into()),
                parameters: Some(json!({
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
        };

        // Convert to RmcpTool
        let rmcp_tool: RmcpTool = openai_tool.into();

        // Verify conversion
        assert_eq!(rmcp_tool.name, "get_weather");
        assert_eq!(
            rmcp_tool.description,
            "Get the current weather in a location"
        );

        let schema = &*rmcp_tool.input_schema;
        assert!(schema.contains_key("type"));
        assert!(schema.contains_key("properties"));
        assert!(schema.contains_key("required"));

        // Verify it can be converted back to JSON
        let json_value = serde_json::to_value(&*rmcp_tool.input_schema).unwrap();
        assert_eq!(json_value["type"], "object");
        assert!(json_value["properties"].is_object());
        assert!(json_value["properties"]["location"].is_object());
    }

    #[test]
    fn test_handle_none_parameters() {
        // Create a tool with None parameters
        let openai_tool = Tool {
            r#type: ToolType::Function,
            function: FunctionTool {
                name: "simple_tool".into(),
                description: Some("A tool with no parameters".into()),
                parameters: None,
            },
        };

        // Convert to RmcpTool
        let rmcp_tool: RmcpTool = openai_tool.into();

        // Verify conversion
        assert_eq!(rmcp_tool.name, "simple_tool");
        assert_eq!(rmcp_tool.description, "A tool with no parameters");

        // The input_schema should be empty but valid
        assert!(rmcp_tool.input_schema.is_empty());
    }
}
