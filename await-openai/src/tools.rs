use anyhow::{anyhow, Result};
use schemars::{gen::SchemaSettings, JsonSchema};

use crate::entity::create_chat_completion::{FunctionTool, Tool, ToolType};

pub fn get_function_param_json_value<T: JsonSchema>() -> &'static Result<serde_json::Value> {
    static PARAM_JSON: std::sync::OnceLock<Result<serde_json::Value>> = std::sync::OnceLock::new();
    PARAM_JSON.get_or_init(|| parse_function_param::<T>())
}

pub fn get_function_tool<T: JsonSchema>(name: &str, desc: Option<String>) -> &'static Result<Tool> {
    static FUNCTION_TOOL: std::sync::OnceLock<Result<Tool>> = std::sync::OnceLock::new();
    FUNCTION_TOOL.get_or_init(|| {
        let json_value = parse_function_param::<T>()?;
        Ok(Tool {
            r#type: ToolType::Function,
            function: FunctionTool {
                name: name.to_string(),
                description: desc,
                parameters: Some(json_value.clone()),
            },
        })
    })
}

fn parse_function_param<T: JsonSchema>() -> Result<serde_json::Value> {
    let settings = SchemaSettings::default().with(|s| {
        s.option_nullable = false;
        s.option_add_null_type = false;
        s.inline_subschemas = true;
    });
    let schema = settings.into_generator().into_root_schema_for::<T>();
    let mut json_value = serde_json::to_value(schema)?;
    let schema_type = json_value
        .get("type")
        .ok_or_else(|| anyhow!("Require json schema type"))?;
    if *schema_type != serde_json::Value::String("object".to_string()) {
        return Err(anyhow!("Require json schema type object"));
    }
    if let Some(obj) = json_value.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");
        obj.remove("definitions");
    };
    Ok(json_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;

    #[derive(JsonSchema, serde::Deserialize)]
    pub struct MyStruct {
        /// The city and state, e.g. San Francisco, CA
        pub location: String,
        pub unit: Option<UnitEnum>,
    }

    #[derive(JsonSchema, serde::Deserialize, PartialEq, Debug)]
    #[serde(rename_all = "lowercase")]
    pub enum UnitEnum {
        Celsius,
        Fahrenheit,
    }

    #[test]
    fn test_serialize_example_function_tool() {
        let schema = get_function_param_json_value::<MyStruct>()
            .as_ref()
            .unwrap();

        // println!("{}", serde_json::to_string_pretty(&schema).unwrap());
        let got = serde_json::to_string(&schema).unwrap();
        let want = serde_json::json!(
            {
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
        })
        .to_string();
        assert_eq!(want, got, "Expected: {} Got: {}", want, got);
    }

    #[test]
    fn test_deserilize_example_function_tool() {
        let s = "{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}";
        let got: MyStruct = serde_json::from_str(s).unwrap();
        assert_eq!(
            "Boston, MA", got.location,
            "Expected: Boston, MA Got: {}",
            got.location
        );
        assert_eq!(
            Some(UnitEnum::Celsius),
            got.unit,
            "Expected: celsius got: {:?}",
            got.unit
        );
    }
}
