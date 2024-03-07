use anyhow::{anyhow, Result};
use schemars::{gen::SchemaSettings, JsonSchema};

use crate::entity::create_chat_completion::{FunctionTool, Tool, ToolType};
pub use paste;

/// get_function_tool accept function name, description and parameters type and return [Tool]
/// use define_function_tool macro to create tool if you want to use the tool multiple times
pub fn get_function_tool<T: JsonSchema>(name: &str, desc: Option<String>) -> Result<Tool> {
    let json_value = parse_function_param::<T>()?;
    Ok(Tool {
        r#type: ToolType::Function,
        function: FunctionTool {
            name: name.to_string(),
            description: desc,
            parameters: Some(json_value),
        },
    })
}

/// define_function_tool macro will create a static OnceLock<Tool> with the name <TOOL_NAME>_ONCE_LOCK
#[macro_export]
macro_rules! define_function_tool {
    ($tool_name:ident, $function_name:expr, $description:expr, $param_type:ty) => {
        $crate::tool::paste::paste! {
            static [<$tool_name _ONCE_LOCK>]: std::sync::OnceLock<$crate::entity::create_chat_completion::Tool> = std::sync::OnceLock::new();

            pub fn [<get_ $tool_name:lower>]() -> &'static $crate::entity::create_chat_completion::Tool {
                [<$tool_name _ONCE_LOCK>].get_or_init(|| {
                    $crate::tool::get_function_tool::<$param_type>(
                        $function_name,
                        Some($description.to_string()),
                    )
                    .unwrap()
                })
            }
        }
    };
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
    use schemars::JsonSchema;

    use crate::tool::parse_function_param;

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
        let schema = parse_function_param::<MyStruct>().unwrap();

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

    #[test]
    fn test_macro() {
        define_function_tool!(MY_TOOL, "my_tool", "my tool description", MyStruct);
        let tool = get_my_tool();
        assert_eq!(
            tool.r#type,
            crate::entity::create_chat_completion::ToolType::Function
        );
        assert_eq!(tool.function.name, "my_tool");
        assert_eq!(
            tool.function.description,
            Some("my tool description".to_string())
        );
        assert!(tool.function.parameters.is_some());

        define_function_tool!(MY_TOOL2, "my_tool2", "my tool description", MyStruct);
        let tool2 = get_my_tool2();
        assert_eq!(
            tool2.r#type,
            crate::entity::create_chat_completion::ToolType::Function
        );
        assert_eq!(tool2.function.name, "my_tool2");
        assert_eq!(
            tool2.function.description,
            Some("my tool description".to_string())
        );
        assert!(tool2.function.parameters.is_some());
    }
}
