use std::borrow::Cow;

use anyhow::{Result, anyhow};
pub use schemars::{self, JsonSchema};
use schemars::{
    generate::SchemaSettings,
    transform::{self, Transform},
};
use serde_json::Value;

use crate::entity::create_chat_completion::{FunctionTool, Tool, ToolType};
use async_claude::messages::Tool as ClaudeTool;
pub use paste;

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AddNullable {
    /// When set to `true` (the default), `"null"` will also be removed from the schemas `type`.
    pub remove_null_type: bool,
}

impl Default for AddNullable {
    fn default() -> Self {
        Self {
            remove_null_type: true,
        }
    }
}

impl AddNullable {
    fn has_type(schema: &schemars::Schema, ty: &str) -> bool {
        match schema.get("type") {
            Some(Value::Array(values)) => values.iter().any(|v| v.as_str() == Some(ty)),
            Some(Value::String(s)) => s == ty,
            _ => false,
        }
    }
}

impl Transform for AddNullable {
    fn transform(&mut self, schema: &mut schemars::Schema) {
        if Self::has_type(schema, "null") {
            // Don't add nullable property, just handle the null type removal
            if let Some(ty) = schema.get_mut("type") {
                if self.remove_null_type {
                    // Remove null from type array and clean up enum if present
                    if let Value::Array(array) = ty {
                        array.retain(|t| t.as_str() != Some("null"));
                        if array.len() == 1 {
                            *ty = array[0].clone();
                        }
                    }

                    // Also clean up enum arrays that contain null
                    if let Some(enum_val) = schema.get_mut("enum") {
                        if let Value::Array(enum_array) = enum_val {
                            enum_array.retain(|v| !v.is_null());
                        }
                    }
                }
            }
        }

        transform::transform_subschemas(self, schema);
    }
}

/// get_function_tool accept function name, description and parameters type and return [Tool]
/// use define_function_tool macro to create tool if you need a static value
pub fn get_function_tool<T: JsonSchema, S1, S2>(name: S1, desc: Option<S2>) -> Result<Tool>
where
    S1: Into<Cow<'static, str>>,
    S2: Into<Cow<'static, str>>,
{
    let json_value = parse_function_param::<T>()?;
    Ok(Tool {
        r#type: ToolType::Function,
        function: FunctionTool {
            name: name.into(),
            description: desc.map(Into::into),
            parameters: Some(json_value),
        },
    })
}

/// define_function_tool macro will create a function get_{tool_name in lowercase}, the function return a static reference to the tool
#[macro_export]
macro_rules! define_function_tool {
    ($tool_name:ident, $function_name:expr, $description:expr, $param_type:ty) => {
        $crate::tool::paste::paste! {
            static [<$tool_name _ONCE_LOCK>]: std::sync::OnceLock<anyhow::Result<$crate::entity::create_chat_completion::Tool>> = ::std::sync::OnceLock::new();

            pub fn [<get_ $tool_name:lower>]() -> Result<&'static $crate::entity::create_chat_completion::Tool, &'static anyhow::Error> {
                [<$tool_name _ONCE_LOCK>].get_or_init(|| {
                    $crate::tool::get_function_tool::<$param_type, _, _>(
                        $function_name,
                        Some($description),
                    )
                }).as_ref()
            }
        }
    };
}

fn parse_function_param<T: JsonSchema>() -> Result<serde_json::Value> {
    let settings = SchemaSettings::draft2020_12()
        .with(|s| {
            // s.option_nullable = false;
            // s.option_add_null_type = false;
            s.inline_subschemas = true;
        })
        .with_transform(AddNullable::default());
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

impl From<ClaudeTool> for Tool {
    fn from(claude_tool: ClaudeTool) -> Self {
        Tool {
            r#type: ToolType::Function,
            function: FunctionTool {
                name: claude_tool.name.clone(),
                description: claude_tool.description,
                parameters: Some(claude_tool.input_schema),
            },
        }
    }
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
        pub arr: Option<Vec<String>>,
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
                },
                "arr": {
                  "type": "array",
                  "items": {
                    "type": "string"
                  }
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
        let tool = get_my_tool().unwrap();
        assert_eq!(
            tool.r#type,
            crate::entity::create_chat_completion::ToolType::Function
        );
        assert_eq!(tool.function.name, "my_tool");
        assert_eq!(
            tool.function.description,
            Some("my tool description".to_string().into())
        );
        assert!(tool.function.parameters.is_some());

        define_function_tool!(MY_TOOL2, "my_tool2", "my tool description", MyStruct);
        let tool2 = get_my_tool2().unwrap();
        assert_eq!(
            tool2.r#type,
            crate::entity::create_chat_completion::ToolType::Function
        );
        assert_eq!(tool2.function.name, "my_tool2");
        assert_eq!(
            tool2.function.description,
            Some("my tool description".to_string().into())
        );
        assert!(tool2.function.parameters.is_some());
    }
}
