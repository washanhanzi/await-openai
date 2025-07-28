use anyhow::{anyhow, Result};
pub use paste;
use schemars::{
    generate::SchemaSettings,
    transform::{self, Transform},
};
pub use schemars::{self, JsonSchema};
use serde_json::Value;
use std::borrow::Cow;

use crate::messages::Tool;

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

pub fn get_tool<T: JsonSchema, S1, S2>(name: S1, desc: Option<S2>) -> Result<Tool>
where
    S1: Into<Cow<'static, str>>,
    S2: Into<Cow<'static, str>>,
{
    let json_value = parse_input_schema::<T>()?;
    Ok(Tool {
        name: name.into(),
        description: desc.map(Into::into),
        input_schema: json_value,
    })
}

#[macro_export]
macro_rules! define_tool {
    ($tool_name:ident, $function_name:expr, $description:expr, $param_type:ty) => {
        $crate::tool::paste::paste! {
            static [<$tool_name _ONCE_LOCK>]: std::sync::OnceLock<anyhow::Result<$crate::messages::Tool>> = ::std::sync::OnceLock::new();

            pub fn [<get_ $tool_name:lower>]() -> Result<&'static $crate::messages::Tool, &'static anyhow::Error> {
                [<$tool_name _ONCE_LOCK>].get_or_init(|| {
                    $crate::tool::get_tool::<$param_type, _, _>(
                        $function_name,
                        Some($description),
                    )
                }).as_ref()
            }
        }
    };
}

pub fn parse_input_schema<T: JsonSchema>() -> Result<serde_json::Value> {
    let settings = SchemaSettings::draft2019_09()
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
