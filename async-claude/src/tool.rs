use anyhow::{anyhow, Result};
pub use paste;
use schemars::r#gen::SchemaSettings;
pub use schemars::{self, JsonSchema};
use std::borrow::Cow;

use crate::messages::Tool;

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
    let settings = SchemaSettings::draft2019_09().with(|s| {
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
