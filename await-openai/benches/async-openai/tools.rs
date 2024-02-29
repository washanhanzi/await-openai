use async_openai::types::ChatCompletionToolType;
use await_openai::openai::{entity::create_chat_completion::Tool, tool::get_function_tool};
use openai_func_enums::{
    arg_description, get_tool_chat_completion_args, EnumDescriptor, FunctionCallResponse,
    VariantDescriptors,
};
use schemars::JsonSchema;
use serde::Deserialize;

pub fn de_function_tool_param() {
    #[derive(JsonSchema, serde::Deserialize)]
    pub struct MyStruct {
        /// The only valid locations that can be passed.
        pub location: Location,
        /// A temperature unit chosen from the enum.
        pub unit: Option<UnitEnum>,
    }

    #[derive(JsonSchema, serde::Deserialize, PartialEq, Debug)]
    #[serde(rename_all = "lowercase")]
    pub enum Location {
        Atlanta,
        Boston,
        Chicago,
        Dallas,
        Denver,
        LosAngeles,
        Miami,
        Nashville,
        NewYork,
        Philadelphia,
        Seattle,
        StLouis,
        Washington,
    }

    #[derive(JsonSchema, serde::Deserialize, PartialEq, Debug)]
    #[serde(rename_all = "lowercase")]
    pub enum UnitEnum {
        Celsius,
        Fahrenheit,
    }

    let _tools: Vec<Tool> = vec![get_function_tool::<MyStruct>(
        "get_current_weather",
        Some(
            "Get the current weather in the location closest to the one provided location"
                .to_string(),
        ),
    )
    .unwrap()];
}

pub fn de_function_tool_param_use_func_enums() {
    #[derive(Debug, FunctionCallResponse)]
    pub enum FunctionDef {
        #[func_description(
            description = "Get the current weather in the location closest to the one provided location"
        )]
        GetCurrentWeather(Location, TemperatureUnits),
    }

    #[derive(Clone, Debug, Deserialize, EnumDescriptor, VariantDescriptors)]
    #[arg_description(description = "The only valid locations that can be passed.")]
    pub enum Location {
        Atlanta,
        Boston,
        Chicago,
        Dallas,
        Denver,
        LosAngeles,
        Miami,
        Nashville,
        NewYork,
        Philadelphia,
        Seattle,
        StLouis,
        Washington,
    }

    #[derive(Clone, Debug, Deserialize, EnumDescriptor, VariantDescriptors)]
    #[arg_description(description = "A temperature unit chosen from the enum.")]
    pub enum TemperatureUnits {
        Celcius,
        Fahrenheit,
    }
    let tool_args =
        get_tool_chat_completion_args(GetCurrentWeatherResponse::get_function_json).unwrap();
    let _param = tool_args.0;
}
