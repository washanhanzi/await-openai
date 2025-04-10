use await_openai::{
    define_function_tool,
    entity::create_chat_completion::Tool,
    tool::{JsonSchema, schemars},
};

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

    define_function_tool!(
        GET_WEATHER,
        "get_current_weather",
        "Get the current weather in the location closest to the one provided location",
        MyStruct
    );

    let _tools: Vec<Tool> = vec![get_get_weather().unwrap().clone()];
}
