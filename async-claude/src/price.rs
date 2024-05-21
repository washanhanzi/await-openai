use crate::messages::Usage;

pub fn price(model: &str, usage: &Usage) -> f32 {
    let (prompt_price, completion_price) = match model {
        "claude-3-opus-20240229" => (0.00025, 0.00125),
        "claude-3-sonnet-20240229" => (0.003, 0.015),
        "claude-3-haiku-20240307" => (0.015, 0.075),
        _ => return 0.0,
    };
    let price = usage.input_tokens.unwrap_or_default() as f32 * prompt_price
        + usage.output_tokens as f32 * completion_price;
    price / 1000.0
}
