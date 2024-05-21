use crate::entity::chat_completion_object::Usage;

pub fn price(model: &str, usage: &Usage) -> f32 {
    let (prompt_price, completion_price) = match model {
        "gpt-4o" => (0.005, 0.015),
        "gpt-4-turbo" => (0.01, 0.03),
        "gpt-4" => (0.03, 0.06),
        "gpt-3.5-turbo" => (0.0005, 0.0015),
        "gpt-3.5-turbo-instruct" => (0.0015, 0.002),
        _ => return 0.0, // Early return on unknown model
    };
    let total_price = (usage.prompt_tokens as f32 * prompt_price)
        + (usage.completion_tokens as f32 * completion_price);
    total_price / 1000.0
}
