use crate::entity::create_chat_completion::{Content, ContentPart, ImageUrlDetail, Message};
use anyhow::{anyhow, Result};
use tiktoken_rs::{
    get_bpe_from_tokenizer,
    tokenizer::{get_tokenizer, Tokenizer},
};

pub fn get_prompt_tokens(model: &str, messages: &[Message]) -> Result<i32> {
    let tokenizer =
        get_tokenizer(model).ok_or_else(|| anyhow!("No tokenizer found for model {}", model))?;
    if tokenizer != Tokenizer::Cl100kBase {
        anyhow::bail!("Only Cl100kBase model is supported for now")
    }
    let bpe = get_bpe_from_tokenizer(tokenizer)?;

    let (tokens_per_message, tokens_per_name) = if model.starts_with("gpt-3.5") {
        (
            4,  // every message follows <im_start>{role/name}\n{content}<im_end>\n
            -1, // if there's a name, the role is omitted
        )
    } else {
        (3, 1)
    };
    let mut num_tokens: i32 = 0;
    for message in messages {
        num_tokens += tokens_per_message;
        match message {
            Message::System(m) => {
                if let Some(name) = m.name.as_ref() {
                    num_tokens += tokens_per_name;
                    num_tokens += bpe.encode_with_special_tokens(name).len() as i32;
                }
                num_tokens += bpe.encode_with_special_tokens(&m.content).len() as i32
            }
            Message::User(m) => {
                if let Some(name) = m.name.as_ref() {
                    num_tokens += tokens_per_name;
                    num_tokens += bpe.encode_with_special_tokens(name).len() as i32;
                }
                match &m.content {
                    Content::Text(text) => {
                        num_tokens += bpe.encode_with_special_tokens(text).len() as i32
                    }
                    Content::Array(array) => {
                        for part in array {
                            match part {
                                ContentPart::Text(t) => {
                                    num_tokens +=
                                        bpe.encode_with_special_tokens(&t.text).len() as i32;
                                }
                                ContentPart::Image(image) => {
                                    if let Some((w, h)) = image.dimensions {
                                        num_tokens +=
                                            get_image_tokens((w, h), &image.image_url.detail)
                                                as i32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Message::Assistant(m) => {
                if let Some(name) = m.name.as_ref() {
                    num_tokens += tokens_per_name;
                    num_tokens += bpe.encode_with_special_tokens(name).len() as i32;
                }
                num_tokens += bpe
                    .encode_with_special_tokens(m.content.as_deref().unwrap_or(""))
                    .len() as i32
            }
            Message::Tool(m) => {
                num_tokens += bpe.encode_with_special_tokens(&m.content).len() as i32
            }
        }
    }
    num_tokens += 3; // every reply is primed with <|start|>assistant<|message|>
    Ok(num_tokens)
}

fn get_image_tokens(image: (u32, u32), detail: &Option<ImageUrlDetail>) -> u32 {
    match detail {
        Some(ImageUrlDetail::Low) => 85,
        None | Some(ImageUrlDetail::Auto) => {
            let w = image.0.div_ceil(512);
            let h = image.1.div_ceil(512);
            if w <= 1 && h <= 1 {
                return 85;
            }
            85 + 170 * w * h
        }
        //detail: high images are first scaled to fit within a 2048 x 2048 square, maintaining their aspect ratio. Then, they are scaled such that the shortest side of the image is 768px long. Finally, we count how many 512px squares the image consists of.
        //case 1: A 1024 x 1024 square image in detail: high mode costs 765 tokens
        //1024 is less than 2048, so there is no initial resize.
        //The shortest side is 1024, so we scale the image down to 768 x 768.
        //4 512px square tiles are needed to represent the image, so the final token cost is 170 * 4 + 85 = 765.
        //case 2: A 2048 x 4096 image in detail: high mode costs 1105 tokens
        //We scale down the image to 1024 x 2048 to fit within the 2048 square.
        //The shortest side is 1024, so we further scale down to 768 x 1536.
        //6 512px tiles are needed, so the final token cost is 170 * 6 + 85 = 1105.
        Some(ImageUrlDetail::High) => {
            //get min max of the image width and height
            let (mut min, mut max) = {
                let width = image.0 as f32;
                let height = image.1 as f32;
                (width.min(height), width.max(height))
            };
            //if the max side is above 2048, scale down to 2048
            if max > 2048.0 {
                let scale = 2048.0 / max;
                max = 2048.0;
                min *= scale;
            }

            //scale the shortest side to 768
            let scale = 768.0 / min;
            max *= scale;

            //fit into 512 squares
            let max_squares = (max / 512.0).ceil() as u32;

            //min_squares is always 768.div_ceil(512)=2
            //170*2=340
            85 + 340 * max_squares
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::create_chat_completion::{Content, Message};

    #[test]
    fn test_get_prompt_token() {
        let messages = vec![
            Message::System(crate::entity::create_chat_completion::SystemMessage {
                name: None,
                content: "You are a helpful assistant.".to_string(),
            }),
            Message::User(crate::entity::create_chat_completion::UserMessage {
                name: None,
                content: Content::Text("hi, how are you".to_string()),
            }),
        ];
        let num_tokens = get_prompt_tokens("gpt-3.5-turbo-1106", &messages).unwrap();
        assert_eq!(num_tokens, 22);
    }
    #[test]
    fn test_get_image_tokens() {
        let test_data = vec![
            ("512-auto", 512, 512, Some(ImageUrlDetail::Auto), 85),
            ("512-none", 512, 512, None, 85),
            ("512-low", 512, 512, Some(ImageUrlDetail::Low), 85),
            ("4096-low", 4096, 8192, Some(ImageUrlDetail::Low), 85),
            ("1024-high", 1024, 1024, Some(ImageUrlDetail::High), 765),
            ("2048-high", 2048, 4096, Some(ImageUrlDetail::High), 1105),
        ];
        for t in test_data {
            let got = get_image_tokens((t.1, t.2), &t.3);
            assert_eq!(got, t.4, "test case: {}", t.0)
        }
    }
}
