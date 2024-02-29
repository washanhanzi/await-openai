use crate::entity::create_chat_completion::ImageUrlDetail;

const BASE_TOKENS: u32 = 85;
const TOKENS_PER_TILE: u32 = 170;
const HIGH_DETAIL_THRESHOLD: f32 = 2048.0;

pub fn get_image_tokens(image: (u32, u32), detail: &Option<ImageUrlDetail>) -> u32 {
    match detail {
        Some(ImageUrlDetail::Low) => BASE_TOKENS,
        None | Some(ImageUrlDetail::Auto) => {
            let (min, max) = {
                let width = image.0 as f32;
                let height = image.1 as f32;
                (width.min(height), width.max(height))
            };
            //<2048, non high detail mode
            if max < HIGH_DETAIL_THRESHOLD {
                let w = image.0.div_ceil(512);
                let h = image.1.div_ceil(512);
                return BASE_TOKENS + TOKENS_PER_TILE * w * h;
            }
            //high detail mode
            cal_high_detail_image_token(min, max)
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
            let (min, max) = {
                let width = image.0 as f32;
                let height = image.1 as f32;
                (width.min(height), width.max(height))
            };
            cal_high_detail_image_token(min, max)
        }
    }
}

fn cal_high_detail_image_token(mut min: f32, mut max: f32) -> u32 {
    //if the max side is above 2048, scale down to 2048
    if max > HIGH_DETAIL_THRESHOLD {
        let scale = HIGH_DETAIL_THRESHOLD / max;
        max = HIGH_DETAIL_THRESHOLD;
        min *= scale;
    }

    //scale the shortest side to 768
    let scale = 768.0 / min;
    max *= scale;

    //fit into 512 squares
    let max_squares = (max / 512.0).ceil() as u32;

    //min_squares is always 768.div_ceil(512)=2
    //170*2=340
    BASE_TOKENS + 340 * max_squares
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_image_tokens() {
        //https://openai.com/pricing
        let test_data = vec![
            ("512-auto", 512, 512, Some(ImageUrlDetail::Auto), 255),
            ("512-none", 512, 512, None, 255),
            ("512-low", 512, 512, Some(ImageUrlDetail::Low), 85),
            ("4096-low", 4096, 8192, Some(ImageUrlDetail::Low), 85),
            ("1024-high", 1024, 1024, Some(ImageUrlDetail::High), 765),
            ("2048-high", 2048, 4096, Some(ImageUrlDetail::High), 1105),
            ("150-auto", 150, 150, Some(ImageUrlDetail::Auto), 255),
            ("1024-auto", 1024, 1024, Some(ImageUrlDetail::Auto), 765),
            ("1000-auto", 1000, 500, Some(ImageUrlDetail::Auto), 425),
            ("2048-auto", 2048, 1024, Some(ImageUrlDetail::Auto), 1105),
            ("4096-auto", 4096, 5801, Some(ImageUrlDetail::Auto), 1105),
        ];
        for t in test_data {
            let got = get_image_tokens((t.1, t.2), &t.3);
            assert_eq!(got, t.4, "test case: {}", t.0)
        }
    }
}
