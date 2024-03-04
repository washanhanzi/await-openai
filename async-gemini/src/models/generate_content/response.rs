use serde::{Deserialize, Serialize};

use super::{Content, HarmCategory};

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    /// Candidate responses from the model.
    candidates: Vec<Candidate>,
    prompt_feedback: Option<PromptFeedback>,
}

/// A response candidate generated from the model.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    content: Content,
    #[serde(default)]
    finish_reason: Option<FinishReason>,
    /// List of ratings for the safety of a response candidate.
    /// There is at most one rating per category.
    safety_ratings: Vec<SafetyRating>,
    /// Citation information for model-generated candidate.
    /// This field may be populated with recitation information for any text included in the content. These are passages that are "recited" from copyrighted material in the foundational LLM's training data.
    citation_metadata: Option<CitationMetadata>,
    index: u32,
}

/// Defines the reason why the model stopped generating tokens.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Default)]
pub enum FinishReason {
    /// Default value. This value is unused.
    #[default]
    #[serde(rename = "FINISH_REASON_UNSPECIFIED")]
    Unspecified,
    /// Natural stop point of the model or provided stop sequence.
    Stop,
    /// The maximum number of tokens as specified in the request was reached.
    MaxTokens,
    /// The candidate content was flagged for safety reasons.
    Safety,
    /// The candidate content was flagged for recitation reasons.
    Recitation,
    /// Unknown reason.
    Other,
}

/// Safety rating for a piece of content.
/// The safety rating contains the category of harm and the harm probability level in that category for a piece of content. Content is classified for safety across a number of harm categories and the probability of the harm classification is included here.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    /// The category for this rating.
    category: HarmCategory,
    /// The probability of harm for this content.
    probability: HarmProbability,
    /// Was this content blocked because of this rating?
    blocked: Option<bool>,
}

/// The probability that a piece of content is harmful.
/// The classification system gives the probability of the content being unsafe. This does not indicate the severity of harm for a piece of content.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmProbability {
    /// Probability is unspecified.
    #[serde(rename = "HARM_PROBABILITY_UNSPECIFIED")]
    Unspecified,
    /// Content has a negligible chance of being unsafe.
    Negligible,
    /// Content has a low chance of being unsafe.
    Low,
    /// Content has a medium chance of being unsafe.
    Medium,
    /// Content has a high chance of being unsafe.
    High,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CitationMetadata {
    /// Citations to sources for a specific response.
    citation_sources: Vec<CitationSource>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CitationSource {
    start_index: Option<u32>,
    end_index: Option<u32>,
    uri: Option<String>,
    title: Option<String>,
    license: Option<String>,
    /// The date a citation was published. Its valid formats are YYYY, YYYY-MM, and YYYY-MM-DD.
    publication_date: Option<PublicationDate>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct PublicationDate {
    year: Option<u32>,
    month: Option<u32>,
    day: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    block_reason: Option<BlockReason>,
    safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BlockReason {
    #[serde(rename = "BLOCK_REASON_UNSPECIFIED")]
    Unspecified,
    Safety,
    Other,
}

#[cfg(test)]
mod tests {
    use crate::models::{Part, Role};

    use super::*;

    #[test]
    fn serde() {
        let tests = vec![
            (
            "text-only",
            r#"{
                "candidates": [
                  {
                    "content": {
                      "parts": [
                        {
                          "text": "Once upon a time, in a small town nestled at the foot of towering mountains, there lived a young girl named Lily. Lily was an adventurous and imaginative child, always dreaming of exploring the world beyond her home. One day, while wandering through the attic of her grandmother's house, she stumbled upon a dusty old backpack tucked away in a forgotten corner. Intrigued, Lily opened the backpack and discovered that it was an enchanted one. Little did she know that this magical backpack would change her life forever.\n\nAs Lily touched the backpack, it shimmered with an otherworldly light. She reached inside and pulled out a map that seemed to shift and change before her eyes, revealing hidden paths and distant lands. Curiosity tugged at her heart, and without hesitation, Lily shouldered the backpack and embarked on her first adventure.\n\nWith each step she took, the backpack adjusted to her needs. When the path grew treacherous, the backpack transformed into sturdy hiking boots, providing her with the confidence to navigate rocky terrains. When a sudden rainstorm poured down, the backpack transformed into a cozy shelter, shielding her from the elements.\n\nAs days turned into weeks, Lily's journey took her through lush forests, across treacherous rivers, and to the summits of towering mountains. The backpack became her loyal companion, guiding her along the way, offering comfort, protection, and inspiration.\n\nAmong her many adventures, Lily encountered a lost fawn that she gently carried in the backpack's transformed cradle. She helped a friendly giant navigate a dense fog by using the backpack's built-in compass. And when faced with a raging river, the backpack magically transformed into a sturdy raft, transporting her safely to the other side.\n\nThrough her travels, Lily discovered the true power of the magic backpack. It wasn't just a magical object but a reflection of her own boundless imagination and tenacity. She realized that the world was hers to explore, and the backpack was a tool to help her reach her full potential.\n\nAs Lily returned home, enriched by her adventures and brimming with stories, she decided to share the magic of the backpack with others. She organized a special adventure club, where children could embark on their own extraordinary journeys using the backpack's transformative powers. Together, they explored hidden worlds, learned valuable lessons, and formed lifelong friendships.\n\nAnd so, the legend of the magic backpack lived on, passed down from generation to generation. It became a reminder that even the simplest objects can hold extraordinary power when combined with imagination, courage, and a sprinkle of magic."
                        }
                      ],
                      "role": "model"
                    },
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [
                      {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE"
                      },
                      {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE"
                      },
                      {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE"
                      },
                      {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE"
                      }
                    ]
                  }
                ],
                "promptFeedback": {
                  "safetyRatings": [
                    {
                      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                      "probability": "NEGLIGIBLE"
                    },
                    {
                      "category": "HARM_CATEGORY_HATE_SPEECH",
                      "probability": "NEGLIGIBLE"
                    },
                    {
                      "category": "HARM_CATEGORY_HARASSMENT",
                      "probability": "NEGLIGIBLE"
                    },
                    {
                      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                      "probability": "NEGLIGIBLE"
                    }
                  ]
                }
              }"#,
            GenerateContentResponse {
                candidates: vec![
                    Candidate {
                        content:  Content {
                                parts: vec![
                                    Part::Text("Once upon a time, in a small town nestled at the foot of towering mountains, there lived a young girl named Lily. Lily was an adventurous and imaginative child, always dreaming of exploring the world beyond her home. One day, while wandering through the attic of her grandmother's house, she stumbled upon a dusty old backpack tucked away in a forgotten corner. Intrigued, Lily opened the backpack and discovered that it was an enchanted one. Little did she know that this magical backpack would change her life forever.\n\nAs Lily touched the backpack, it shimmered with an otherworldly light. She reached inside and pulled out a map that seemed to shift and change before her eyes, revealing hidden paths and distant lands. Curiosity tugged at her heart, and without hesitation, Lily shouldered the backpack and embarked on her first adventure.\n\nWith each step she took, the backpack adjusted to her needs. When the path grew treacherous, the backpack transformed into sturdy hiking boots, providing her with the confidence to navigate rocky terrains. When a sudden rainstorm poured down, the backpack transformed into a cozy shelter, shielding her from the elements.\n\nAs days turned into weeks, Lily's journey took her through lush forests, across treacherous rivers, and to the summits of towering mountains. The backpack became her loyal companion, guiding her along the way, offering comfort, protection, and inspiration.\n\nAmong her many adventures, Lily encountered a lost fawn that she gently carried in the backpack's transformed cradle. She helped a friendly giant navigate a dense fog by using the backpack's built-in compass. And when faced with a raging river, the backpack magically transformed into a sturdy raft, transporting her safely to the other side.\n\nThrough her travels, Lily discovered the true power of the magic backpack. It wasn't just a magical object but a reflection of her own boundless imagination and tenacity. She realized that the world was hers to explore, and the backpack was a tool to help her reach her full potential.\n\nAs Lily returned home, enriched by her adventures and brimming with stories, she decided to share the magic of the backpack with others. She organized a special adventure club, where children could embark on their own extraordinary journeys using the backpack's transformative powers. Together, they explored hidden worlds, learned valuable lessons, and formed lifelong friendships.\n\nAnd so, the legend of the magic backpack lived on, passed down from generation to generation. It became a reminder that even the simplest objects can hold extraordinary power when combined with imagination, courage, and a sprinkle of magic.".to_string()),
                                ],
                                role: Role::Model,
                            },
                        finish_reason: Some(FinishReason::Stop),
                        index:0,
                        safety_ratings:vec![
                            SafetyRating{
                            category:HarmCategory::SexuallyExplicit,
                            probability:HarmProbability::Negligible,
                            blocked:None,
                            },
                            SafetyRating{
                            category:HarmCategory::HateSpeech,
                            probability:HarmProbability::Negligible,
                            blocked:None,
                            },
                            SafetyRating{
                            category:HarmCategory::Harassment,
                            probability:HarmProbability::Negligible,
                            blocked:None,
                            },
                            SafetyRating{
                            category:HarmCategory::DangerousContent,
                            probability:HarmProbability::Negligible,
                            blocked:None,
                            },
                            ],
                        ..Default::default()
                    },
                ],
                prompt_feedback:Some(
                    PromptFeedback{
                    block_reason:None,
                    safety_ratings:Some(vec![
                        SafetyRating{
                        category:HarmCategory::SexuallyExplicit,
                        probability:HarmProbability::Negligible,
                        blocked:None,
                    },
                        SafetyRating{
                        category:HarmCategory::HateSpeech,
                        probability:HarmProbability::Negligible,
                        blocked:None,
                    },
                        SafetyRating{
                        category:HarmCategory::Harassment,
                        probability:HarmProbability::Negligible,
                        blocked:None,
                    },
                        SafetyRating{
                        category:HarmCategory::DangerousContent,
                        probability:HarmProbability::Negligible,
                        blocked:None,
                    },
                    ]),
                })
            },
        ),
        (
            "sse",
            r#"{"candidates": [{"content": {"parts": [{"text": "I do not have real-time capabilities and my knowledge cutoff is April 2"}],"role": "model"},"finishReason": "STOP","index": 0,"safetyRatings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT","probability": "NEGLIGIBLE"},{"category": "HARM_CATEGORY_HATE_SPEECH","probability": "NEGLIGIBLE"},{"category": "HARM_CATEGORY_HARASSMENT","probability": "NEGLIGIBLE"},{"category": "HARM_CATEGORY_DANGEROUS_CONTENT","probability": "NEGLIGIBLE"}]}],"promptFeedback": {"safetyRatings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT","probability": "NEGLIGIBLE"},{"category": "HARM_CATEGORY_HATE_SPEECH","probability": "NEGLIGIBLE"},{"category": "HARM_CATEGORY_HARASSMENT","probability": "NEGLIGIBLE"},{"category": "HARM_CATEGORY_DANGEROUS_CONTENT","probability": "NEGLIGIBLE"}]}}"#,
            GenerateContentResponse{
                candidates:vec![
                    Candidate{
                        content:Content{
                            parts:vec![
                                Part::Text("I do not have real-time capabilities and my knowledge cutoff is April 2".to_string())
                            ],
                            role:Role::Model,
                        },
                        finish_reason:Some(FinishReason::Stop),
                        index:0,
                        safety_ratings:vec![
                            SafetyRating{
                                category:HarmCategory::SexuallyExplicit,
                                probability:HarmProbability::Negligible,
                                blocked:None,
                            },
                            SafetyRating{
                                category:HarmCategory::HateSpeech,
                                probability:HarmProbability::Negligible,
                                blocked:None,
                            },
                            SafetyRating{
                                category:HarmCategory::Harassment,
                                probability:HarmProbability::Negligible,
                                blocked:None,
                            },
                            SafetyRating{
                                category:HarmCategory::DangerousContent,
                                probability:HarmProbability::Negligible,
                                blocked:None,
                            },
                        ],
                        ..Default::default()
                    }
                ],
                prompt_feedback:Some(PromptFeedback{
                    block_reason:None,
                    safety_ratings:Some(vec![
                        SafetyRating{
                            category:HarmCategory::SexuallyExplicit,
                            probability:HarmProbability::Negligible,
                            blocked:None,
                        },
                        SafetyRating{
                            category:HarmCategory::HateSpeech,
                            probability:HarmProbability::Negligible,
                            blocked:None,
                        },
                        SafetyRating{
                            category:HarmCategory::Harassment,
                            probability:HarmProbability::Negligible,
                            blocked:None,
                        },
                        SafetyRating{
                            category:HarmCategory::DangerousContent,
                            probability:HarmProbability::Negligible,
                            blocked:None,
                        },
                    ]),
                })
            }
        ),
        ];
        for (name, json, expected) in tests {
            //test deserialize
            let actual: GenerateContentResponse = serde_json::from_str(json).unwrap();
            assert_eq!(actual, expected, "deserialize test failed: {}", name);
            //test serialize
            let serialized = serde_json::to_string(&expected).unwrap();
            let actual: GenerateContentResponse = serde_json::from_str(&serialized).unwrap();
            assert_eq!(actual, expected, "serialize test failed: {}", name);
        }
    }
}
