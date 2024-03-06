use std::time::{SystemTime, UNIX_EPOCH};

use await_openai::entity::{
    chat_completion_chunk::{Choice, Chunk, ChunkResponse, DeltaMessage},
    chat_completion_object::Role as OpenaiRole,
    create_chat_completion::{Content, ContentPart, FinishReason, Message as OpenaiMessage, Stop},
};

use super::{
    request::Request, stream_response::EventData, ContentBlock, ImageSource, Message,
    MessageContent, Role, StopReason,
};

impl From<await_openai::entity::create_chat_completion::RequestBody> for Request {
    fn from(body: await_openai::entity::create_chat_completion::RequestBody) -> Self {
        let mut res = Request {
            model: body.model,
            stream: body.stream,
            temperature: body.temperature,
            top_p: body.top_p,
            ..Default::default()
        };
        let mut messages = vec![];
        let mut system_message = None;
        for message in body.messages {
            match message {
                OpenaiMessage::System(system) => {
                    system_message.replace(system.content);
                }
                OpenaiMessage::User(user) => match user.content {
                    Content::Text(text) => messages.push(Message {
                        role: Role::User,
                        content: MessageContent::Text(text),
                    }),
                    Content::Array(parts) => {
                        let mut blocks = vec![];
                        for p in parts {
                            match p {
                                ContentPart::Text(text_part) => blocks.push(ContentBlock::Text {
                                    text: text_part.text,
                                }),
                                ContentPart::Image(image_part) => {
                                    if !image_part.image_url.url.starts_with("http") {
                                        if let Some(mime) =
                                            parse_mime_from_base64(&image_part.image_url.url)
                                        {
                                            blocks.push(ContentBlock::Image {
                                                source: ImageSource::Base64 {
                                                    media_type: mime,
                                                    data: image_part.image_url.url,
                                                },
                                            })
                                        }
                                    }
                                    tracing::warn!("Image URL is not supported in Claude yet");
                                }
                            }
                        }
                        messages.push(Message {
                            role: Role::User,
                            content: MessageContent::Blocks(blocks),
                        });
                    }
                },
                OpenaiMessage::Assistant(assistant) => {
                    if let Some(text) = assistant.content {
                        messages.push(Message {
                            role: Role::Assistant,
                            content: MessageContent::Text(text),
                        })
                    }
                }
                _ => {}
            }
        }
        res.system = system_message;
        res.messages = messages;
        res.max_tokens = body.max_tokens.unwrap_or(4000);
        if let Some(stop) = body.stop {
            match stop {
                Stop::String(s) => res.stop_sequences = Some(vec![s]),
                Stop::StringArray(ss) => res.stop_sequences = Some(ss),
            }
        }
        res
    }
}

fn parse_mime_from_base64(s: &str) -> Option<String> {
    let arr: Vec<&str> = s.split(',').collect();
    if arr.len() < 2 {
        return None;
    }
    match arr[0] {
        "data:image/jpeg;base64" => Some("image/jpeg".to_string()),
        "data:image/png;base64" => Some("image/png".to_string()),
        "data:image/gif;base64" => Some("image/gif".to_string()),
        "data:image/webp;base64" => Some("image/webp".to_string()),
        _ => None,
    }
}

impl From<super::stream_response::EventData>
    for Option<await_openai::entity::chat_completion_chunk::Chunk>
{
    fn from(value: super::stream_response::EventData) -> Self {
        //TODO option?
        //TODO u32 or usize?
        //TODO how to get usage out of claude
        let mut id = String::new();
        let mut model = String::new();
        let created_at = {
            match SystemTime::now().duration_since(UNIX_EPOCH) {
                Ok(n) => n.as_secs(),
                Err(_) => 0,
            }
        };
        match value {
            EventData::Error(e) => {
                tracing::error!("Error from Claude: {}", e);
                Some(Chunk::Done)
            }
            EventData::MessageStart { message } => {
                id = message.id;
                model = message.model;
                Some(Chunk::Data(ChunkResponse {
                    id,
                    choices: get_choices(0, "", None),
                    created: created_at,
                    model,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                }))
            }
            EventData::Ping => None,
            EventData::ContentBlockStart {
                index,
                content_block,
            } => {
                let s = match content_block {
                    ContentBlock::Text { text } => text,
                    _ => "".to_string(),
                };
                Some(Chunk::Data(ChunkResponse {
                    id,
                    choices: get_choices(index as usize, &s, None),
                    created: created_at,
                    model,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                }))
            }
            EventData::ContentBlockDelta { index, delta } => {
                let s = match delta {
                    ContentBlock::TextDelta { text } => text,
                    _ => "".to_string(),
                };
                Some(Chunk::Data(ChunkResponse {
                    id,
                    choices: get_choices(index as usize, &s, None),
                    created: created_at,
                    model,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                }))
            }
            EventData::ContentBlockStop { index: _ } => None,
            EventData::MessageDelta { delta, usage: _ } => Some(Chunk::Data(ChunkResponse {
                id,
                choices: get_choices(0, "", Some(delta.stop_reason.into())),
                created: created_at,
                model,
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
            })),
            EventData::MessageStop => Some(Chunk::Done),
        }
    }
}

fn get_choices(index: usize, text: &str, finish_reason: Option<FinishReason>) -> Vec<Choice> {
    vec![Choice {
        index,
        delta: DeltaMessage {
            role: Some(OpenaiRole::Assistant),
            content: Some(text.to_string()),
            ..Default::default()
        },
        finish_reason,
        ..Default::default()
    }]
}

impl From<StopReason> for FinishReason {
    fn from(reason: StopReason) -> Self {
        match reason {
            StopReason::EndTurn => FinishReason::Stop,
            StopReason::MaxTokens => FinishReason::Length,
            StopReason::StopSequence => FinishReason::Stop,
        }
    }
}

#[cfg(test)]
mod tests {
    use await_openai::entity::create_chat_completion::RequestBody;

    use crate::messages::{
        request::Request, ContentBlock, ImageSource, Message, MessageContent, Role,
    };

    #[test]
    fn serde() {
        let tests = vec![
            (
                "default",
                r#"{"model":"gpt-3.5-turbo","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello!"}]}"#,
                Request {
                    model: "gpt-3.5-turbo".to_string(),
                    system: Some("You are a helpful assistant.".to_string()),
                    messages: vec![Message {
                        role: Role::User,
                        content: MessageContent::Text("Hello!".to_string()),
                    }],
                    max_tokens: 4000,
                    ..Default::default()
                },
            ),
            (
                "image",
                r#"{"model": "gpt-4-vision-preview","messages": [{"role": "user","content": [{"type": "text","text": "What's in this image?"},{"type": "image_url","image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALgAAAAmCAYAAAB3X1H0AAABnGlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4KPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNi4wLjAiPgogPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgeG1sbnM6ZXhpZj0iaHR0cDovL25zLmFkb2JlLmNvbS9leGlmLzEuMC8iCiAgIGV4aWY6Q29sb3JTcGFjZT0iMSIKICAgZXhpZjpQaXhlbFhEaW1lbnNpb249IjE4NCIKICAgZXhpZjpQaXhlbFlEaW1lbnNpb249IjM4Ii8+CiA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgo8P3hwYWNrZXQgZW5kPSJyIj8+WCK4LwAAAAFzUkdCAK7OHOkAAAt9SURBVHgB7ZxnqFVHEIDXHhU0auzGFnvsvYEdQRIbWMAuqKAgNuxd0SAo4g97FyUEDSgqlkTFH3YFW0zsvfeuWOK3OMc5553br+bleQbu293Z2TY7OzszezTdzp0735sAAg6kUQ5kZF2NGzdOo8sLlvU1c2DXrl0m/dfMgGDtaZ8DgYCn/T3+qlcYCPhXvf1pf/GBgKf9Pf6qVxgI+Fe9/Wl/8TaKEs0yjx8/bi5evOgizZ49u8mbN68pVqyYyZEjh6suKETmwP37983u3bvN+fPnLXGRIkVMnTp1LD8jtw4oouFA1AK+ceNGs27dupB99ujRw/Tv399kyJAhJE1Q8YkDBw4cMMOGDTPPnz//hPyYa9eunRkzZkwKfFpEvHv3zty8edMuLX/+/EmXn6SZKCtWrDCrVq1Ki3uQ9DU9fvw4pHAzWO7cuZM+Zmrt8NixY6Z169b257UQkjHnuAS8cOHCpn379qZo0aKuOaxZs8ZVDgr+HNi2bZtLc1epUsX06dPHVKtWzTZo06aNf8M0iN27d+9nXVXUJoqeRY0aNczo0aMtipQNA7Apnz17ZrDNOY2XL1+2+IoVK5pvvvnG8LJ048YN07x5c8fOfPv2rTl58qT5+++/zaNHj0z58uVN5cqVXTb9oUOHHIHgUBUvXtz2S//61GfLls3UrFnT1r18+dJgBgiApx44evSo+eeff8ydO3esD1GuXDk7ptCSvn//3pw6dcrO7eHDh6Z06dKmevXqrnk9ePDA4JsA2M/Mi7nSP0Jbu3ZtW+f9453zwoUL7dXcr18/c+nSJVOoUCHbBF6dOXPG5jNmzGjq16/vdMXaWCNQokQJ8/333xs9n1KlSpmCBQva+R08eNDky5fPNGjQwHU7xEovg0ezZ9CyznAygP+2fft26dbu17Vr16zilD12KuPMxCXgeiw2XQQcfLp06Ww1Nvvy5cttfuzYsWbr1q0GRgNlypSxAs5Vja3pPcVc0bNnzzYVKlSw9PPmzbNCQwHtNm7cOIvnxli7dq3Nyx/GYA4IxpAhQwRttmzZYtKnT2+GDx9u9uzZ4+Al07BhQzsm5Tdv3pjp06eb9evXS7VNEZJZs2YZDgTAAZAxWrVqZdfEXIHu3buHFHDWLZArVy6X3YnDLsA8mYcAh0cAnqJQAHyf3r17u+YDn7hp586dK02scGNKIviAnn809LSJds+gDScDKMGVK1ea27dvQ2ph5syZNu3Vq5cZMGDAR2xiSVwmih7y9OnTThGtJVrSQX7IwGQRbvA4E8DgwYNTCDd4Ng4B4TQDP/74o035IxqNPFrYC+KwXLhwwaniwHz33XdmyZIlvsINIQdVYM6cOSmEmzo2o2/fvo7mFHpSbicRbsoFChQg8QW0rQBrPHLkiBSTlnI4tXDTMXydOHGi7xjR0ke7Z95BvDLAurVwe+mTVY5LwE+cOGEWL15suFJ///13Zy5NmzZ18jojmgbhx8Rg8xF4rnIB2tKfhtWrV9uiaEwKf/31l8Hz5ifmgW4jgn3u3DkHzcED/vzzTweHyYEgoymwfdu2bWvr0FDal/j555+t1peGRD02bNggRSfV0ZCSJUtak8Wp9GT0gaWKQ7Njxw4PVeJF+D1w4EDnJqTHw4cPWzPRr/dI9LHsmbd/rww0a9bM9OzZ00UGH8aPH29CyZGLOMpCXAJO3Hb+/PmWWTLO5MmTTZcuXaSYIkVQPnyaaw8EMXN9/XOVchXjaLFIgd9++81qS0waDbdu3XK0O3hMB4GzZ8/arL5ZsOu9gCmATct1uGjRIse2/uOPP1ykQ4cONR07drR+g1To20hwpBwaDhHz1vaypiFfr149x6GUOkwn5sHBTRYMGjTI3oRerb1//37fISLRx7JnfgNoGahatapp0qSJiwyhJ6Lit18uwhgKcQm4X//YzIR8QgGaUsfItaOF/St1devWdXWByeF1OHDE5HEEYv25rwg4jquA3ACMI4AgY0+/fv1aUDYVs4gCGo2DggmBkydw9epVybpSDmjOnDldOL8CPsIvv/zi0qzQLViwwOL92sSDw1kHuFFYiwDOtR9Eoo9lz/z698qAH02ycXEJOM4fVwmhQgGuIB4u/EBsYF0n3jU4rYGxlTVgpyFclSpVctCYISLIILW2JBpz7949J+pCvdwA+iCAxxTp3Lmz0eYMkQsBzA5uFH44qQL6EAiOFA0eLeTJk8cKdKNGjVxNMPn27dvnwsVb0IeNW1JAzAUpSxqJPpY9kz4l9ZMBqfucaVwCTgiKq4QQ4YQJE5z5wTjNBKlgMzVwDWubleiGQKZMmSRr0xcvXthUCzjaWzubCLAIF3Xa+dSMxZGU8KYMwnw7derkzJtQZbygBSSaPrJmzWpmzJhhD5mmj0bAJUSo23nzciuC13wVnsZCH8+e6f69MqDrPmf+070b5yhyrUlzHFDvA5DUSYpAc22KmfHq1Supcgk+SMJogJgZ5Gn35MkTsvbqJZ6KgIvQawERB9MSf/jDrcMNNGLECJcdTwQEP0DfJsxx5MiR0tRJtbA4yDgzCCE3HyFBUQ6hbGTiz9B7hS3U0CgRDjjw9OlTh0xwDuJjJhx9PHvm7f+/KH9SnXGOfuXKlbhaaoHVJoJEQaRThBcQDU2eMeVw4JBg03KrCGgB93NYGBvzRM9BIjryyEJfCAXOEJpf//RtImPGkvo5knxkJYCJBWgNTFnMp2h5Lr4CCkQOD/2I0iCvIRK95lc0e6b7/q/ycQk4jhmbz8PDsmXLXHPnRS8a0ALJaxaRETZehx3RNKJRea0T0DZk2bJlLRptKyDCT1lvCqE4uS14aJBXT+h43AGkP/LY/5s2bSLrAKFJnNxEAGeUiIkIMuPoxyd57PFe64T4AO+cQs1l8+bNtooYvQbNS42PRB/rnum+/fJZsmRxoeUAu5AJFuIyUXC4tNMlc0AYtYAI3i9t2bKljaWLLd6hQwf7GKM1DXFS0WLiaHpj3z/88IPtPtSmac1PuIxPBoimYGboryNl3tRp82nSpEk2vPntt9/alz/MoGnTpjmfGvitLRIO84qICT+iG8IDaSffpGjHkLopU6YY3gb0AZY2fikCy82knWLGC/UJQST6WPfMb04aJ8pLcPhHrJkb0usrCU2saVwaPNQgRFa8pzIULa+ZhMoE2GQt3AimPL4IjfeBBLxobnl+FlpSbgAxcQSP9uehRgs3dd26dbMk2JreaBDfbNNGbHzpKxmpV7gRwK5du9quWZvXhxDh1gc33Dy0cEPHZ82ZM2cO2SQcfTx7FnKgDxW8h9SqVcshgRfwOBoH2mkUIRO1gOtIh+6TE8fHUzxu6Bi2phctrNuRJ7zHdyU6Rgv+p59+MkuXLk2B97OnJUbOePLtCn0AXuFAcLw0aBEiGVqrkf/1119T0NJnixYtHLxeI3XyHQ75cEBo0s8Rhx+Mq//xyNSpU11mFoeWyBUPVALeeQheKxBwPFjxzUooiIY+lj3T8wolA6NGjXKUlMxLTDQpJ5Km4z/+ady4cSJ9JNwW+xf7i5SND8WMhAf62AGaAkcN7Y7DFU4w0SbXr1+3NxP04bRfrPPjs4C7d+/aZiiKcLcfjzP4PWy+Fhw9JnY8T/MCfARH6BLnkYNMWFJDrPS6bTL3jC838UP4EpX1JWv/8T3issH1QpORx77WHyAlo89wfXBjiM0djo46bHYxgyLRxlqPptbaOlx7DpfX3ApHL3UISywaMVr6ZO4ZCkY+wJN5JyuN2kRJ1oBBPwEHviQHAgH/ktwOxvriHAgE/IuzPBjwS3IgVdjgX3LBaXksYvU67Bbpk4JY6f+PvEsVUZT/I+OCOad+DhBFCUyU1L9PwQwT4EAg4AkwL2ia+jkQCHjq36NghglwIBDwBJgXNE39HAgEPPXvUTDDBDhgw4Te74UT6C9oGnAgVXHgX+rCSB0jTfe/AAAAAElFTkSuQmCC"}}]}],"max_tokens": 300}"#,
                Request {
                    model: "gpt-4-vision-preview".to_string(),
                    messages: vec![Message {
                        role: Role::User,
                        content: MessageContent::Blocks(vec![
                            ContentBlock::Text {
                                text: "What's in this image?".to_string(),
                            },
                            ContentBlock::Image {
                                source: ImageSource::Base64 {
                                    media_type: "image/png".to_string(),
                                    data: "data:image/png;base64,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".to_string(),
                                },
                            },
                        ]),
                    }],
                    max_tokens: 300,
                    ..Default::default()
                },
            ),
        ];
        for (name, json, want) in tests {
            //test deserialize
            let parsed: RequestBody = serde_json::from_str(json).unwrap();
            let got: Request = parsed.into();
            assert_eq!(got, want, "deserialize test failed: {}", name);
        }
    }
}
