# async-gemini

> [!WARNING]
> not stable

There are two versions of Gemini API.

- [Gemini API](https://ai.google.dev/api/rest)

The endpoint is: `generativelanguage.googleapis.com/v1beta`

- [Gemini API on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini)

The endpoint is: `{location}-aiplatform.googleapis.com/v1`

This library is primarily designed to support the first version of the Gemini API. While the Gemini API on Vertex AI has not been thoroughly tested.

Gemini API support both camelCase and snake_case as request JSON key, but this lib only support camelCase.

Gemini API support trailing comma in request JSON, but this lib do not support it.