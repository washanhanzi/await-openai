# await-openai

just some types I need to serialize/deserialize openai api

## compare to [async-openai](https://github.com/64bit/async-openai)

1. no builder

2. no request, only types

3. move type names to mod name

 for example:
 - async_openai::types::ChatCompletionRequestAssistantMessageArgs
 - await_openai::entity::create_chat_completion::SystemMessage

4. tag enum variant

5. dedicate completion chunk type

6. token calculation with `tiktoken` feature