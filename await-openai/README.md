# await-openai

just some types I need to serialize/deserialize openai api

## compare to [async-openai](https://github.com/64bit/async-openai)

1. no builder

2. only types

3. refactor type names to mod name

 for example:
 - async_openai::types::ChatCompletionRequestAssistantMessageArgs
 - await_openai::entity::create_chat_completion::SystemMessage

4. tag enum variant

5. dedicate completion chunk type

6. token calculation with `tiktoken` feature

7. function tool with `tools` feature, much more faster serilization than `openai-func-enums`

## benchmark

| Test Category | Test Performed                          | Time (ns or µs)          | Outliers           |
|---------------|-----------------------------------------|--------------------------|--------------------|
| Deserialize   | Default Request                         | 413.61 ns - 415.89 ns    |                    |
| Deserialize   | Default Request with async-openai       | 432.63 ns - 433.11 ns    | 6 (6.00%): 2 low mild, 2 high mild, 2 high severe |
| Deserialize   | Image Input Request                     | 1.1024 µs - 1.1066 µs    | 5 (5.00%): 1 low severe, 1 low mild, 2 high mild, 1 high severe |
| Serialize     | Default Request                         | 240.61 ns - 241.19 ns    |                    |
| Serialize     | Default Request with async-openai       | 847.53 ns - 851.97 ns    | 6 (6.00%): 1 low mild, 3 high mild, 2 high severe |
| Serialize     | Function Tool                           | 916.96 ns - 924.43 ns    | 6 (6.00%): 4 high mild, 2 high severe |
| Serialize     | Function Tool with Func Enum            | 6.4547 µs - 6.4676 µs    | 8 (8.00%): 6 high mild, 2 high severe |
