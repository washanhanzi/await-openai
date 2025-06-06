# about this workspace

ease your way to work with openai, gemini and anthropic api

> [!WARNING]
> wip, not stable.

- [await-openai](https://github.com/washanhanzi/await-openai/tree/main/await-openai)
- [async-gemini](https://github.com/washanhanzi/await-openai/tree/main/async-gemini)
- [async-claude](https://github.com/washanhanzi/await-openai/tree/main/async-claude)

# await-openai

compare to [async-openai](https://github.com/64bit/async-openai)

- dedicate completion chunk type
- only types

## features

### tool

blazing fast serilization compared to [openai-func-enums](https://github.com/frankfralick/openai-func-enums)

### claude

transform openai's request to anthropic's request.

transform anthropic's response to openai's response.

### rmcp

bidirectional convert between openai's tool and rmcp's tool.

## benchmark

| Test Category | Test Performed                          | Time (ns or µs)          | Outliers           |
|---------------|-----------------------------------------|--------------------------|--------------------|
| Deserialize   | Default Request                         | 413.61 ns - 415.89 ns    |                    |
| Deserialize   | Default Request with async-openai       | 432.63 ns - 433.11 ns    | 6 (6.00%): 2 low mild, 2 high mild, 2 high severe |
| Deserialize   | Image Input Request                     | 1.1024 µs - 1.1066 µs    | 5 (5.00%): 1 low severe, 1 low mild, 2 high mild, 1 high severe |
| Serialize     | Default Request                         | 477.29 ns - 479.68 ns    |                    |
| Serialize     | With async-openai builder pattern       | 847.53 ns - 851.97 ns    | 6 (6.00%): 1 low mild, 3 high mild, 2 high severe |
| Serialize     | Function Tool                           | 910.31 ns - 912.73 ns    | 13 (13.00%): 5 high mild, 8 high severe |
| Serialize     | Function Tool with Func Enum            | 6.4547 µs - 6.4676 µs    | 8 (8.00%): 6 high mild, 2 high severe |

# async-gemini

- [async-gemini](https://github.com/washanhanzi/await-openai/tree/main/async-gemini)

# async-claude

- [async-claude](https://github.com/washanhanzi/await-openai/tree/main/async-claude)