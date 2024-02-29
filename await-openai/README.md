# await-openai

just some code I need to deal with openai api.

> [!WARNING]
> the API is not stable

## features

### openai

types to work with openai api

only chat completion for now

compare to [async-openai](https://github.com/64bit/async-openai)

- tag enum variant
- dedicate completion chunk type

### tiktoken

tokens usage calculation for openai api.

support tools tokens usage estimation and image tokens usage calculation.

### openai_tools

blazing fast serilization compared to [openai-func-enums](https://github.com/frankfralick/openai-func-enums)

## benchmark

| Test Category | Test Performed                          | Time (ns or µs)          | Outliers           |
|---------------|-----------------------------------------|--------------------------|--------------------|
| Deserialize   | Default Request                         | 413.61 ns - 415.89 ns    |                    |
| Deserialize   | Default Request with async-openai       | 432.63 ns - 433.11 ns    | 6 (6.00%): 2 low mild, 2 high mild, 2 high severe |
| Deserialize   | Image Input Request                     | 1.1024 µs - 1.1066 µs    | 5 (5.00%): 1 low severe, 1 low mild, 2 high mild, 1 high severe |
| Serialize     | Default Request                         | 477.29 ns - 479.68 ns    |                    |
| Serialize     | With async-openai builder pattern       | 847.53 ns - 851.97 ns    | 6 (6.00%): 1 low mild, 3 high mild, 2 high severe |
| Serialize     | Function Tool                           | 2.4649 µs - 2.4893 µs    | 10 (10.00%): 9 low mild, 1 high mild |
| Serialize     | Function Tool with Func Enum            | 6.4547 µs - 6.4676 µs    | 8 (8.00%): 6 high mild, 2 high severe |
