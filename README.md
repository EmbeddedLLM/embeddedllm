# EmbeddedLLM

Run local LLMs on iGPU and APU (AMD , Intel, and Qualcomm (Coming Soon))

|Support matrix| Supported now| Under Development | On the roadmap|
|--------------|--------------|-------------------|---------------|
|Model architectures|  Gemma <br/> Llama * <br/> Mistral + <br/>Phi <br/>||
|Platform| Linux <br/> Windows  |  ||||
|Architecture|x86 <br/> x64 <br/> | Arm64 |||
|Hardware Acceleration|CUDA<br/>DirectML<br/>|QNN <br/> ROCm |OpenVINO

\* The Llama model architecture supports similar model families such as CodeLlama, Vicuna, Yi, and more.

\+ The Mistral model architecture supports similar model families such as Zephyr.



## 🚀 Latest News
- [2024/06] Support chat inference on iGPU and CPU.


## Supported Models (Quick Start)
| Models              | Parameters | Context Length | Link |
|---------------------|------------|----------------|------|
|Gemma-2b-Instruct v1 |       2B   | 8192           | [EmbeddedLLM/gemma-2b-it-onnx](https://huggingface.co/EmbeddedLLM/gemma-2b-it-onnx) |
|Llama-2-7b-chat      | 7B         |    4096        | [EmbeddedLLM/llama-2-7b-chat-int4-onnx-directml](https://huggingface.co/EmbeddedLLM/llama-2-7b-chat-int4-onnx-directml) |
|Llama-2-13b-chat      | 13B         |    4096        | [EmbeddedLLM/llama-2-13b-chat-int4-onnx-directml](https://huggingface.co/EmbeddedLLM/llama-2-13b-chat-int4-onnx-directml) |
|Llama-3-8b-chat      | 8B           | 8192           | [EmbeddedLLM/mistral-7b-instruct-v0.3-onnx](https://huggingface.co/EmbeddedLLM/mistral-7b-instruct-v0.3-onnx) |
|Mistral-7b-v0.3-instruct| 7B        |   32768        | [EmbeddedLLM/mistral-7b-instruct-v0.3-onnx](https://huggingface.co/EmbeddedLLM/mistral-7b-instruct-v0.3-onnx)|
| Phi3-mini-4k-instruct | 3.8B      |   4096        | [microsoft/Phi-3-mini-4k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) |
| Phi3-mini-128k-instruct | 3.8B |  128k | [microsoft/Phi-3-mini-128k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx) |
| Phi3-medium-4k-instruct | 17B | 4096 | [microsoft/Phi-3-medium-4k-instruct-onnx-directml](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-directml)|
| Phi3-medium-128k-instruct | 17B | 128k | [microsoft/Phi-3-medium-128k-instruct-onnx-directml](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-directml)|


## Acknowledgements
* Excellent open-source projects: [vLLM](https://github.com/vllm-project/vllm.git), [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai.git) and many others.

* Thanks to all the [contributors](./docs/contributors.md).