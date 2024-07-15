# EmbeddedLLM

Run local LLMs on iGPU, APU and CPU (AMD , Intel, and Qualcomm (Coming Soon)). Easiest way to launch OpenAI API Compatible Server on Windows, Linux and MacOS

| Support matrix        | Supported now                                       | Under Development | On the roadmap |
| --------------------- | --------------------------------------------------- | ----------------- | -------------- |
| Model architectures   | Gemma <br/> Llama \* <br/> Mistral + <br/>Phi <br/> |                   |                |
| Platform              | Linux <br/> Windows                                 |                   |                |
| Architecture          | x86 <br/> x64 <br/>                                 | Arm64             |                |
| Hardware Acceleration | CUDA<br/>DirectML<br/>IpexLLM                       | QNN <br/> ROCm    | OpenVINO       |

\* The Llama model architecture supports similar model families such as CodeLlama, Vicuna, Yi, and more.

\+ The Mistral model architecture supports similar model families such as Zephyr.

## 🚀 Latest News

- [2024/06] Support Phi-3 (mini, small, medium), Phi-3-Vision-Mini, Llama-2, Llama-3, Gemma (v1), Mistral v0.3, Starling-LM, Yi-1.5.
- [2024/06] Support vision/chat inference on iGPU, APU, CPU and CUDA.

## Table Content

- [Supported Models](#supported-models-quick-start)
  - [Onnxruntime Models](./docs/model/onnxruntime_models.md)
  - [Ipex-LLM Models](./docs/model/ipex_models.md)
- [Getting Started](#getting-started)
  - [Installation From Source](#installation)
  - [Launch OpenAI API Compatible Server](#launch-openai-api-compatible-server)
  - [Launch Chatbot Web UI](#launch-chatbot-web-ui)
  - [Launch Model Management UI](#launch-model-management-ui)
- [Compile OpenAI-API Compatible Server into Windows Executable](#compile-openai-api-compatible-server-into-windows-executable)
- [Acknowledgements](#acknowledgements)

## Supported Models (Quick Start)

| Models | Parameters | Context Length | Link |
| --- | --- | --- | --- |
| Gemma-2b-Instruct v1 | 2B | 8192 | [EmbeddedLLM/gemma-2b-it-onnx](https://huggingface.co/EmbeddedLLM/gemma-2b-it-onnx) |
| Llama-2-7b-chat | 7B | 4096 | [EmbeddedLLM/llama-2-7b-chat-int4-onnx-directml](https://huggingface.co/EmbeddedLLM/llama-2-7b-chat-int4-onnx-directml) |
| Llama-2-13b-chat | 13B | 4096 | [EmbeddedLLM/llama-2-13b-chat-int4-onnx-directml](https://huggingface.co/EmbeddedLLM/llama-2-13b-chat-int4-onnx-directml) |
| Llama-3-8b-chat | 8B | 8192 | [EmbeddedLLM/mistral-7b-instruct-v0.3-onnx](https://huggingface.co/EmbeddedLLM/mistral-7b-instruct-v0.3-onnx) |
| Mistral-7b-v0.3-instruct | 7B | 32768 | [EmbeddedLLM/mistral-7b-instruct-v0.3-onnx](https://huggingface.co/EmbeddedLLM/mistral-7b-instruct-v0.3-onnx) |
| Phi-3-mini-4k-instruct-062024 | 3.8B | 4096 | [EmbeddedLLM/Phi-3-mini-4k-instruct-062024-onnx](https://huggingface.co/EmbeddedLLM/Phi-3-mini-4k-instruct-062024-onnx/tree/main/onnx/directml/Phi-3-mini-4k-instruct-062024-int4) |
| Phi3-mini-4k-instruct | 3.8B | 4096 | [microsoft/Phi-3-mini-4k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) |
| Phi3-mini-128k-instruct | 3.8B | 128k | [microsoft/Phi-3-mini-128k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx) |
| Phi3-medium-4k-instruct | 17B | 4096 | [microsoft/Phi-3-medium-4k-instruct-onnx-directml](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-directml) |
| Phi3-medium-128k-instruct | 17B | 128k | [microsoft/Phi-3-medium-128k-instruct-onnx-directml](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-directml) |
| Openchat-3.6-8b | 8B | 8192 | [EmbeddedLLM/openchat-3.6-8b-20240522-onnx](https://huggingface.co/EmbeddedLLM/openchat-3.6-8b-20240522-onnx) |
| Yi-1.5-6b-chat | 6B | 32k | [EmbeddedLLM/01-ai_Yi-1.5-6B-Chat-onnx](https://huggingface.co/EmbeddedLLM/01-ai_Yi-1.5-6B-Chat-onnx) |
| Phi-3-vision-128k-instruct |  | 128k | [EmbeddedLLM/Phi-3-vision-128k-instruct-onnx](https://huggingface.co/EmbeddedLLM/Phi-3-vision-128k-instruct-onnx/tree/main/onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4) |

## Getting Started

### Installation

#### Prerequisites
  1. **Install Git**
     - Download and install Git from [git-scm.com](https://git-scm.com/).
    
  2. **Install Miniconda/Anaconda**
   - Download and install Miniconda or Anaconda from [docs.anaconda.com](https://docs.anaconda.com/anaconda/install/).

#### Model Download

1. **Install Huggingface Hub CLI**
   - Install the CLI:
     ```sh
     pip install huggingface-hub[cli]
     ```

2. **Download Model using Huggingface CLI**
   - Download the model:
     ```sh
     huggingface-cli download EmbeddedLLM/<MODEL>-onnx --include="onnx/directml/*" --local-dir .\<MODEL>
     ```
     
For more information on steps to download the model, kindly refer to [Supported Models](#supported-models-quick-start).


#### From Source
  1. **Clone the EmbeddedLLM Repository**
     1. **Open Git Bash**
   - Navigate to your `Documents` folder:
     ```sh
     cd ~/Documents
     ```

  2. **Initialize Git and Clone the Repository**
   - Initialize a new Git repository:
     ```sh
     git init
     ```
   - Clone the EmbeddedLLM repository:
     ```sh
     git clone https://github.com/EmbeddedLLM/embeddedllm.git
     ```

#### Setup Conda Environment

  1. **Custom Setup**

| Step | Windows | Linux |
|------|---------|-------|
| **Create and Activate<br>Conda Environment** | `conda create -n ellm python=3.10 libuv`<br>`conda activate ellm` | `conda create -n ellm python=3.10 libuv`<br>`conda activate ellm` |
| **Additional Dependencies<br>for DirectML** | `conda install conda-forge::vs2015_runtime` | `conda install conda-forge::vs2015_runtime` |


  2. **Install EmbeddedLLM Package** (Set Target Device and Install Package)

  Without Web UI

| Step | Windows | Linux |
|------|---------|-------|
| **DirectML** | `$env:ELLM_TARGET_DEVICE='directml'; pip install -e .[directml]` | `ELLM_TARGET_DEVICE='directml' pip install -e .[directml]` |
| **CPU** | `$env:ELLM_TARGET_DEVICE='cpu'; pip install -e .[cpu]` | `ELLM_TARGET_DEVICE='cpu' pip install -e .[cpu]` |
| **CUDA** | `$env:ELLM_TARGET_DEVICE='cuda'; pip install -e .[cuda]` | `ELLM_TARGET_DEVICE='cuda' pip install -e .[cuda]` |
| **XPU** | `$env:ELLM_TARGET_DEVICE='xpu'; pip install -e .[xpu]` | `ELLM_TARGET_DEVICE='xpu' pip install -e .[xpu]` |

  With Web UI

| Step | Windows | Linux |
|------|---------|-------|
| **DirectML** | `$env:ELLM_TARGET_DEVICE='directml'; pip install -e .[directml,webui]` | `ELLM_TARGET_DEVICE='directml' pip install -e .[directml,webui]` |
| **CPU** | `$env:ELLM_TARGET_DEVICE='cpu'; pip install -e .[cpu,webui]` | `ELLM_TARGET_DEVICE='cpu' pip install -e .[cpu,webui]` |
| **CUDA** | `$env:ELLM_TARGET_DEVICE='cuda'; pip install -e .[cuda,webui]` | `ELLM_TARGET_DEVICE='cuda' pip install -e .[cuda,webui]` |
| **XPU** | `$env:ELLM_TARGET_DEVICE='xpu'; pip install -e .[xpu,webui]` | `ELLM_TARGET_DEVICE='xpu' pip install -e .[xpu,webui]` |
  

### Launch OpenAI API Compatible Server

1. Custom Setup:

   - **Ipex**

     - For **Intel iGPU**:

       ```cmd
       set SYCL_CACHE_PERSISTENT=1
       set BIGDL_LLM_XMX_DISABLED=1
       ```

     - For **Intel Arc™ A-Series Graphics**:
       ```cmd
       set SYCL_CACHE_PERSISTENT=1
       ```

2. `ellm_server --model_path <path/to/model/weight>`.
3. Example code to connect to the api server can be found in `scripts/python`. **Note:** To find out more of the supported arguments. `ellm_server --help`.

### Launch Chatbot Web UI

1.  `ellm_chatbot`. **Note:** To find out more of the supported arguments. `ellm_chatbot --help`.

   ![asset/ellm_chatbot_vid.webp](asset/ellm_chatbot_vid.webp)

### Launch Model Management UI

It is an interface that allows you to download and deploy OpenAI API compatible server. You can find out the disk space required to download the model in the UI.

1.  `ellm_modelui --port 6678`. **Note:** To find out more of the supported arguments. `ellm_modelui --help`.

   ![Model Management UI](asset/ellm_modelui.png)

## Compile OpenAI-API Compatible Server into Windows Executable

1. Install `embeddedllm`.
2. Install PyInstaller: `pip install pyinstaller`.
3. Compile Windows Executable: `pyinstaller .\ellm_api_server.spec`.
4. You can find the executable in the `dist\ellm_api_server`.

## Acknowledgements

- Excellent open-source projects: [vLLM](https://github.com/vllm-project/vllm.git), [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai.git), [Ipex-LLM](https://github.com/intel-analytics/ipex-llm/tree/main) and many others.
