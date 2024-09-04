# EmbeddedLLM

Run local LLMs on iGPU, APU and CPU (AMD , Intel, and Qualcomm (Coming Soon)). Easiest way to launch OpenAI API Compatible Server on Windows, Linux and MacOS

| Support matrix        | Supported now                                       | Under Development | On the roadmap |
| --------------------- | --------------------------------------------------- | ----------------- | -------------- |
| Model architectures   | Gemma <br/> Llama \* <br/> Mistral + <br/>Phi <br/> |                   |                |
| Platform              | Linux <br/> Windows                                 |                   |                |
| Architecture          | x86 <br/> x64 <br/>                                 | Arm64             |                |
| Hardware Acceleration | CUDA<br/>DirectML<br/>IpexLLM<br/>OpenVINO          | QNN <br/> ROCm    |                |

\* The Llama model architecture supports similar model families such as CodeLlama, Vicuna, Yi, and more.

\+ The Mistral model architecture supports similar model families such as Zephyr.

## 🚀 Latest News

- [2024/06] Support Phi-3 (mini, small, medium), Phi-3-Vision-Mini, Llama-2, Llama-3, Gemma (v1), Mistral v0.3, Starling-LM, Yi-1.5.
- [2024/06] Support vision/chat inference on iGPU, APU, CPU and CUDA.

## Table Content

- [Supported Models](#supported-models-quick-start)
- [Getting Started](#getting-started)
  - [Installation From Source](#installation)
  - [Launch OpenAI API Compatible Server](#launch-openai-api-compatible-server)
  - [Launch Chatbot Web UI](#launch-chatbot-web-ui)
  - [Launch Model Management UI](#launch-model-management-ui)
- [Compile OpenAI-API Compatible Server into Windows Executable](#compile-openai-api-compatible-server-into-windows-executable)
- [Prebuilt Binary (Alpha)](#compile-openai-api-compatible-server-into-windows-executable)
- [Acknowledgements](#acknowledgements)

## Supported Models (Quick Start)
  * Onnxruntime DirectML Models [Link](./docs/model/onnxruntime_directml_models.md)
  * Onnxruntime CPU Models [Link](./docs/model/onnxruntime_cpu_models.md)
  * Ipex-LLM Models [Link](./docs/model/ipex_models.md)
  * OpenVINO-LLM Models [Link](./docs/model/openvino_models.md)
  * NPU-LLM Models [Link](./docs/model/npu_models.md)

## Getting Started

### Installation

#### From Source

- **Windows**

  1. Custom Setup:

  - **IPEX(XPU)**: Requires anaconda environment. `conda create -n ellm python=3.10 libuv; conda activate ellm`.
  - **DirectML**: If you are using Conda Environment. Install additional dependencies: `conda install conda-forge::vs2015_runtime`.

  2. Install embeddedllm package. `$env:ELLM_TARGET_DEVICE='directml'; pip install -e .`. Note: currently support `cpu`, `directml` and `cuda`.

     - **DirectML:** `$env:ELLM_TARGET_DEVICE='directml'; pip install -e .[directml]`
     - **CPU:** `$env:ELLM_TARGET_DEVICE='cpu'; pip install -e .[cpu]`
     - **CUDA:** `$env:ELLM_TARGET_DEVICE='cuda'; pip install -e .[cuda]`
     - **IPEX:** `$env:ELLM_TARGET_DEVICE='ipex'; python setup.py develop`
     - **OpenVINO:** `$env:ELLM_TARGET_DEVICE='openvino'; pip install -e .[openvino]`
     - **NPU:** `$env:ELLM_TARGET_DEVICE='npu'; pip install -e .[npu]`
     - **With Web UI**:
       - **DirectML:** `$env:ELLM_TARGET_DEVICE='directml'; pip install -e .[directml,webui]`
       - **CPU:** `$env:ELLM_TARGET_DEVICE='cpu'; pip install -e .[cpu,webui]`
       - **CUDA:** `$env:ELLM_TARGET_DEVICE='cuda'; pip install -e .[cuda,webui]`
       - **IPEX:** `$env:ELLM_TARGET_DEVICE='ipex'; python setup.py develop; pip install -r requirements-webui.txt`
       - **OpenVINO:** `$env:ELLM_TARGET_DEVICE='openvino'; pip install -e .[openvino,webui]`
       - **NPU:** `$env:ELLM_TARGET_DEVICE='npu'; pip install -e .[npu,webui]`

- **Linux**

  1. Custom Setup:

  - **IPEX(XPU)**: Requires anaconda environment. `conda create -n ellm python=3.10 libuv; conda activate ellm`.
  - **DirectML**: If you are using Conda Environment. Install additional dependencies: `conda install conda-forge::vs2015_runtime`.

  2. Install embeddedllm package. `ELLM_TARGET_DEVICE='directml' pip install -e .`. Note: currently support `cpu`, `directml` and `cuda`.

     - **DirectML:** `ELLM_TARGET_DEVICE='directml' pip install -e .[directml]`
     - **CPU:** `ELLM_TARGET_DEVICE='cpu' pip install -e .[cpu]`
     - **CUDA:** `ELLM_TARGET_DEVICE='cuda' pip install -e .[cuda]`
     - **IPEX:** `ELLM_TARGET_DEVICE='ipex' python setup.py develop`
     - **OpenVINO:** `ELLM_TARGET_DEVICE='openvino' pip install -e .[openvino]`
     - **NPU:** `ELLM_TARGET_DEVICE='npu' pip install -e .[npu]`
     - **With Web UI**:
       - **DirectML:** `ELLM_TARGET_DEVICE='directml' pip install -e .[directml,webui]`
       - **CPU:** `ELLM_TARGET_DEVICE='cpu' pip install -e .[cpu,webui]`
       - **CUDA:** `ELLM_TARGET_DEVICE='cuda' pip install -e .[cuda,webui]`
       - **IPEX:** `ELLM_TARGET_DEVICE='ipex' python setup.py develop; pip install -r requirements-webui.txt`
       - **OpenVINO:** `ELLM_TARGET_DEVICE='openvino' pip install -e .[openvino,webui]`
       - **NPU:** `ELLM_TARGET_DEVICE='npu' pip install -e .[npu,webui]`

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

1.  `ellm_chatbot --port 7788 --host localhost --server_port <ellm_server_port> --server_host localhost --model_name <served_model_name>`. **Note:** To find out more of the supported arguments. `ellm_chatbot --help`.

![asset/ellm_chatbot_vid.webp](asset/ellm_chatbot_vid.webp)

### Launch Model Management UI

It is an interface that allows you to download and deploy OpenAI API compatible server. You can find out the disk space required to download the model in the UI.

1.  `ellm_modelui --port 6678`. **Note:** To find out more of the supported arguments. `ellm_modelui --help`.

![Model Management UI](asset/ellm_modelui.png)

## Compile OpenAI-API Compatible Server into Windows Executable

**NOTE:** OpenVINO packaging currently uses `torch==2.4.0`. It will not be able to run due to missing dependencies which is `libomp`. Make sure to install `libomp` and add the `libomp-xxxxxxx.dll` to `C:\\Windows\\System32`.

1. Install `embeddedllm`.
2. Install PyInstaller: `pip install pyinstaller==6.9.0`.
3. Compile Windows Executable: `pyinstaller .\ellm_api_server.spec`.
4. You can find the executable in the `dist\ellm_api_server`.
5. Use it like `ellm_server`. `.\ellm_api_server.exe --model_path <path/to/model/weight>`.

   _Powershell/Terminal Usage_:

   ```powershell
   ellm_server --model_path <path/to/model/weight>

   # DirectML
   ellm_server --model_path 'EmbeddedLLM_Phi-3-mini-4k-instruct-062024-onnx\onnx\directml\Phi-3-mini-4k-instruct-062024-int4' --port 5555

   # IPEX-LLM
   ellm_server --model_path '.\meta-llama_Meta-Llama-3.1-8B-Instruct\'  --backend 'ipex' --device 'xpu' --port 5555 --served_model_name 'meta-llama_Meta/Llama-3.1-8B-Instruct'

   # OpenVINO
   ellm_server --model_path '.\meta-llama_Meta-Llama-3.1-8B-Instruct\'  --backend 'openvino' --device 'gpu' --port 5555 --served_model_name 'meta-llama_Meta/Llama-3.1-8B-Instruct'

   # NPU
   ellm_server --model_path 'microsoft/Phi-3-mini-4k-instruct'  --backend 'npu' --device 'npu' --port 5555 --served_model_name 'microsoft/Phi-3-mini-4k-instruct'
   ```

## Prebuilt OpenAI API Compatible Windows Executable (Alpha)

You can find the prebuilt OpenAI API Compatible Windows Executable in the Release page.

_Powershell/Terminal Usage (Use it like `ellm_server`)_:

```powershell
.\ellm_api_server.exe --model_path <path/to/model/weight>

# DirectML
.\ellm_api_server.exe --model_path 'EmbeddedLLM_Phi-3-mini-4k-instruct-062024-onnx\onnx\directml\Phi-3-mini-4k-instruct-062024-int4' --port 5555

# IPEX-LLM
.\ellm_api_server.exe --model_path '.\meta-llama_Meta-Llama-3.1-8B-Instruct\'  --backend 'ipex' --device 'xpu' --port 5555 --served_model_name 'meta-llama_Meta/Llama-3.1-8B-Instruct'

# OpenVINO
.\ellm_api_server.exe --model_path '.\meta-llama_Meta-Llama-3.1-8B-Instruct\'  --backend 'openvino' --device 'gpu' --port 5555 --served_model_name 'meta-llama_Meta/Llama-3.1-8B-Instruct'

# NPU
.\ellm_api_server.exe --model_path 'microsoft/Phi-3-mini-4k-instruct'  --backend 'npu' --device 'npu' --port 5555 --served_model_name 'microsoft/Phi-3-mini-4k-instruct'
```

## Acknowledgements

- Excellent open-source projects: [vLLM](https://github.com/vllm-project/vllm.git), [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai.git), [Ipex-LLM](https://github.com/intel-analytics/ipex-llm/tree/main) and many others.
