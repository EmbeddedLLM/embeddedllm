import importlib.metadata
import os
import re
import subprocess
import time
from typing import Optional

import gradio as gr
import httpx
import pandas as pd
import requests
from huggingface_hub import HfApi, snapshot_download
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_embeddedllm_backend():
    try:
        # Get the version of embeddedllm
        version = importlib.metadata.version("embeddedllm")

        # Use regex to extract the backend
        match = re.search(r"\+(directml|cpu|cuda|ipex)$", version)

        if match:
            backend = match.group(1)
            return backend
        else:
            return "Unknown backend"

    except importlib.metadata.PackageNotFoundError:
        return "embeddedllm not installed"


backend = get_embeddedllm_backend()


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    port: int = Field(default=6678, description="Gradio port.")
    host: str = Field(default="127.0.0.1", description="Gradio host.")


config = Config()


class DeployedModel(BaseModel):
    process: subprocess.Popen | None = None
    model_name: str = ""

    class Config:
        arbitrary_types_allowed = True


deployed_model: DeployedModel = DeployedModel()


class ModelCard(BaseModel):
    repo_id: str
    hf_url: str
    model_name: str
    subfolder: str
    repo_type: str
    context_length: int
    size: Optional[int] = 0


ipex_model_dict_list = {
    "microsoft/Phi-3-mini-4k-instruct": ModelCard(
        hf_url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/tree/main/",
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        model_name="Phi-3-mini-4k-instruct",
        subfolder=".",
        repo_type="model",
        context_length=4096,
    ),
    "microsoft/Phi-3-mini-128k-instruct": ModelCard(
        hf_url="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/tree/main",
        repo_id="microsoft/Phi-3-mini-128k-instruct",
        model_name="Phi-3-mini-128k-instruct",
        subfolder=".",
        repo_type="model",
        context_length=131072,
    ),
    "microsoft/Phi-3-medium-4k-instruct": ModelCard(
        hf_url="https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/tree/main",
        repo_id="microsoft/Phi-3-medium-4k-instruct",
        model_name="Phi-3-medium-4k-instruct",
        subfolder=".",
        repo_type="model",
        context_length=4096,
    ),
    "microsoft/Phi-3-medium-128k-instruct": ModelCard(
        hf_url="https://huggingface.co/microsoft/Phi-3-medium-128k-instruct/tree/main",
        repo_id="microsoft/Phi-3-medium-128k-instruct",
        model_name="Phi-3-medium-128k-instruct",
        subfolder=".",
        repo_type="model",
        context_length=131072,
    ),
}

dml_model_dict_list = {
    "EmbeddedLLM/Phi-3-mini-4k-instruct-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/Phi-3-mini-4k-instruct-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/Phi-3-mini-4k-instruct-onnx-directml",
        model_name="Phi-3-mini-4k-instruct-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=4096,
    ),
    "EmbeddedLLM/Phi-3-mini-128k-instruct-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/Phi-3-mini-128k-instruct-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/Phi-3-mini-128k-instruct-onnx-directml",
        model_name="Phi-3-mini-128k-instruct-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=131072,
    ),
    "EmbeddedLLM/Phi-3-medium-4k-instruct-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/Phi-3-medium-4k-instruct-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/Phi-3-medium-4k-instruct-onnx-directml",
        model_name="Phi-3-medium-4k-instruct-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=4096,
    ),
    "EmbeddedLLM/Phi-3-medium-128k-instruct-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/Phi-3-medium-128k-instruct-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/Phi-3-medium-128k-instruct-onnx-directml",
        model_name="Phi-3-medium-128k-instruct-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=131072,
    ),
    "EmbeddedLLM/Phi-3-mini-4k-instruct-062024-int4-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/Phi-3-mini-4k-instruct-062024-int4-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/Phi-3-mini-4k-instruct-062024-int4-onnx-directml",
        model_name="Phi-3-mini-4k-instruct-062024-int4-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=4096,
    ),
    "EmbeddedLLM/mistralai_Mistral-7B-Instruct-v0.3-int4-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/mistralai_Mistral-7B-Instruct-v0.3-int4-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/mistralai_Mistral-7B-Instruct-v0.3-int4-onnx-directml",
        model_name="mistralai_Mistral-7B-Instruct-v0.3-int4-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=32768,
    ),
    "EmbeddedLLM/gemma-2b-it-int4-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/gemma-2b-it-int4-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/gemma-2b-it-int4-onnx-directml",
        model_name="gemma-2b-it-int4-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=8192,
    ),
    "EmbeddedLLM/gemma-7b-it-int4-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/gemma-7b-it-int4-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/gemma-7b-it-int4-onnx-directml",
        model_name="gemma-7b-it-int4-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=8192,
    ),
    "EmbeddedLLM/llama-2-7b-chat-int4-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/llama-2-7b-chat-int4-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/llama-2-7b-chat-int4-onnx-directml",
        model_name="llama-2-7b-chat-int4-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=4096,
    ),
    "EmbeddedLLM/Starling-LM-7b-beta-int4-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/Starling-LM-7b-beta-int4-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/Starling-LM-7b-beta-int4-onnx-directml",
        model_name="Starling-LM-7b-beta-int4-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=8192,
    ),
    "EmbeddedLLM/openchat-3.6-8b-20240522-int4-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/openchat-3.6-8b-20240522-int4-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/openchat-3.6-8b-20240522-int4-onnx-directml",
        model_name="openchat-3.6-8b-20240522-int4-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=8192,
    ),
    "EmbeddedLLM/01-ai_Yi-1.5-6B-Chat-int4-onnx-directml": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/01-ai_Yi-1.5-6B-Chat-int4-onnx-directml/tree/main",
        repo_id="EmbeddedLLM/01-ai_Yi-1.5-6B-Chat-int4-onnx-directml",
        model_name="01-ai_Yi-1.5-6B-Chat-int4-onnx-directml",
        subfolder=".",
        repo_type="model",
        context_length=4096,
    ),
}

cpu_model_dict_list = {
    "EmbeddedLLM/Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32/tree/main",
        repo_id="EmbeddedLLM/Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32",
        model_name="Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32",
        subfolder=".",
        repo_type="model",
        context_length=4096,
    ),
    "EmbeddedLLM/Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4/tree/main",
        repo_id="EmbeddedLLM/Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4",
        model_name="Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4",
        subfolder=".",
        repo_type="model",
        context_length=4096,
    ),
    "EmbeddedLLM/Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32/tree/main",
        repo_id="EmbeddedLLM/Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32",
        model_name="Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32",
        subfolder=".",
        repo_type="model",
        context_length=131072,
    ),
    "EmbeddedLLM/Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4/tree/main",
        repo_id="EmbeddedLLM/Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4",
        model_name="Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4",
        subfolder=".",
        repo_type="model",
        context_length=131072,
    ),
    "EmbeddedLLM/mistral-7b-instruct-v0.3-onnx-cpu-int4-rtn-block-32-acc-level-4": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/mistral-7b-instruct-v0.3-onnx-cpu-int4-rtn-block-32-acc-level-4/tree/main",
        repo_id="EmbeddedLLM/mistral-7b-instruct-v0.3-onnx-cpu-int4-rtn-block-32-acc-level-4",
        model_name="mistral-7b-instruct-v0.3-onnx-cpu-int4-rtn-block-32-acc-level-4",
        subfolder=".",
        repo_type="model",
        context_length=32768,
    ),
    "EmbeddedLLM/mistral-7b-instruct-v0.3-onnx-cpu-int4-rtn-block-32": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/mistral-7b-instruct-v0.3-onnx-cpu-int4-rtn-block-32/tree/main",
        repo_id="EmbeddedLLM/mistral-7b-instruct-v0.3-onnx-cpu-int4-rtn-block-32",
        model_name="mistral-7b-instruct-v0.3-onnx-cpu-int4-rtn-block-32",
        subfolder=".",
        repo_type="model",
        context_length=32768,
    ),
    "EmbeddedLLM/openchat-3.6-8b-20240522-onnx-cpu-int4-rtn-block-32-acc-level-4": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/openchat-3.6-8b-20240522-onnx-cpu-int4-rtn-block-32-acc-level-4/tree/main",
        repo_id="EmbeddedLLM/openchat-3.6-8b-20240522-onnx-cpu-int4-rtn-block-32-acc-level-4",
        model_name="openchat-3.6-8b-20240522-onnx-cpu-int4-rtn-block-32-acc-level-4",
        subfolder=".",
        repo_type="model",
        context_length=8192,
    ),
    "EmbeddedLLM/openchat-3.6-8b-20240522-onnx-cpu-int4-rtn-block-32": ModelCard(
        hf_url="https://huggingface.co/EmbeddedLLM/openchat-3.6-8b-20240522-onnx-cpu-int4-rtn-block-32/tree/main",
        repo_id="EmbeddedLLM/openchat-3.6-8b-20240522-onnx-cpu-int4-rtn-block-32",
        model_name="openchat-3.6-8b-20240522-onnx-cpu-int4-rtn-block-32",
        subfolder=".",
        repo_type="model",
        context_length=8192,
    ),
    # 'EmbeddedLLM/Phi-3-vision-128k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4': ModelCard(
    #     hf_url='https://huggingface.co/EmbeddedLLM/Phi-3-vision-128k-instruct-onnx/tree/main/onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4',
    #     repo_id='EmbeddedLLM/Phi-3-vision-128k-instruct-onnx',
    #     model_name='Phi-3-vision-128k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4',
    #     subfolder= 'onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4',
    #     repo_type='model',
    #    context_length=128000
    # ),
}


def bytes_to_gb(bytes_value):
    """
    Convert bytes to gigabytes.

    :param bytes_value: The value in bytes to convert
    :return: The value in gigabytes, rounded to 2 decimal places
    """
    gb_value = bytes_value / (1024**3)  # 1 GB = 1024^3 bytes
    return round(gb_value, 4)


def compute_memory_size(repo_id, path_in_repo, repo_type: str = "model"):
    # Initialize the API
    api = HfApi()

    # Get the list of files in the repository
    files_info = api.list_repo_tree(
        repo_id=repo_id, revision="main", path_in_repo=path_in_repo, repo_type=repo_type
    )

    total_size_bytes = 0
    # Print the file sizes
    for file_info in files_info:
        # print(f"File: {file_info.path}, Size: {file_info.size} bytes")
        total_size_bytes += int(file_info.size)

    return bytes_to_gb(total_size_bytes)


for k, v in cpu_model_dict_list.items():
    v.size = compute_memory_size(
        repo_id=v.repo_id, path_in_repo=v.subfolder, repo_type=v.repo_type
    )

for k, v in dml_model_dict_list.items():
    v.size = compute_memory_size(
        repo_id=v.repo_id, path_in_repo=v.subfolder, repo_type=v.repo_type
    )

for k, v in ipex_model_dict_list.items():
    v.size = compute_memory_size(
        repo_id=v.repo_id, path_in_repo=v.subfolder, repo_type=v.repo_type
    )


def convert_to_dataframe(model_dict_list):
    # Create lists to store the data
    model_names = []
    hf_urls = []
    repo_ids = []
    model_names_full = []
    subfolders = []
    repo_types = []
    sizes = []
    context_lengths = []

    # Iterate through the dictionary and extract the data
    for key, model_card in model_dict_list.items():
        model_names.append(key)
        hf_urls.append(model_card.hf_url)
        repo_ids.append(model_card.repo_id)
        model_names_full.append(model_card.model_name)
        subfolders.append(model_card.subfolder)
        repo_types.append(model_card.repo_type)
        sizes.append(model_card.size)
        context_lengths.append(model_card.context_length)

    # Create a dictionary with the extracted data
    data = {
        "Model Name": model_names,
        "Size (GB)": sizes,
        "Context Length": context_lengths,
        "HuggingFace URL": hf_urls,
        "Repository ID": repo_ids,
        "Full Model Name": model_names_full,
        "Subfolder": subfolders,
        "Repository Type": repo_types,
    }

    # Create and return the pandas DataFrame
    return pd.DataFrame(data)


def check_health(url, max_retries=100, retry_delay=5):
    """
    Check the health status of an OpenAI API-compatible server.

    :param url: The URL of the health endpoint
    :param max_retries: Maximum number of retry attempts
    :param retry_delay: Delay between retries in seconds
    :return: True if healthy, False if max retries reached
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                print(f"Server is healthy after {attempt + 1} attempt(s).")
                return True

            print(
                f"Attempt {attempt + 1}/{max_retries}: Server not healthy. Retrying in {retry_delay} seconds..."
            )
            time.sleep(retry_delay)

        except requests.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Error connecting to server: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    print(f"Max retries ({max_retries}) reached. Server is not healthy.")
    return False


def chat_completion(url: str, payload: dict):
    with httpx.Client() as client:
        response = client.post(url, json=payload)
        if response.status_code == 200:
            print(response.text)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


def update_model_list(engine_type):
    if isinstance(engine_type, list):
        engine_type = engine_type[0]

    if engine_type == "DirectML":
        models = sorted(list(dml_model_dict_list.keys()))
        models_pandas = convert_to_dataframe(dml_model_dict_list)
    elif engine_type == "Ipex":
        models = sorted(list(ipex_model_dict_list.keys()))
        models_pandas = convert_to_dataframe(ipex_model_dict_list)
    else:
        models = sorted(list(cpu_model_dict_list.keys()))
        models_pandas = convert_to_dataframe(cpu_model_dict_list)

    return gr.Dropdown(choices=models, value=models[0] if models else None), gr.Dataframe(
        value=models_pandas if len(models_pandas) > 0 else None, datatype="markdown"
    )


def deploy_model(engine_type, model_name, port_number):
    global deployed_model

    # If engine_type is a list, take the first element
    if isinstance(engine_type, list):
        engine_type = engine_type[0]

    # Handle model_name if it's a list
    if isinstance(model_name, list):
        model_name = model_name[0]

    if engine_type == "DirectML":
        llm_model_card = dml_model_dict_list[model_name]
    elif engine_type == "Ipex":
        llm_model_card = ipex_model_dict_list[model_name]
    else:
        llm_model_card = cpu_model_dict_list[model_name]

    snapshot_path = snapshot_download(
        repo_id=llm_model_card.repo_id,
        allow_patterns=(
            f"{llm_model_card.subfolder}/*" if llm_model_card.subfolder != "." else None
        ),
        repo_type="model",
    )

    if llm_model_card.subfolder != ".":
        model_path = os.path.join(snapshot_path, llm_model_card.subfolder)
    else:
        model_path = snapshot_path

    print("Model path:", model_path)

    if engine_type == "Ipex":
        device = "xpu"

    else:
        device = "cpu"

    deployed_model.process = subprocess.Popen(
        [
            "ellm_server",
            "--model_path",
            model_path,
            "--backend",
            backend,
            "--device",
            device,
            "--port",
            f"{port_number}",
            # "--served_model_name",
            # model_name
        ]
    )

    deployed_model.model_name = model_name

    while True:
        # ping the server to see if it is up.
        if check_health(f"http://localhost:{port_number}/health"):
            break

    deployment_message = f"""
    <div style="padding: 10px; background-color: #58DE3A; border-radius: 5px;">
        <h2 style="color: #2D2363;">Deployment Status:</h2>
        <p style="color: #2D2363;"><strong>Model:</strong> {model_name}</p>
        <p style="color: #2D2363;"><strong>Engine:</strong> {engine_type}</p>
        <p style="color: #2D2363;"><strong>Port:</strong> {port_number}</p>
        <p style="color: #2D2363;"><strong>Model Path:</strong> {model_path}</p>
    </div>
    """

    return gr.Button(value="Stop Chat Server", interactive=True), gr.HTML(deployment_message)


def stop_deploy_model():
    global deployed_model

    try:
        if deployed_model.process:
            deployed_model.process.terminate()
            deployed_model.process = None
            deployed_model.model_name = ""
    except Exception as e:
        print(f"Error: When killing chat server encounter error - {str(e)}")

    return gr.Button(value="Stop Chat Server", interactive=False), gr.HTML("")


def download_model(engine_type, model_name):
    # If engine_type is a list, take the first element
    if isinstance(engine_type, list):
        engine_type = engine_type[0]

    if engine_type == "DirectML":
        llm_model_card = dml_model_dict_list[model_name]
    elif engine_type == "Ipex":
        llm_model_card = ipex_model_dict_list[model_name]
    else:
        llm_model_card = cpu_model_dict_list[model_name]

    # Handle model_name if it's a list
    if isinstance(model_name, list):
        model_name = model_name[0]

    yield "Downloading ..."
    snapshot_path = snapshot_download(
        repo_id=llm_model_card.repo_id,
        allow_patterns=(
            f"{llm_model_card.subfolder}/*" if llm_model_card.subfolder != "." else None
        ),
        repo_type="model",
    )
    yield snapshot_path


def main():
    with gr.Blocks(title="EmbeddedLLM Chatbot", theme="freddyaboulton/dracula_revamped") as demo:
        gr.HTML(
            """
        <div style="height: 100%; width: 100%; font-size: 24px; text-align: center;">Embedded LLM Engine</div>
        """
        )

        html_content = f"""
        <div style="
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <p style="font-size: 24px; font-weight: bold; color: #007bff;">Backend: {backend}</p>
        </div>
        """
        gr.HTML(html_content)

        with gr.Accordion("See More Model Details", open=False):
            model_info_pandas_frame = gr.Dataframe(value=None)

        default_value = "CPU"  # Default value
        if backend == "directml":
            default_value = "DirectML"
        elif backend == "ipex":
            default_value = "Ipex"
        selected_engine_type = gr.Dropdown(
            choices=["DirectML", "Ipex", "CPU"],
            value=default_value,
            multiselect=False,
            label="LLM Engine",
            show_label=True,
            key="engine_type_dropdown",
            interactive=True,
        )

        selected_model = gr.Dropdown(
            choices=[], label="Select Model", interactive=True, multiselect=False
        )

        selected_engine_type.change(
            fn=update_model_list,
            inputs=selected_engine_type,
            outputs=[selected_model, model_info_pandas_frame],
        )

        # Initialize the model list based on the default engine type
        demo.load(
            fn=update_model_list,
            inputs=selected_engine_type,
            outputs=[selected_model, model_info_pandas_frame],
        )

        with gr.Row():
            with gr.Column(scale=1):
                log_textbox = gr.Textbox(label="Download to", interactive=False)
                download_button = gr.Button(value="Download Model", interactive=True)
                download_button.click(
                    download_model,
                    inputs=[selected_engine_type, selected_model],
                    outputs=[log_textbox],
                )

            with gr.Column(scale=1):
                port_number = gr.Number(label="Port Number", value=5555)

                with gr.Row():
                    with gr.Column(scale=1):
                        deploy_button = gr.Button(value="Start Chat Server", interactive=True)
                    with gr.Column(scale=1):
                        stop_deploy_button = gr.Button(value="Stop Chat Server", interactive=False)

                deployment_status = gr.HTML()  # New component to display deployment status

                deploy_button.click(
                    deploy_model,
                    inputs=[selected_engine_type, selected_model, port_number],
                    outputs=[stop_deploy_button, deployment_status],
                )
                stop_deploy_button.click(
                    stop_deploy_model, outputs=[stop_deploy_button, deployment_status]
                )

    demo.queue()
    demo.launch(server_port=config.port, server_name=config.host)


if __name__ == "__main__":
    main()
