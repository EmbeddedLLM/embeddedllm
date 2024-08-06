# from embeddedllm.transformers_utils.image_processing_phi3v import Phi3VImageProcessor
# import contextlib
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import AsyncIterator, List, Optional

# import onnxruntime_genai as og
from loguru import logger

from embeddedllm.inputs import PromptInputs
from embeddedllm.protocol import CompletionOutput, RequestOutput
from embeddedllm.sampling_params import SamplingParams
import os
import platform


def get_processor_type():
    processor_info = platform.processor()
    if "intel" in processor_info.lower():
        return "Intel"
    elif "amd" in processor_info.lower():
        return "AMD"
    else:
        return "Unknown processor brand"


class EmbeddedLLMEngine:
    def __init__(self, model_path: str, vision: bool, device: str = "xpu", backend: str = "ipex"):
        self.model_path = model_path
        self.vision = vision
        self.backend = backend
        self.device = device
        self.engine = None
        if self.backend == "ipex":
            from embeddedllm.backend.ipex_engine import IpexEngine

            assert (
                self.device == "xpu"
            ), f"To run ipex on cpu, set `backend` to `cpu` and `device` to `cpu`. EmbeddedLLMEngine load model with ipex on Intel processor."
            if self.device == "xpu":
                os.environ["SYCL_CACHE_PERSISTENT"] = "1"
                os.environ["BIGDL_LLM_XMX_DISABLED"] = "1"
            self.engine = IpexEngine(self.model_path, self.vision, self.device)
            logger.info(f"Initializing ipex-llm backend (XPU): IpexEngine")
        elif self.backend == "openvino" and self.device == "gpu":
            from embeddedllm.backend.openvino_engine import OpenVinoEngine

            assert (
                self.device == "gpu"
            ), f"To run openvino on cpu, set `backend` to `openvino` and `device` to `cpu`. EmbeddedLLMEngine load model with openvino on Intel processor."
            self.engine = OpenVinoEngine(self.model_path, self.vision, self.device)
            logger.info(f"Initializing openvino backend (GPU): OpenVinoEngine")
        elif self.backend in ("directml", "cuda"):
            from embeddedllm.backend.onnxruntime_engine import OnnxruntimeEngine

            self.engine = OnnxruntimeEngine(self.model_path, self.vision, self.device)
            logger.info(f"Initializing onnxruntime backend ({backend.upper()}): OnnxruntimeEngine")
        elif self.backend == "cpu":
            assert self.device == "cpu", f"To run `cpu` backend, `device` must be `cpu`."
            processor = get_processor_type()
            if self.backend == "openvino":
                from embeddedllm.backend.openvino_engine import OpenVinoEngine

                self.engine = OpenVinoEngine(self.model_path, self.vision, self.device)
                logger.info(f"Initializing openvino backend (CPU): OpenVinoEngine")
            elif processor == "Intel":
                from embeddedllm.backend.ipex_engine import IpexEngine

                self.engine = IpexEngine(self.model_path, self.vision, self.device)
                logger.info(f"Initializing ipex-llm backend (CPU): IpexEngine")
            elif processor == "AMD":
                from embeddedllm.backend.onnxruntime_engine import OnnxruntimeEngine

                self.engine = OnnxruntimeEngine(self.model_path, self.vision, self.device)
                logger.info(f"Initializing onnxruntime backend (CPU): OnnxruntimeEngine")

            else:
                raise SystemError(f"Only support `intel` and `amd` CPU processor.")

        else:
            raise ValueError(
                f"EmbeddedLLMEngine only supports `cpu`, `ipex`, `cuda` and `directml`."
            )
        self.tokenizer = self.engine.tokenizer

    async def generate_vision(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        stream: bool = True,
    ) -> AsyncIterator[RequestOutput]:
        async for res in self.engine.generate_vision(inputs, sampling_params, request_id, stream):
            yield res

    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        stream: bool = True,
    ) -> AsyncIterator[RequestOutput]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        """

        async for res in self.engine.generate(inputs, sampling_params, request_id, stream):
            yield res
