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


class EmbeddedLLMEngine:
    def __init__(self, model_path: str, vision: bool, device: str = "xpu", backend: str = "ipex"):
        self.model_path = model_path
        self.vision = vision
        self.backend = backend
        self.device = device
        self.engine = None
        if self.backend == "ipex":
            from embeddedllm.backend.ipex_engine import IpexEngine

            if self.device == "xpu":
                os.environ["SYCL_CACHE_PERSISTENT"] = "1"
            self.engine = IpexEngine(self.model_path, self.vision, self.device)
            logger.info(f"Initializing xpu backend: IpexEngine")
        elif self.backend == "directml":
            from embeddedllm.backend.onnxruntime_engine import OnnxruntimeEngine

            self.engine = OnnxruntimeEngine(self.model_path, self.vision, self.device)
            logger.info(f"Initializing xpu backend: OnnxruntimeEngine")
        else:
            raise ValueError(f"EmbeddedLLMEngine only supports `xpu` and `directml`.")
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
