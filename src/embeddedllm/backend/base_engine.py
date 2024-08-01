# from embeddedllm.transformers_utils.image_processing_phi3v import Phi3VImageProcessor
# import contextlib
# import time
# from pathlib import Path
# from tempfile import TemporaryDirectory
from typing import AsyncIterator, List, Optional

# import onnxruntime_genai as og
from loguru import logger

# from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from embeddedllm.inputs import PromptInputs
from embeddedllm.protocol import CompletionOutput, RequestOutput
from embeddedllm.sampling_params import SamplingParams

RECORD_TIMING = True


class BaseLLMEngine:
    def __init__(self, model_path: str, vision: bool, device: str = "xpu"):
        pass

        # self.model_path = model_path
        # self.model_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        # # model_config is to find out the max length of the model
        # self.max_model_len = _get_and_verify_max_len(
        #     hf_config=self.model_config,
        #     max_model_len=None,
        #     disable_sliding_window=False,
        #     sliding_window_len=self.get_hf_config_sliding_window(),
        # )

        # logger.info("Model Context Lenght: " + str(self.max_model_len))

        # try:
        #     logger.info("Attempt to load fast tokenizer")
        #     self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_path)
        # except Exception:
        #     logger.info("Attempt to load slower tokenizer")
        #     self.tokenizer = PreTrainedTokenizer.from_pretrained(self.model_path)

        # self.model = og.Model(model_path)
        # logger.info("Model loaded")
        # self.onnx_tokenizer = og.Tokenizer(self.model)
        # self.onnx_tokenizer_stream = self.onnx_tokenizer.create_stream()
        # logger.info("Tokenizer created")

        # self.vision = vision

        # if self.vision:
        #     self.onnx_processor = self.model.create_multimodal_processor()
        #     self.processor = AutoImageProcessor.from_pretrained(
        #         self.model_path, trust_remote_code=True
        #     )

    def get_hf_config_sliding_window(self) -> Optional[int]:
        """Get the sliding window size, or None if disabled."""

        # Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
        # addition to sliding window size. We check if that field is present
        # and if it's False, return None.
        if (
            hasattr(self.model_config, "use_sliding_window")
            and not self.model_config.use_sliding_window
        ):
            return None
        return getattr(self.model_config, "sliding_window", None)

    async def generate_vision(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        stream: bool = True,
    ) -> AsyncIterator[RequestOutput]:
        raise NotImplementedError(
            f"`generate_vision` has to be overwritten and implemented by the inherited class."
        )

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

        raise NotImplementedError(
            f"`generate` has to be overwritten and implemented by the inherited class."
        )


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
    disable_sliding_window: bool,
    sliding_window_len: Optional[int],
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys.
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len \
                else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if disable_sliding_window and sliding_window_len is not None:
        max_len_key = "sliding_window" \
            if sliding_window_len < derived_max_model_len else max_len_key
        derived_max_model_len = min(derived_max_model_len, sliding_window_len)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.", possible_keys,
            default_max_len)
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        if "type" in rope_scaling:
            rope_type = rope_scaling["type"]
        elif "rope_type" in rope_scaling:
            rope_type = rope_scaling["rope_type"]
        else:
            raise ValueError(
                "rope_scaling must have a 'type' or 'rope_type' key.")

        # The correct one should be "longrope", kept "su" here
        # to be backward compatible
        if rope_type not in ("su", "longrope", "llama3"):
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that supports rope_scaling
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "with rope_scaling. Please raise an issue so we can "
                    "investigate.")

            assert "factor" in rope_scaling
            scaling_factor = rope_scaling["factor"]
            if rope_type == "yarn":
                derived_max_model_len = rope_scaling[
                    "original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    # If the user specified a max length, make sure it is smaller than the
    # derived length from the HF model config.
    if max_model_len is None:
        max_model_len = int(derived_max_model_len)
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_config, "model_max_length", None)
        if model_max_length is not None and max_model_len <= model_max_length:
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that has model_max_length
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "model_max_length in the config. Please raise an issue "
                    "so we can investigate.")
            pass
        else:
            raise ValueError(
                f"User-specified max_model_len ({max_model_len}) is greater "
                "than the derived max_model_len "
                f"({max_len_key}={derived_max_model_len} or model_max_length="
                f"{model_max_length} in model's config.json). This may lead "
                "to incorrect model outputs or CUDA errors. Make sure the "
                "value is correct and within the model context size.")
    return int(max_model_len)
