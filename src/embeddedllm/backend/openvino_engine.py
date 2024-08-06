import contextlib
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import AsyncIterator, List, Optional

from loguru import logger
from PIL import Image
from transformers import (
    AutoConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TextIteratorStreamer,
)

from threading import Thread

from optimum.intel import OVModelForCausalLM

from embeddedllm.inputs import PromptInputs
from embeddedllm.protocol import CompletionOutput, RequestOutput
from embeddedllm.sampling_params import SamplingParams
from embeddedllm.backend.base_engine import BaseLLMEngine, _get_and_verify_max_len

RECORD_TIMING = True


class OpenVinoEngine(BaseLLMEngine):
    def __init__(self, model_path: str, vision: bool, device: str = "gpu"):
        self.model_path = model_path
        self.model_config: AutoConfig = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.device = device

        # model_config is to find out the max length of the model
        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.model_config,
            max_model_len=None,
            disable_sliding_window=False,
            sliding_window_len=self.get_hf_config_sliding_window(),
        )

        logger.info("Model Context Lenght: " + str(self.max_model_len))

        try:
            logger.info("Attempt to load fast tokenizer")
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_path)
        except Exception:
            logger.info("Attempt to load slower tokenizer")
            self.tokenizer = PreTrainedTokenizer.from_pretrained(self.model_path)

        self.model = OVModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, export=False, device=self.device
        )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path, trust_remote_code=True
        # ).to(self.device)
        logger.info("Model loaded")
        self.tokenizer_stream = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        logger.info("Tokenizer created")

        self.vision = vision

        # if self.vision:
        #     self.onnx_processor = self.model.create_multimodal_processor()
        #     self.processor = AutoImageProcessor.from_pretrained(
        #         self.model_path, trust_remote_code=True
        #     )
        #     print(dir(self.processor))

    async def generate_vision(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        stream: bool = True,
    ) -> AsyncIterator[RequestOutput]:
        raise NotImplementedError(f"`generate_vision` yet to be implemented.")
        # prompt_text = inputs["prompt"]
        # input_tokens = self.onnx_tokenizer.encode(prompt_text)
        # file_data = inputs["multi_modal_data"][0]["image_pixel_data"]
        # mime_type = inputs["multi_modal_data"][0]["mime_type"]

        # assert "image" in mime_type
        # ext = "." + mime_type.split("/")[-1]
        # with TemporaryDirectory() as tmpdirname:
        #     filepath = Path(tmpdirname) / "tmpfile"
        #     image_path = filepath.with_suffix(ext)
        #     with image_path.open("wb") as tmpfile:
        #         tmpfile.write(file_data)  # Decode bytes to string
        #         tmpfile.flush()
        #     image_pixel_data = Image.open(image_path.as_posix())
        #     input_token_length = self.processor.calc_num_image_tokens(image_pixel_data)[0]
        #     max_tokens = sampling_params.max_tokens

        #     assert input_token_length is not None

        #     if input_token_length + max_tokens > self.max_model_len:
        #         raise ValueError("Exceed Context Length")

        #     search_options = {
        #         name: getattr(sampling_params, name)
        #         for name in [
        #             "do_sample",
        #             "max_length",
        #             "min_length",
        #             "top_p",
        #             "top_k",
        #             "temperature",
        #             "repetition_penalty",
        #         ]
        #         if hasattr(sampling_params, name)
        #     }
        #     image = og.Images.open(image_path.as_posix())
        #     inputs = self.onnx_processor(prompt_text, images=image)
        #     search_options["max_length"] = input_token_length + max_tokens
        #     params = og.GeneratorParams(self.model)
        #     params.set_search_options(**search_options)
        #     params.set_inputs(inputs)

        #     token_list: List[int] = []
        #     output_text: str = ""
        #     if stream:
        #         with onnx_generator_context(self.model, params) as generator:
        #             if RECORD_TIMING:
        #                 started_timestamp = time.time()
        #                 first_token_timestamp = 0
        #                 first = True
        #                 has_input_tokens = False
        #                 new_tokens = []
        #             try:
        #                 while not generator.is_done():
        #                     # logger.debug("Compute Logits")
        #                     generator.compute_logits()
        #                     # logger.debug("Compute generate_next_token")
        #                     generator.generate_next_token()
        #                     if RECORD_TIMING:
        #                         if first:
        #                             first_token_timestamp = time.time()
        #                             first = False

        #                     # new_token_list = generator.get_next_tokens()
        #                     # new_token = new_token_list[0]
        #                     token_list = generator.get_sequence(0)
        #                     new_token = token_list[-1]
        #                     output_text += self.onnx_tokenizer_stream.decode(new_token)
        #                     # logger.debug(self.onnx_tokenizer_stream.decode(new_token))

        #                     if not has_input_tokens:
        #                         input_tokens = token_list[:-1]

        #                     output = RequestOutput(
        #                         request_id=request_id,
        #                         prompt=inputs,
        #                         prompt_token_ids=input_tokens,
        #                         finished=False,
        #                         outputs=[
        #                             CompletionOutput(
        #                                 index=0,
        #                                 text=output_text,
        #                                 token_ids=token_list,
        #                                 cumulative_logprob=-1.0,
        #                             )
        #                         ],
        #                     )
        #                     yield output
        #                     # logits = generator.get_output("logits")
        #                     # print(output)
        #                     if RECORD_TIMING:
        #                         new_tokens.append(new_token)

        #                 yield RequestOutput(
        #                     request_id=request_id,
        #                     prompt=inputs,
        #                     prompt_token_ids=input_tokens,
        #                     finished=True,
        #                     outputs=[
        #                         CompletionOutput(
        #                             index=0,
        #                             text=output_text,
        #                             token_ids=token_list,
        #                             cumulative_logprob=-1.0,
        #                             finish_reason="stop",
        #                         )
        #                     ],
        #                 )
        #                 if RECORD_TIMING:
        #                     prompt_time = first_token_timestamp - started_timestamp
        #                     run_time = time.time() - first_token_timestamp
        #                     logger.info(
        #                         f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps"
        #                     )

        #             except Exception as e:
        #                 logger.error(str(e))

        #                 error_output = RequestOutput(
        #                     prompt=inputs,
        #                     prompt_token_ids=input_tokens,
        #                     finished=True,
        #                     request_id=request_id,
        #                     outputs=[
        #                         CompletionOutput(
        #                             index=0,
        #                             text=output_text,
        #                             token_ids=token_list,
        #                             cumulative_logprob=-1.0,
        #                             finish_reason="error",
        #                             stop_reason=str(e),
        #                         )
        #                     ],
        #                 )
        #                 yield error_output
        #     else:
        #         try:
        #             token_list = self.model.generate(params)

        #             output_text = self.onnx_tokenizer.decode(token_list[0])

        #             yield RequestOutput(
        #                 request_id=request_id,
        #                 prompt=inputs,
        #                 prompt_token_ids=input_tokens,
        #                 finished=True,
        #                 outputs=[
        #                     CompletionOutput(
        #                         index=0,
        #                         text=output_text,
        #                         token_ids=token_list,
        #                         cumulative_logprob=-1.0,
        #                         finish_reason="stop",
        #                     )
        #                 ],
        #             )

        #         except Exception as e:
        #             logger.error(str(e))

        #             error_output = RequestOutput(
        #                 prompt=inputs,
        #                 prompt_token_ids=input_tokens,
        #                 finished=True,
        #                 request_id=request_id,
        #                 outputs=[
        #                     CompletionOutput(
        #                         index=0,
        #                         text=output_text,
        #                         token_ids=token_list,
        #                         cumulative_logprob=-1.0,
        #                         finish_reason="error",
        #                         stop_reason=str(e),
        #                     )
        #                 ],
        #             )
        #             yield error_output

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

        prompt_text = inputs["prompt"]
        input_token_length = None
        input_tokens = None  # for text only use case
        # logger.debug("inputs: " + prompt_text)

        input_tokens = self.tokenizer.encode(prompt_text, return_tensors="pt")
        # logger.debug(f"input_tokens: {input_tokens}")
        input_token_length = len(input_tokens[0])

        max_tokens = sampling_params.max_tokens

        assert input_token_length is not None

        if input_token_length + max_tokens > self.max_model_len:
            raise ValueError("Exceed Context Length")

        generation_options = {
            name: getattr(sampling_params, name)
            for name in [
                "do_sample",
                # "max_length",
                "max_new_tokens",
                "min_length",
                "top_p",
                "top_k",
                "temperature",
                "repetition_penalty",
            ]
            if hasattr(sampling_params, name)
        }
        generation_options["max_length"] = self.max_model_len
        generation_options["input_ids"] = input_tokens.clone()
        # generation_options["input_ids"] = input_tokens.clone().to(self.device)
        print(generation_options)

        token_list: List[int] = []
        output_text: str = ""
        if stream:
            generation_options["streamer"] = self.tokenizer_stream
            if RECORD_TIMING:
                started_timestamp = time.time()
                first_token_timestamp = 0
                first = True
                new_tokens = []
            try:
                thread = Thread(target=self.model.generate, kwargs=generation_options)
                started_timestamp = time.time()
                first_token_timestamp = None
                thread.start()
                output_text = ""
                first = True
                for new_text in self.tokenizer_stream:
                    if new_text == "":
                        continue
                    if RECORD_TIMING:
                        if first:
                            first_token_timestamp = time.time()
                            first = False
                    # logger.debug(f"new text: {new_text}")
                    output_text += new_text
                    token_list = self.tokenizer.encode(output_text, return_tensors="pt")

                    output = RequestOutput(
                        request_id=request_id,
                        prompt=prompt_text,
                        prompt_token_ids=input_tokens[0],
                        finished=False,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text=output_text,
                                token_ids=token_list[0],
                                cumulative_logprob=-1.0,
                            )
                        ],
                    )
                    yield output
                    # logits = generator.get_output("logits")
                    # print(logits)
                    if RECORD_TIMING:
                        new_tokens = token_list[0]

                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt_text,
                    prompt_token_ids=input_tokens[0],
                    finished=True,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=output_text,
                            token_ids=token_list[0],
                            cumulative_logprob=-1.0,
                            finish_reason="stop",
                        )
                    ],
                )
                if RECORD_TIMING:
                    prompt_time = first_token_timestamp - started_timestamp
                    run_time = time.time() - first_token_timestamp
                    logger.info(
                        f"Prompt length: {len(input_tokens[0])}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens[0])/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps"
                    )

            except Exception as e:
                logger.error(str(e))

                error_output = RequestOutput(
                    prompt=inputs,
                    prompt_token_ids=input_tokens,
                    finished=True,
                    request_id=request_id,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=output_text,
                            token_ids=token_list,
                            cumulative_logprob=-1.0,
                            finish_reason="error",
                            stop_reason=str(e),
                        )
                    ],
                )
                yield error_output
        else:
            try:
                token_list = self.model.generate(**generation_options)[0]

                output_text = self.tokenizer.decode(
                    token_list[input_token_length:], skip_special_tokens=True
                )

                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt_text,
                    prompt_token_ids=input_tokens[0],
                    finished=True,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=output_text,
                            token_ids=token_list,
                            cumulative_logprob=-1.0,
                            finish_reason="stop",
                        )
                    ],
                )

            except Exception as e:
                logger.error(str(e))

                error_output = RequestOutput(
                    prompt=prompt_text,
                    prompt_token_ids=input_tokens[0],
                    finished=True,
                    request_id=request_id,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=output_text,
                            token_ids=token_list,
                            cumulative_logprob=-1.0,
                            finish_reason="error",
                            stop_reason=str(e),
                        )
                    ],
                )
                yield error_output
