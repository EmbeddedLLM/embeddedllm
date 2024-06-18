import onnxruntime_genai as og
from typing import Optional, AsyncIterator, List, Iterator
from loguru import logger
import time
from transformers import AutoConfig, PretrainedConfig
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from transformers import AutoProcessor, AutoImageProcessor
from embeddedllm.sampling_params import SamplingParams
from embeddedllm.protocol import RequestOutput, CompletionOutput
from embeddedllm.inputs import PromptInputs
from tempfile import TemporaryDirectory
from pathlib import Path
from PIL import Image
# from embeddedllm.transformers_utils.image_processing_phi3v import Phi3VImageProcessor
import contextlib

RECORD_TIMING=True

@contextlib.contextmanager
def onnx_generator_context(model, params):
    generator = None
    try:
        generator = og.Generator(model, params)
        yield generator
    finally:
        if generator is not None:
            # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
            del generator

class EmbeddedLLMEngine():

    def __init__(self, model_path: str, vision: bool):
        self.model_path = model_path
        self.model_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        ## model_config is to find out the max length of the model
        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.model_config,
            max_model_len=None,
            disable_sliding_window=False,
            sliding_window_len=self.get_hf_config_sliding_window())

        logger.info(self.max_model_len)

        try:
            logger.info("Attempt to load fast tokenizer")
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_path)
        except Exception as e:
            logger.info("Attempt to load slower tokenizer")
            self.tokenizer = PreTrainedTokenizer.from_pretrained(self.model_path)

        self.model = og.Model(model_path)
        logger.info("Model loaded")
        self.onnx_tokenizer = og.Tokenizer(self.model)
        self.onnx_tokenizer_stream = self.onnx_tokenizer.create_stream()
        logger.info("Tokenizer created")

        self.vision = vision

        if self.vision:
            self.onnx_processor = self.model.create_multimodal_processor()
            self.processor = AutoImageProcessor.from_pretrained(self.model_path, trust_remote_code=True) 
            print(dir(self.processor))

        
        # input_token_length = len(self.tokenizer.encode("hello world how areyou"))
        # logger.info(input_token_length)

        # search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}
        
        # # Set the max length to something sensible by default, unless it is specified by the user,
        # # since otherwise it will be set to the entire context length
        # if 'max_length' not in search_options:
        #     search_options['max_length'] = 2048


    def get_hf_config_sliding_window(self) -> Optional[int]:
        """Get the sliding window size, or None if disabled.
        """

        # Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
        # addition to sliding window size. We check if that field is present
        # and if it's False, return None.
        if (hasattr(self.model_config, "use_sliding_window")
                and not self.model_config.use_sliding_window):
            return None
        return getattr(self.model_config, "sliding_window", None)


    async def generate_vision(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        stream: bool = True,
    ) -> AsyncIterator[RequestOutput]:

        prompt_text = inputs["prompt"]
        # print(f"inputs: {str(inputs)}")
        # print(inputs.keys())
        input_tokens = self.onnx_tokenizer.encode(prompt_text)
        # logger.debug(f"inputs: {str(inputs)}")
        # logger.debug(f'inputs["multi_model_data"]: {str(inputs.multi_model_data)}')
        # print(type(inputs))
        # logger.debug(inputs['multi_modal_data'][0])
        file_data = inputs['multi_modal_data'][0]['image_pixel_data']
        mime_type = inputs['multi_modal_data'][0]['mime_type']
    
        assert "image" in mime_type
        ext = "." + mime_type.split("/")[-1]
        with TemporaryDirectory() as tmpdirname:
            filepath = Path(tmpdirname) / "tmpfile"
            image_path = filepath.with_suffix(ext)
            with image_path.open("wb") as tmpfile:
                tmpfile.write(file_data)  # Decode bytes to string
                tmpfile.flush()

            # logger.trace("Loading from temporary file: {name}", name=image_path.as_posix())
            logger.debug("Loading from temporary file: {name}", name=image_path.as_posix())
            
            # if not os.path.exists(image_path.as_posix()):
            #     raise FileNotFoundError(f"Image file not found: {image_path.as_posix()}")
            # logger.debug("Loading ONNX Image")
            image_pixel_data = Image.open(image_path.as_posix())
            input_token_length = self.processor.calc_num_image_tokens(image_pixel_data)[0]

            # logger.debug(f"input_token_length: {str(input_token_length)}")
            max_tokens = sampling_params.max_tokens

            assert input_token_length is not None

            if input_token_length + max_tokens > self.max_model_len:
                raise ValueError("Exceed Context Length")
            
            search_options = {name:getattr(sampling_params, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if hasattr(sampling_params, name)}
            image = og.Images.open(image_path.as_posix())
            inputs = self.onnx_processor(prompt_text, images=image)
            search_options["max_length"] = input_token_length + max_tokens
            params = og.GeneratorParams(self.model)
            params.set_search_options(**search_options)
            params.set_inputs(inputs)

            token_list: List[int] = []
            output_text: str = ""
            if stream:
                with onnx_generator_context(self.model, params) as generator:
                    if RECORD_TIMING:
                        started_timestamp = time.time()
                        first_token_timestamp = 0
                        first = True
                        has_input_tokens = False
                        new_tokens = []
                    try:
                        while not generator.is_done():
                            # logger.debug("Compute Logits")
                            generator.compute_logits()
                            # logger.debug("Compute generate_next_token")
                            generator.generate_next_token()
                            if RECORD_TIMING:
                                if first:
                                    first_token_timestamp = time.time()
                                    first = False

                            # new_token_list = generator.get_next_tokens()
                            # new_token = new_token_list[0]
                            token_list = generator.get_sequence(0)
                            new_token = token_list[-1]
                            output_text += self.onnx_tokenizer_stream.decode(new_token)
                            # logger.debug(self.onnx_tokenizer_stream.decode(new_token))

                            if not has_input_tokens:
                                input_tokens = token_list[:-1]

                            output = RequestOutput(
                                request_id=request_id,
                                prompt=inputs,
                                prompt_token_ids=input_tokens,
                                finished=False,
                                outputs=[CompletionOutput(
                                    index=0,
                                    text=output_text,
                                    token_ids=token_list,
                                    cumulative_logprob=-1.0,

                                )]
                            )
                            yield output
                            # logits = generator.get_output("logits")
                            # print(output)
                            if RECORD_TIMING: new_tokens.append(new_token)

                    
                        yield RequestOutput(
                            request_id=request_id,
                            prompt=inputs,
                            prompt_token_ids=input_tokens,
                            finished=True,
                            outputs=[CompletionOutput(
                                index=0,
                                text=output_text,
                                token_ids=token_list,
                                cumulative_logprob=-1.0,
                                finish_reason="stop")]
                        )
                        if RECORD_TIMING:
                            prompt_time = first_token_timestamp - started_timestamp
                            run_time = time.time() - first_token_timestamp
                            logger.info(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")


                    except Exception as e:
                        logger.error(str(e))
                        
                        error_output = RequestOutput(
                            prompt=inputs,
                            prompt_token_ids=input_tokens,
                            finished=True,
                            request_id=request_id,
                            outputs=[CompletionOutput(
                                index=0,
                                text=output_text,
                                token_ids=token_list,
                                cumulative_logprob=-1.0,
                                finish_reason="error",
                                stop_reason=str(e))]
                        )
                        yield error_output
            else:
                try:
                    token_list=self.model.generate(params)

                    output_text = self.onnx_tokenizer.decode(token_list[0])
                
                    yield RequestOutput(
                        request_id=request_id,
                        prompt=inputs,
                        prompt_token_ids=input_tokens,
                        finished=True,
                        outputs=[CompletionOutput(
                            index=0,
                            text=output_text,
                            token_ids=token_list,
                            cumulative_logprob=-1.0,
                            finish_reason="stop")]
                    )

                except Exception as e:
                    logger.error(str(e))
                    
                    error_output = RequestOutput(
                        prompt=inputs,
                        prompt_token_ids=input_tokens,
                        finished=True,
                        request_id=request_id,
                        outputs=[CompletionOutput(
                            index=0,
                            text=output_text,
                            token_ids=token_list,
                            cumulative_logprob=-1.0,
                            finish_reason="error",
                            stop_reason=str(e))]
                    )
                    yield error_output

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
        input_tokens = None # for text only use case
        logger.debug("inputs: "+ prompt_text)

        input_tokens = self.onnx_tokenizer.encode(prompt_text)
        input_token_length = len(input_tokens)
        
        max_tokens = sampling_params.max_tokens

        assert input_token_length is not None

        if input_token_length + max_tokens > self.max_model_len:
            raise ValueError("Exceed Context Length")
        
        search_options = {name:getattr(sampling_params, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if hasattr(sampling_params, name)}

        search_options["max_length"] = input_token_length + max_tokens
        params = og.GeneratorParams(self.model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens

        token_list: List[int] = []
        output_text: str = ""
        if stream:
            with onnx_generator_context(self.model, params) as generator:
                if RECORD_TIMING:
                    started_timestamp = time.time()
                    first_token_timestamp = 0
                    first = True
                    new_tokens = []
                try:
                    while not generator.is_done():
                        # logger.debug("Compute Logits")
                        generator.compute_logits()
                        # logger.debug("Compute generate_next_token")
                        generator.generate_next_token()
                        if RECORD_TIMING:
                            if first:
                                first_token_timestamp = time.time()
                                first = False

                        # new_token_list = generator.get_next_tokens()
                        # new_token = new_token_list[0]
                        token_list = generator.get_sequence(0)
                        new_token = token_list[-1]
                        output_text += self.onnx_tokenizer_stream.decode(new_token)
                        # logger.debug(self.onnx_tokenizer_stream.decode(new_token))

                        output = RequestOutput(
                            request_id=request_id,
                            prompt=inputs,
                            prompt_token_ids=input_tokens,
                            finished=False,
                            outputs=[CompletionOutput(
                                index=0,
                                text=output_text,
                                token_ids=token_list,
                                cumulative_logprob=-1.0,

                            )]
                        )
                        yield output
                        # logits = generator.get_output("logits")
                        # print(logits)
                        if RECORD_TIMING: new_tokens.append(new_token)

                
                    yield RequestOutput(
                        request_id=request_id,
                        prompt=inputs,
                        prompt_token_ids=input_tokens,
                        finished=True,
                        outputs=[CompletionOutput(
                            index=0,
                            text=output_text,
                            token_ids=token_list,
                            cumulative_logprob=-1.0,
                            finish_reason="stop")]
                    )
                    if RECORD_TIMING:
                        prompt_time = first_token_timestamp - started_timestamp
                        run_time = time.time() - first_token_timestamp
                        logger.info(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")


                except Exception as e:
                    logger.error(str(e))
                    
                    error_output = RequestOutput(
                        prompt=inputs,
                        prompt_token_ids=input_tokens,
                        finished=True,
                        request_id=request_id,
                        outputs=[CompletionOutput(
                            index=0,
                            text=output_text,
                            token_ids=token_list,
                            cumulative_logprob=-1.0,
                            finish_reason="error",
                            stop_reason=str(e))]
                    )
                    yield error_output
        else:
            try:
                token_list=self.model.generate(params)

                output_text = self.onnx_tokenizer.decode(token_list[0])
            
                yield RequestOutput(
                    request_id=request_id,
                    prompt=inputs,
                    prompt_token_ids=input_tokens,
                    finished=True,
                    outputs=[CompletionOutput(
                        index=0,
                        text=output_text,
                        token_ids=token_list,
                        cumulative_logprob=-1.0,
                        finish_reason="stop")]
                )

            except Exception as e:
                logger.error(str(e))
                
                error_output = RequestOutput(
                    prompt=inputs,
                    prompt_token_ids=input_tokens,
                    finished=True,
                    request_id=request_id,
                    outputs=[CompletionOutput(
                        index=0,
                        text=output_text,
                        token_ids=token_list,
                        cumulative_logprob=-1.0,
                        finish_reason="error",
                        stop_reason=str(e))]
                )
                yield error_output
    
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
    if rope_scaling is not None and rope_scaling["type"] != "su":
        if disable_sliding_window:
            # TODO(robertgshaw): Find a model that supports rope_scaling
            # with sliding window to see if this case should be allowed.
            raise NotImplementedError(
                "Disabling sliding window is not supported for models "
                "with rope_scaling. Please raise an issue so we can "
                "investigate.")
        assert "factor" in rope_scaling
        scaling_factor = rope_scaling["factor"]
        if rope_scaling["type"] == "yarn":
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
