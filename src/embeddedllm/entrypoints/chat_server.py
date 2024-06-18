import codecs
import json
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Iterable,
    List,
    Optional,
    TypedDict,
    Union,
    cast,
    final,
)

from fastapi import Request
from loguru import logger
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from embeddedllm.engine import EmbeddedLLMEngine
from embeddedllm.inputs import ImagePixelData, PromptInputs
from embeddedllm.protocol import (  # noqa: E501
    ChatCompletionContentPartParam,
    # ChatCompletionLogProb,
    # ChatCompletionLogProbs,
    # ChatCompletionLogProbsContent,
    ChatCompletionMessageParam,
    # ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    # CompletionOutput,
    CompletionRequest,
    DeltaMessage,
    ErrorResponse,
    # FunctionCall,
    ModelCard,
    ModelList,
    ModelPermission,
    RequestOutput,
    # ToolCall,
    UsageInfo,
)
from embeddedllm.utils import decode_base64, random_uuid


@final  # So that it should be compatible with Dict[str, str]
class ConversationMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]
    image_futures: List[ImagePixelData] = field(default_factory=list)


class OpenAPIChatServer:

    def __init__(
        self,
        model_path: str,
        served_model_name: str = "",
        response_role: str = "assistant",
        chat_template: Optional[str] = None,
        vision: Optional[bool] = False,
    ):
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.response_role = response_role
        self.vision = vision
        self.engine = EmbeddedLLMEngine(model_path, vision=self.vision)

        self.tokenizer = self.engine.tokenizer
        self._load_chat_template(chat_template)

    def _load_chat_template(self, chat_template: Optional[str]):
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = self.tokenizer

        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    tokenizer.chat_template = f.read()
            except OSError as e:
                JINJA_CHARS = "{}\n"
                if not any(c in chat_template for c in JINJA_CHARS):
                    msg = (
                        f"The supplied chat template ({chat_template}) "
                        f"looks like a file path, but it failed to be "
                        f"opened. Reason: {e}"
                    )
                    raise ValueError(msg) from e

                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                tokenizer.chat_template = codecs.decode(chat_template, "unicode_escape")

            logger.info("Using supplied chat template:\n%s", tokenizer.chat_template)
        elif tokenizer.chat_template is not None:
            logger.info("Using default chat template:\n%s", tokenizer.chat_template)
        else:
            logger.warning("No chat template provided. Chat API will not work.")

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> ErrorResponse:
        return ErrorResponse(message=message, type=err_type, code=status_code.value)

    def _parse_chat_message_content_parts(
        self,
        role: str,
        parts: Iterable[ChatCompletionContentPartParam],
    ) -> ChatMessageParseResult:
        texts: List[str] = []
        image_futures: List[ImagePixelData] = []

        for part in parts:
            # logger.debug(f"part: {str(part)}")
            part_type = part["type"]
            logger.debug(f"part_type: {part_type}")
            if part_type == "text":
                text = cast(ChatCompletionContentPartTextParam, part)["text"]

                texts.append(text)
            elif part_type == "image_url":
                if not self.vision:
                    raise ValueError(
                        "'image_url' input is not supported as the loaded "
                        "model is not multimodal."
                    )

                elif len(image_futures) == 0:
                    assert self.tokenizer is not None
                    image_url = cast(ChatCompletionContentPartImageParam, part)["image_url"]

                    if image_url.get("detail", "auto") != "auto":
                        logger.warning(
                            "'image_url.detail' is currently not supported and " "will be ignored."
                        )

                    file_data, mime_type = decode_base64(image_url["url"])

                    logger.debug(f"file_data: {type(file_data)}")
                    logger.debug(f"mime_type: {str(mime_type)}")

                    image_future: ImagePixelData = {
                        "image_pixel_data": file_data,
                        "mime_type": mime_type,
                    }

                    image_futures.append(image_future)
                else:
                    raise NotImplementedError(
                        "Multiple 'image_url' input is currently not supported."
                    )

            else:
                raise NotImplementedError(f"Unknown part type: {part_type}")

        text_prompt = "\n".join(texts)

        messages = [ConversationMessage(role=role, content=text_prompt)]

        logger.debug(f"messages: {str(messages)}")
        return ChatMessageParseResult(messages=messages, image_futures=image_futures)

    def _parse_chat_message_content(
        self,
        message: ChatCompletionMessageParam,
    ) -> ChatMessageParseResult:
        role = message["role"]
        content = message.get("content")

        # logger.debug(f"content: {str(content)}")

        if content is None:
            return ChatMessageParseResult(messages=[], image_futures=[])
        if isinstance(content, str):
            # logger.debug(f"Content")
            messages = [ConversationMessage(role=role, content=content)]
            return ChatMessageParseResult(messages=messages, image_futures=[])

        # logger.debug(f"ContentPart")
        return self._parse_chat_message_content_parts(role, content)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Optional[Request] = None
    ) -> Union[ErrorResponse, AsyncGenerator[str, None], ChatCompletionResponse]:

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"cmpl-{random_uuid()}"
        try:
            conversation: List[ConversationMessage] = []
            image_futures: List[Awaitable[ImagePixelData]] = []

            for msg in request.messages:
                # logger.debug(f"msg: {str(msg)}")
                chat_parsed_result = self._parse_chat_message_content(msg)

                conversation.extend(chat_parsed_result.messages)
                image_futures.extend(chat_parsed_result.image_futures)

            # print(conversation)
            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
            )
        except Exception as e:
            logger.error(f"Error in applying chat template from request: {str(e)}")
            return self.create_error_response(str(e))

        inputs: PromptInputs = {
            "prompt": prompt,
        }
        # print(image_futures)

        if self.vision:
            if len(image_futures) < 1:
                error_message = "No image has been provided."
                return self.create_error_response(str(error_message))
            inputs["multi_modal_data"] = image_futures

        result_generator = None
        if self.vision:
            result_generator = self.engine.generate_vision(
                inputs, request.to_sampling_params(), request_id, stream=request.stream
            )
        else:
            result_generator = self.engine.generate(
                inputs, request.to_sampling_params(), request_id, stream=request.stream
            )
        # Streaming response
        if request.stream:
            logger.error("stream: " + str(request.stream))
            return self.chat_completion_stream_generator(
                request, result_generator, request_id, conversation
            )
        else:
            # raise NotImplementedError("Not Yet Implemented Error")
            try:
                return await self.chat_completion_full_generator(
                    request, raw_request, result_generator, request_id, conversation
                )
            except ValueError as e:
                # TODO: Use a vllm-specific Validation Error
                return self.create_error_response(str(e))

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        conversation: List[ConversationMessage],
    ) -> AsyncGenerator[str, None]:
        model_name = self.served_model_name
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        assert request.n is not None
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        try:
            async for res in result_generator:
                logger.debug("res:" + str(res))
                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)
                    for i in range(request.n):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(role=role),
                            logprobs=None,
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                        )

                        # logger.debug("chunk: "+ str(chunk))

                        if request.stream_options and request.stream_options.include_usage:
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content = ""
                        if (
                            conversation
                            and conversation[-1].get("content")
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"]

                        if last_msg_content:
                            for i in range(request.n):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    logprobs=None,
                                    model=model_name,
                                )
                                if request.stream_options and request.stream_options.include_usage:
                                    chunk.usage = None
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    time.sleep(0.5)
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:

                        # @TODO: Add when ONNX support logits on DML
                        logprobs = None
                    else:
                        logprobs = None

                    # logger.debug("chunk: "+ str(chunk))

                    delta_text = output.text[len(previous_texts[i]) :]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)

                    delta_message = DeltaMessage(content=delta_text)

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n

                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i, delta=delta_message, logprobs=logprobs, finish_reason=None
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                        )
                        if request.stream_options and request.stream_options.include_usage:
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                    else:
                        # Send the finish response for each request.n only once
                        prompt_tokens = len(res.prompt_token_ids)
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                        )
                        if request.stream_options and request.stream_options.include_usage:
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                        # logger.debug("chunk: "+ str(chunk))
                        finish_reason_sent[i] = True

            if request.stream_options and request.stream_options.include_usage:
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=previous_num_tokens[i],
                    total_tokens=prompt_tokens + previous_num_tokens[i],
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"
                # logger.debug("final_usage_data: "+ str(final_usage_data))

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request],
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        conversation: List[ConversationMessage],
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = self.served_model_name
        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        choices: List[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            # token_ids = output.token_ids

            if request.logprobs and request.top_logprobs is not None:
                # @TODO: Add when ONNX support logits on DML
                logprobs = None
            else:
                logprobs = None

            message = ChatMessage(role=role, content=output.text)

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if (
                conversation
                and conversation[-1].get("content")
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1]["role"]

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> str:
        json_str = json.dumps(
            {
                "error": self.create_error_response(
                    message=message, err_type=err_type, status_code=status_code
                ).model_dump()
            }
        )
        return json_str

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(
                id=self.served_model_name,
                max_model_len=self.engine.max_model_len,
                root=self.served_model_name,
                permission=[ModelPermission()],
            )
        ]
        return ModelList(data=model_cards)

    async def _check_model(
        self, request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> Optional[ErrorResponse]:
        if request.model in [self.served_model_name]:
            return None
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
