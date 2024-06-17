import codecs
import time
from http import HTTPStatus
from typing import (AsyncGenerator, AsyncIterator, Awaitable, Dict, Iterable,
                    List, Optional, Union)
from typing import TypedDict, cast, final
import json

from fastapi import Request
from dataclasses import dataclass

from openai.types.chat import (ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartTextParam)
from embeddedllm.protocol import (  # noqa: E501
    ChatCompletionContentPartParam, ChatCompletionLogProb,
    ChatCompletionLogProbs, ChatCompletionLogProbsContent,
    ChatCompletionMessageParam, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    FunctionCall, ToolCall, UsageInfo, ChatCompletionRequest,
    CompletionRequest,
    ModelCard, ModelList,
    ModelPermission, RequestOutput, CompletionOutput)
from embeddedllm.engine import embeddedllmEngine
from embeddedllm.utils import random_uuid

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from loguru import logger

@final  # So that it should be compatible with Dict[str, str]
class ConversationMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]

class OpenAPIChatServer():

    def __init__(
        self,
        model_path: str,
        served_model_name: str = '',
        response_role: str = 'assistant',
        chat_template: Optional[str] = None
    ):
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.response_role = response_role
        self.engine = embeddedllmEngine(model_path)

        self.tokenizer=self.engine.tokenizer
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
                    msg = (f"The supplied chat template ({chat_template}) "
                           f"looks like a file path, but it failed to be "
                           f"opened. Reason: {e}")
                    raise ValueError(msg) from e

                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                tokenizer.chat_template = codecs.decode(
                    chat_template, "unicode_escape")

            logger.info("Using supplied chat template:\n%s",
                        tokenizer.chat_template)
        elif tokenizer.chat_template is not None:
            logger.info("Using default chat template:\n%s",
                        tokenizer.chat_template)
        else:
            logger.warning(
                "No chat template provided. Chat API will not work.")

    def create_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(message=message,
                             type=err_type,
                             code=status_code.value)
    
    async def _check_model(
        self, request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> Optional[ErrorResponse]:
        if request.model in [self.served_model_name]:
            return None
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND)            

    def _parse_chat_message_content_parts(
        self,
        role: str,
        parts: Iterable[ChatCompletionContentPartParam],
    ) -> ChatMessageParseResult:
        texts: List[str] = []

        for part in parts:
            part_type = part["type"]
            if part_type == "text":
                text = cast(ChatCompletionContentPartTextParam, part)["text"]

                texts.append(text)
            elif part_type == "image_url":
                raise NotImplementedError(
                    "'image_url' input is currently not supported."
                )

            else:
                raise NotImplementedError(f"Unknown part type: {part_type}")

        text_prompt = "\n".join(texts)

        messages = [ConversationMessage(role=role, content=text_prompt)]

        return ChatMessageParseResult(messages=messages)

    def _parse_chat_message_content(
        self,
        message: ChatCompletionMessageParam,
    ) -> ChatMessageParseResult:
        role = message["role"]
        content = message.get("content")

        if content is None:
            return ChatMessageParseResult(messages=[])
        if isinstance(content, str):
            messages = [ConversationMessage(role=role, content=content)]
            return ChatMessageParseResult(messages=messages)

        return self._parse_chat_message_content_parts(role, content)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
            ChatCompletionResponse]:

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"cmpl-{random_uuid()}"
        try:
            conversation: List[ConversationMessage] = []

            for msg in request.messages:
                chat_parsed_result = self._parse_chat_message_content(msg)

                conversation.extend(chat_parsed_result.messages)

            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
            )
        except Exception as e:
            logger.error("Error in applying chat template from request: %s", e)
            return self.create_error_response(str(e))


        # Streaming response
        if request.stream:
            result_generator = self.engine.generate_stream(
                    prompt, request.to_sampling_params(), request_id)
            # logger.debug("result_generator: " + str(result_generator) )
            return self.chat_completion_stream_generator(
                request, result_generator, request_id, conversation)
        else:
            raise NotImplementedError("Not Yet Implemented Error")
            # try:
            #     return await self.chat_completion_full_generator(
            #         request, raw_request, result_generator, request_id,
            #         conversation)
            # except ValueError as e:
            #     # TODO: Use a vllm-specific Validation Error
            #     return self.create_error_response(str(e))

    async def chat_completion_stream_generator(
            self, request: ChatCompletionRequest,
            result_generator: AsyncIterator[RequestOutput], request_id: str,
            conversation: List[ConversationMessage]
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
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        
                        # logger.debug("chunk: "+ str(chunk))

                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content = ""
                        if conversation and conversation[-1].get(
                                "content") and conversation[-1].get(
                                    "role") == role:
                            last_msg_content = conversation[-1]["content"]

                        if last_msg_content:
                            for i in range(request.n):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    logprobs=None,
                                    model=model_name)
                                if (request.stream_options and
                                        request.stream_options.include_usage):
                                    chunk.usage = None
                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:

                        # @TODO: Add when ONNX support logits on DML
                        logprobs = None
                    else:
                        logprobs = None

                    # logger.debug("chunk: "+ str(chunk))

                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)

                    delta_message = DeltaMessage(content=delta_text)

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n

                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if (request.stream_options
                                and request.stream_options.include_usage):
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
                            stop_reason=output.stop_reason)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                        # logger.debug("chunk: "+ str(chunk))
                        finish_reason_sent[i] = True

            if (request.stream_options
                    and request.stream_options.include_usage):
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
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"
                # logger.debug("final_usage_data: "+ str(final_usage_data))

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"            


    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1]["role"]

    def create_streaming_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
        json_str = json.dumps({
            "error":
            self.create_error_response(message=message,
                                       err_type=err_type,
                                       status_code=status_code).model_dump()
        })
        return json_str
    
    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(id=self.served_model_name,
                      max_model_len=self.engine.max_model_len,
                      root=self.served_model_name,
                      permission=[ModelPermission()])
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
            status_code=HTTPStatus.NOT_FOUND)