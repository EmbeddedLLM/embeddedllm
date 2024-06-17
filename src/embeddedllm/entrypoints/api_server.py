import os
import codecs
import time
from http import HTTPStatus
from typing import (AsyncGenerator, AsyncIterator, Awaitable, Dict, Iterable,
                    List, Optional, Union)
from typing import TypedDict, cast, final
import json

import uvicorn
from fastapi.exceptions import RequestValidationError
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
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
from embeddedllm.entrypoints.chat_server import OpenAPIChatServer

from transformers import AutoConfig, PretrainedConfig
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
import argparse

app = FastAPI()

openai_chat_server: OpenAPIChatServer

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    port: int = 6979
    host: str = "0.0.0.0"
    response_role: str='assistant'
    uvicorn_log_level: str = 'info'
    served_model_name: str = 'phi3-mini-int4'
    model_path: str = None

config = Config()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_chat_server.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_chat_server.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_chat_server.show_available_models()
    return JSONResponse(content=models.model_dump())

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_chat_server.create_chat_completion(
        request, raw_request)
    print(generator)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content="Non-streaming Chat Generation Yet to be Implemented.",
                            status_code=404)
        # assert isinstance(generator, ChatCompletionResponse)
        # return JSONResponse(content=generator.model_dump())

if __name__ == "__main__":
    import uvicorn
    import os

    # if os.name == "nt":
    #     from multiprocessing import freeze_support

    #     freeze_support()
    #     print("The system is Windows.")
    # else:
    #     print("The system is not Windows.")

    openai_chat_server = OpenAPIChatServer(
        config.model_path, 
        served_model_name=config.served_model_name, 
        response_role=config.response_role
    )
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.uvicorn_log_level,
        loop='asyncio'
    )