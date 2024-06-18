from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic_settings import BaseSettings, SettingsConfigDict

from embeddedllm.entrypoints.chat_server import OpenAPIChatServer
from embeddedllm.protocol import (  # noqa: E501
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)

app = FastAPI()

openai_chat_server: OpenAPIChatServer


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    port: int = 6979
    host: str = "0.0.0.0"
    response_role: str = "assistant"
    uvicorn_log_level: str = "info"
    served_model_name: str = "phi3-mini-int4"
    model_path: str = None
    vision: bool = False


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
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):

    generator = await openai_chat_server.create_chat_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        # return JSONResponse(content="Non-streaming Chat Generation Yet to be Implemented.",
        #                     status_code=404)
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    import os

    import uvicorn

    if os.name == "nt":
        import asyncio
        from multiprocessing import freeze_support

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        freeze_support()
        print("The system is Windows.")
    else:
        print("The system is not Windows.")

    openai_chat_server = OpenAPIChatServer(
        config.model_path,
        served_model_name=config.served_model_name,
        response_role=config.response_role,
        vision=config.vision,
    )

    uvicorn.run(
        app, host=config.host, port=config.port, log_level=config.uvicorn_log_level, loop="asyncio"
    )
