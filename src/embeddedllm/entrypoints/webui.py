import base64
import json
import mimetypes

import gradio as gr
import httpx
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_stream(stream: str):
    stream = stream.replace("data: ", "")

    response_obj = json.loads(stream)

    return response_obj


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", cli_parse_args=True
    )
    port: int = Field(default=7788, description="Gradio port.")
    host: str = Field(default="127.0.0.1", description="Gradio host.")
    server_port: int = Field(default=6979, description="ELLM Server port.")
    server_host: str = Field(default="localhost", description="ELLM Server host.")


config = Config()
URL = f"http://{config.server_host}:{config.server_port}/v1"

# Gradio Variables
temperature = gr.Slider(
    0,
    1,
    value=0.1,
    step=0.05,
    label="Temperature",
    info="Choose between 0 and 1. 0 means Greedy Search",
    render=False,
    interactive=True,
)
top_p = gr.Slider(
    0,
    1,
    value=0.1,
    step=0.05,
    label="Top P",
    info="Choose between 0 and 1.",
    render=False,
    interactive=True,
)
top_k = gr.Slider(
    1,
    30,
    value=1,
    step=1,
    label="Top K",
    info="Choose between 1 and 30.",
    render=False,
    interactive=True,
)
output_token = gr.Slider(
    10,
    512,
    value=50,
    step=1,
    label="Output Tokens",
    info="Choose between 10 and 512.",
    render=False,
    interactive=True,
)


def convert_to_openai_image_url(image_path: str):
    # Function to encode the image and infer its MIME type
    def encode_image(image_path):
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            raise ValueError("Could not infer the MIME type of the image.")

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        return mime_type, base64_image

    # Getting the base64 string and MIME type
    mime_type, base64_image = encode_image(image_path)

    string_url = f"data:{mime_type};base64,{base64_image}"

    return string_url


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"])


async def bot(history, temperature, top_p, top_k, output_token):
    history_openai_format = []
    latest_image = None
    for human, assistant in history:
        if isinstance(human, tuple):  # this an image
            latest_image = ChatCompletionContentPartImageParam(
                image_url={
                    "url": convert_to_openai_image_url(human[0]),
                },
                type="image_url",
            )
            continue

        history_openai_format.append(
            {
                "role": "user",
                "content": [ChatCompletionContentPartTextParam(text=human, type="text")],
            }
        )

        if assistant:
            history_openai_format.append(
                {
                    "role": "assistant",
                    "content": [ChatCompletionContentPartTextParam(text=assistant, type="text")],
                }
            )

    if latest_image:
        history_openai_format[-1]["content"].append(latest_image)

    # print(history_openai_format)
    with open("debug_openai_history.txt", "w") as f:
        f.write(str(history_openai_format))

    url = f"{URL}/chat/completions"
    payload = {
        "messages": history_openai_format,
        "model": "phi3-mini-int4",
        "max_tokens": output_token,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "stream": True,
    }

    history[-1][1] = ""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=json.loads(json.dumps(payload))) as response:
            if response.status_code == 200:
                async for data in response.aiter_bytes():
                    if data:
                        decodes_stream = data.decode("utf-8")
                        if "[DONE]" in decodes_stream:
                            continue
                        resp = parse_stream(decodes_stream)
                        if resp["choices"][0]["delta"].get("content", None):
                            history[-1][1] += resp["choices"][0]["delta"]["content"]
                            yield history

            else:
                print(f"Error: {response.status_code}")
                yield history


def main():
    with gr.Blocks(
        title="EmbeddedLLM Chatbot",
        theme="freddyaboulton/dracula_revamped",
    ) as demo:
        with gr.Row():
            with gr.Column(scale=3, min_width=600):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    show_copy_button=True,
                    avatar_images=[None, "https://i.ibb.co/CzpJcYG/jamai-logo.png"],
                )

                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_types=["image"],
                    placeholder="Enter message or upload file...",
                    show_label=False,
                )

                chat_msg = chat_input.submit(
                    add_message, [chatbot, chat_input], [chatbot, chat_input]
                )
                bot_msg = chat_msg.then(
                    bot,
                    inputs=[chatbot, temperature, top_p, top_k, output_token],
                    outputs=chatbot,
                    api_name="bot_response",
                )
                bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

                chatbot.like(print_like_dislike, None, None)

            with gr.Column(scale=1, min_width=300, variant="compact"):
                temperature.render()
                top_p.render()
                top_k.render()
                output_token.render()

    demo.queue()
    demo.launch(server_port=config.port, server_name=config.host)


if __name__ == "__main__":
    main()
