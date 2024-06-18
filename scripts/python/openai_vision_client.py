from openai import AsyncOpenAI
import asyncio
import base64
import mimetypes
import os

current_file_path = os.path.abspath(__file__)
IMAGE_PATH = os.path.join(os.path.dirname(current_file_path), "..", "images", "catdog.png")


# Function to encode the image and infer its MIME type
def encode_image(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Could not infer the MIME type of the image.")

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    return mime_type, base64_image


# Getting the base64 string and MIME type
mime_type, base64_image = encode_image(IMAGE_PATH)

url = "http://localhost:6979/v1/chat/completions"
# print(f"data:{mime_type};base64,{base64_image}")
string_url = f"data:{mime_type};base64,{base64_image}"
# data = ChatCompletionMessageParam(**payload["messages"])

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": string_url,
                },
            },
        ],
    }
]

client = AsyncOpenAI(base_url="http://localhost:6979/v1", api_key="ellm")


async def main():
    stream = await client.chat.completions.create(
        model="phi3-mini-int4",
        messages=messages,
        max_tokens=80,
        temperature=0,
        stream=True,
    )
    print(stream)
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


asyncio.run(main())

# from openai import OpenAI

# client = OpenAI(base_url="http://localhost:6979/v1", api_key="ellm")

# stream = client.chat.completions.create(
#     model="phi3-mini-int4",
#     messages=[{"role": "user", "content": "Say this is a test"}],
#     max_tokens=80,
#     temperature=0,
#     stream=True,
# )
# for chunk in stream:
#     print(chunk.choices[0].delta.content or "", end="")
