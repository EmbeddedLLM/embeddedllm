import asyncio
import os
import base64
import mimetypes
import httpx


async def stream_chat_completion(url: str, payload: dict):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code == 200:
                async for data in response.aiter_bytes():
                    if data:
                        print(data.decode("utf-8"))
                        # time.sleep(0.1)
            else:
                print(f"Error: {response.status_code}")
                print(await response.text())


# Example usage
if __name__ == "__main__":
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

    payload = {
        "messages": messages,
        "model": "phi3-mini-int4",
        "max_tokens": 80,
        "temperature": 0.0,
        "stream": True,
    }
    asyncio.run(stream_chat_completion(url, payload))
