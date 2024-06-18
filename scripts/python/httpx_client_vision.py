import httpx
import json
from embeddedllm.protocol import ChatCompletionRequest, ChatCompletionMessageParam, CustomChatCompletionMessageParam

def chat_completion(url: str, payload: dict):
    with httpx.Client(timeout=None) as client:
        response = client.post(url, json=payload)
        if response.status_code == 200:
            print(response.text)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

# Example usage
if __name__ == "__main__":
    IMAGE_PATH="C:\\Users\\ryzz\\VDrive\\RyzenAI\\icons8-amd-ryzen-64.png"
    import base64
    import mimetypes

    # Function to encode the image and infer its MIME type
    def encode_image(image_path):
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            raise ValueError("Could not infer the MIME type of the image.")

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        return mime_type, base64_image

    # Getting the base64 string and MIME type
    mime_type, base64_image = encode_image(IMAGE_PATH)

    url = "http://localhost:6979/v1/chat/completions"
    # print(f"data:{mime_type};base64,{base64_image}")
    string_url = f"data:{mime_type};base64,{base64_image}"
    # data = ChatCompletionMessageParam(**payload["messages"])

    messages: CustomChatCompletionMessageParam = [
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
        "stream": False  # Set stream to False
    }
    # print(data)
    # print(messages)
    # print(data.messages[0].content[0])
    chat_completion(url, payload)