import asyncio

import httpx
import json

def parse_stream(stream:str):

    stream = stream.replace('data: ', '')

    response_obj = json.loads(stream)
    # print(response_obj)

    return response_obj

async def stream_chat_completion(url: str, payload: dict):
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code == 200:
                async for data in response.aiter_bytes():
                    if data:
                        decodes_stream = data.decode("utf-8")
                        if "[DONE]" in decodes_stream:
                            continue
                        resp = parse_stream(decodes_stream)
                        if resp["choices"][0]["delta"].get('content', None):
                            print(resp["choices"][0]["delta"]["content"], end='', flush=True)
                        
                        # time.sleep(1)
            else:
                print(f"Error: {response.status_code}")
                print(await response.text())


# Example usage
if __name__ == "__main__":
    url = "http://localhost:6979/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": "What is the fastest bird on earth?"}],
        "model": "phi3-mini-int4",
        "max_tokens": 200,
        "temperature": 0.0,
        "stream": True,
    }
    asyncio.run(stream_chat_completion(url, payload))
