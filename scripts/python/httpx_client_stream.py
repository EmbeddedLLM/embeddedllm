import asyncio

import httpx


async def stream_chat_completion(url: str, payload: dict):
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code == 200:
                async for data in response.aiter_bytes():
                    if data:
                        print(data.decode("utf-8"))
                        # time.sleep(1)
            else:
                print(f"Error: {response.status_code}")
                print(await response.text())


# Example usage
if __name__ == "__main__":
    url = "http://localhost:6979/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": "Hello!"}],
        "model": "phi3-mini-int4",
        "max_tokens": 80,
        "temperature": 0.0,
        "stream": True,
    }
    asyncio.run(stream_chat_completion(url, payload))
