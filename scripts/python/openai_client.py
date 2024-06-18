# from openai import AsyncOpenAI
# import asyncio
# import time

# client = AsyncOpenAI(
#     base_url="http://localhost:6979/v1",
#     api_key='ellm'
# )


# async def main():
#     stream = await client.chat.completions.create(
#         model="phi3-mini-int4",
#         messages=[{"role": "user", "content": "Say this is a test"}],
#         max_tokens=80,
#         temperature=0,
#         stream=True,
#     )
#     print(stream)
#     async for chunk in stream:
#         print(chunk.choices[0].delta.content or "", end="", flush=True)


# asyncio.run(main())

from openai import OpenAI

client = OpenAI(base_url="http://localhost:6979/v1", api_key="ellm")

stream = client.chat.completions.create(
    model="phi3-mini-int4",
    messages=[{"role": "user", "content": "Say this is a test"}],
    max_tokens=80,
    temperature=0,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
