import litellm 

messages = [{"role": "user", "content": "Hey, how's it going?"}]

response = litellm.completion(
            model="phi3-mini-int4", # pass the vllm model name
            messages=messages,
            api_base="http://localhost:6979/v1",
            api_key="EMPTY",
            temperature=0,
            max_tokens=80, stream=True,
            custom_llm_provider="openai")

for part in response:
    print(part.choices[0].delta.content or "")