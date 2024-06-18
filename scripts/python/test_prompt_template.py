MODEL_PATH='C:\\Users\\ryzz\\git\\cpu-int4-rtn-block-32-acc-level-4'


from transformers import AutoTokenizer
from embeddedllm.protocol import ChatCompletionRequest

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

prompt = tokenizer.apply_chat_template(
    conversation=[
        {
            "role": "user",
            "content": "What is in this image?",
        }
    ]
)

print(prompt)