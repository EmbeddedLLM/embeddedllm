from transformers import AutoTokenizer

MODEL_PATH = "C:\\Users\\ryzz\\git\\cpu-int4-rtn-block-32-acc-level-4"

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
