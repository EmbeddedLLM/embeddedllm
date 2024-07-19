import sys
import os
import time
import asyncio
from loguru import logger

# Add the 'src' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the engine module
from embeddedllm import engine
from embeddedllm import sampling_params

async def benchmark(input_token_length, output_token_length, model_path, model_name, backend):
    # Create the profile_model_timing directory if it doesn't exist
    log_dir = "profile_model_timing"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'profile_model_timing_{model_name}_{input_token_length}_{output_token_length}.log')

    # Add the log file to the logger (it will append if the file already exists)
    logger.add(log_file, mode='a')

    # need different parameter for cpu and directml
    if backend == "cpu":
        model = engine.EmbeddedLLMEngine(model_path, vision=False, device="cpu", backend=backend)
    else:
        model = engine.EmbeddedLLMEngine(model_path, vision=False, device="AMD", backend=backend)

    logger.info(f"Model: {model_name}")

    model.tokenizer.chat_template = "{% for message in messages %}{{  message['content']}}{% endfor %}"  # Override

    prompt_text = """

    """
    # Define the path to the file
    file_path = "sampleText.txt"

    # Open the file and read its contents into the variable
    with open(file_path, 'r') as file:
        prompt_text = file.read()

    input_tokens = model.tokenizer.encode(prompt_text)[:input_token_length-1]
    input_text = model.tokenizer.decode(input_tokens)
    print(input_text)
    input_tokens = model.tokenizer.encode(input_text)
    print(len(input_tokens))

    assert input_token_length-1 == len(input_tokens)

    PromptInputs = {
        "prompt": input_text
    }

    sampling_params_config = sampling_params.SamplingParams(
        max_tokens=output_token_length,
        top_p=0.1,
        top_k=1,
        temperature=1,
        repetition_penalty=0.01,
    )

    start = time.perf_counter()

    async def generate():
        results = []
        async for response in model.generate(
            inputs=PromptInputs,
            sampling_params=sampling_params_config,
            request_id="benchmark",
            stream=True,
        ):
            results.append(response)
        return results

    response = await generate()
    end = time.perf_counter()

    logger.info(response[0])  # Access the generated text from the response

    total_time_taken = end - start
    logger.info(f"Total time taken: {total_time_taken:.2f} seconds")

    average_tps = (input_token_length + output_token_length) / total_time_taken
    logger.info("Average tps: "+ str(average_tps))

    # Remove the logger to close the log file
    logger.remove()

token_in_out = [
                (128,256),
                (128,512)
                ]

model_names = [
            "Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32",
            "Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4"
            ]

model_paths = [
    "C:\\Users\\hpamd\\Documents\\Phi-3-mini-4k-instruct-onnx\\cpu_and_mobile\\cpu-int4-rtn-block-32",
    "C:\\Users\\hpamd\\Documents\\Phi-3-mini-4k-instruct-onnx\\cpu_and_mobile\\cpu-int4-rtn-block-32-acc-level-4"
]

# choose cpu or directml for backend
backend = "cpu"
# backend = "directml"

for j in range(len(model_names)):
    model_name = model_names[j]
    model_path = model_paths[j]
    for input_token_length, output_token_length in token_in_out:
        for i in range(50):
            # Run the async function using asyncio.run()
            asyncio.run(benchmark(input_token_length, output_token_length, model_path, model_name, backend))
