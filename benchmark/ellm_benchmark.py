import sys
import os
import time
import asyncio
import argparse
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
        device="cpu"
    elif backend == "ipex":
        device="xpu"
    elif backend == "openvino":
        device="gpu"
    elif backend == "directml":
        device = ""

    model = engine.EmbeddedLLMEngine(model_path=model_path, vision=False, device=device, backend=backend)

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

def main():
    parser = argparse.ArgumentParser(description="Benchmark EmbeddedLLM models.")
    parser.add_argument('--backend', type=str, required=True, choices=['cpu', 'directml', 'openvino', 'ipex'], help='Backend to use (cpu, ipex, openvino or directml)')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--token_in', type=int, required=True, help='Number of input tokens (max 2048)')
    parser.add_argument('--token_out', type=int, required=True, help='Number of output tokens')

    args = parser.parse_args()

    # Cap the input tokens to 2048
    if args.token_in > 2048:
        print("Input tokens capped to 2048.")
        args.token_in = 2048

    # Run the async function using asyncio.run()
    asyncio.run(benchmark(args.token_in, args.token_out, args.model_path, args.model_name, args.backend))

if __name__ == "__main__":
    main()