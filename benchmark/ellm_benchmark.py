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

async def benchmark(input_token_length, output_token_length, model, PromptInputs, sampling_params_config):

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
    logger.info("Average tps: " + str(average_tps))


def main():
    parser = argparse.ArgumentParser(description="Benchmark EmbeddedLLM models.")
    parser.add_argument('--backend', type=str, required=True, choices=['cpu', 'directml', 'ipex'], help='Backend to use (cpu, ipex or directml)')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--token_in', type=int, required=True, help='Number of input tokens (max 2048)')
    parser.add_argument('--token_out', type=int, required=True, help='Number of output tokens')
    parser.add_argument('--loop_count', type=int, required=True, help='Number of loop')

    args = parser.parse_args()

    # Create the profile_model_timing directory if it doesn't exist
    log_dir = "profile_model_timing"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'profile_model_timing_{args.model_name}_{args.token_in}_{args.token_out}.log')

    # Add the log file to the logger (it will append if the file already exists)
    logger.add(log_file, mode='a')

    prompt_text = """
    """
    # Define the path to the file
    file_path = "sampleText.txt"

    # Open the file and read its contents into the variable
    with open(file_path, 'r') as file:
        prompt_text = file.read()

    PromptInputs = {
        "prompt": prompt_text
    }

    sampling_params_config = sampling_params.SamplingParams(
        max_tokens=args.token_out,
        top_p=0.1,
        top_k=1,
        temperature=1,
        repetition_penalty=0.01,
    )

    logger.info(f"Model: {args.model_name}")

    # need different parameter for cpu and directml
    if args.backend == "cpu":
        model = engine.EmbeddedLLMEngine(args.model_path, vision=False, device="cpu", backend=args.backend)
    elif args.backend == "ipex":
        model = engine.EmbeddedLLMEngine(args.model_path, vision=False, device="xpu", backend=args.backend)
    else:
        model = engine.EmbeddedLLMEngine(args.model_path, vision=False, device="", backend=args.backend)

    model.tokenizer.chat_template = "{% for message in messages %}{{  message['content']}}{% endfor %}"  # Override

    input_tokens = model.tokenizer.encode(prompt_text)[:args.token_in-1]
    input_text = model.tokenizer.decode(input_tokens)
    print(input_text)
    input_tokens = model.tokenizer.encode(input_text)
    print(len(input_tokens))

    assert args.token_in-1 == len(input_tokens)

    for _ in range(args.loop_count):
        # Run the async function using asyncio.run()
        asyncio.run(benchmark(args.token_in, args.token_out, model, PromptInputs, sampling_params_config))

    # Remove the logger to close the log file
    logger.remove()


if __name__ == "__main__":
    main()
