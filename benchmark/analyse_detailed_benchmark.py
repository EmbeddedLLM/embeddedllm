import os
import re
import numpy as np

def extract_data_from_log(log_file):
    average_tps_list = []
    prompt_tokens_per_second_list = []
    new_tokens_per_second_list = []

    with open(log_file, 'r') as file:
        for line in file:
            if "Average tps" in line:
                average_tps = float(re.search(r"Average tps: ([\d.]+)", line).group(1))
                average_tps_list.append(average_tps)
            if "Prompt tokens per second" in line:
                prompt_tokens_per_second = float(re.search(r"Prompt tokens per second: ([\d.]+)", line).group(1))
                prompt_tokens_per_second_list.append(prompt_tokens_per_second)
            if "New tokens per second" in line:
                new_tokens_per_second = float(re.search(r"New tokens per second: ([\d.]+)", line).group(1))
                new_tokens_per_second_list.append(new_tokens_per_second)

    return average_tps_list, prompt_tokens_per_second_list, new_tokens_per_second_list

def print_statistics(data, label):
    data_np = np.array(data)
    print(f"{label} Statistics:")
    print(f"Standard Deviation: {np.std(data_np, ddof=1)}")  # Sample standard deviation
    print(f"Mean: {np.mean(data_np)}")
    print(f"Min: {np.min(data_np)}")
    print(f"1%: {np.percentile(data_np, 1)}")
    print(f"25%: {np.percentile(data_np, 25)}")
    print(f"50% (Median): {np.percentile(data_np, 50)}")
    print(f"75%: {np.percentile(data_np, 75)}")
    print(f"99%: {np.percentile(data_np, 99)}")
    print(f"Max: {np.max(data_np)}")
    print()

def main():
    model_path = "C:\\Users\\ryzzai\\Documents\\Phi-3-mini-4k-instruct-062024-int4\\onnx\\directml\\Phi-3-mini-4k-instruct-062024-int4"
    model_name = os.path.basename(model_path)
    backend = "cpu"
    input_token_length = 128
    output_token_length = 128
    log_file = f'profile_model_timing_{os.path.basename(model_path)}_{input_token_length}_{output_token_length}_{backend}.log'

    average_tps_list, prompt_tokens_per_second_list, new_tokens_per_second_list = extract_data_from_log(log_file)

    min_len = min(len(average_tps_list), min(len(prompt_tokens_per_second_list), len(new_tokens_per_second_list)))

    for i in range(min_len):
        print(f"Entry {i+1}:")
        print(f"Average TPS: {average_tps_list[i]}")
        print(f"Prompt Token / sec: {prompt_tokens_per_second_list[i]}")
        print(f"New Token / sec: {new_tokens_per_second_list[i]}")
        print()

    print("from log file: " + log_file)

    # Print statistics
    print_statistics(prompt_tokens_per_second_list[:min_len], "Prompt Token / sec")
    print_statistics(new_tokens_per_second_list[:min_len], "New Token / sec")
    print_statistics(average_tps_list[:min_len], "Average TPS")

if __name__ == "__main__":
    main()
