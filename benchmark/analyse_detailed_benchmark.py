import os
import re
import numpy as np
import pandas as pd

def extract_data_from_log(log_file):
    average_tps_list = []
    prompt_tokens_per_second_list = []
    new_tokens_per_second_list = []
    error_count = 0
    error_state = False

    if not os.path.exists(log_file):
        print(f"Log file does not exist: {log_file}")
        return average_tps_list, prompt_tokens_per_second_list, new_tokens_per_second_list, error_count

    with open(log_file, 'r') as file:
        for line in file:
            if "ERROR" in line:
                error_count += 1
                error_state = True
                continue

            if "Average tps" in line and error_state == True:
                error_state = False
                continue
                
            if "Average tps" in line:
                average_tps = float(re.search(r"Average tps: ([\d.]+)", line).group(1))
                average_tps_list.append(average_tps)
                continue

            if "Prompt tokens per second" in line:
                prompt_tokens_per_second = float(re.search(r"Prompt tokens per second: ([\d.]+)", line).group(1))
                prompt_tokens_per_second_list.append(prompt_tokens_per_second)
            if "New tokens per second" in line:
                new_tokens_per_second = float(re.search(r"New tokens per second: ([\d.]+)", line).group(1))
                new_tokens_per_second_list.append(new_tokens_per_second)

    return average_tps_list, prompt_tokens_per_second_list, new_tokens_per_second_list, error_count

def calculate_statistics(data):
    data_np = np.array(data)
    stats = {
        "std": np.std(data_np, ddof=1),  # Sample standard deviation
        "mean": np.mean(data_np),
        "min": np.min(data_np),
        "1%": np.percentile(data_np, 1),
        "25%": np.percentile(data_np, 25),
        "50%": np.percentile(data_np, 50),  # Median
        "75%": np.percentile(data_np, 75),
        "99%": np.percentile(data_np, 99),
        "max": np.max(data_np)
    }
    return stats

def main():
    model_name = "Phi-3-mini-4k-instruct-062024-onnx-directml"
    token_ins = [128, 256, 512, 1024]
    token_outs = [128, 256, 512, 1024]

    statistics = []

    # Create the profile_model_timing directory if it doesn't exist
    log_dir = "profile_model_timing"
    os.makedirs(log_dir, exist_ok=True)

    for input_token_length in token_ins:
        for output_token_length in token_outs:
            log_file = os.path.join(log_dir, f'profile_model_timing_{model_name}_{input_token_length}_{output_token_length}.log')
            average_tps_list, prompt_tokens_per_second_list, new_tokens_per_second_list, error_count = extract_data_from_log(log_file)

            if not average_tps_list and not prompt_tokens_per_second_list and not new_tokens_per_second_list:
                # Log file does not exist or is empty, append "-" for each statistical value
                statistics.append([
                    model_name, input_token_length, output_token_length,
                    "-", "-", "-", "-", "-", "-", "-", "-", "-",
                    "-", "-", "-", "-", "-", "-", "-", "-", "-",
                    "-", "-", "-", "-", "-", "-", "-", "-", "-",
                    error_count
                ])
            else:
                min_len = min(len(average_tps_list), len(prompt_tokens_per_second_list), len(new_tokens_per_second_list))

                if min_len > 0:
                    prompt_stats = calculate_statistics(prompt_tokens_per_second_list[:min_len])
                    new_token_stats = calculate_statistics(new_tokens_per_second_list[:min_len])
                    average_tps_stats = calculate_statistics(average_tps_list[:min_len])

                    statistics.append([
                        model_name, input_token_length, output_token_length,
                        prompt_stats["std"], prompt_stats["mean"], prompt_stats["min"], prompt_stats["1%"], prompt_stats["25%"], prompt_stats["50%"], prompt_stats["75%"], prompt_stats["99%"], prompt_stats["max"],
                        new_token_stats["std"], new_token_stats["mean"], new_token_stats["min"], new_token_stats["1%"], new_token_stats["25%"], new_token_stats["50%"], new_token_stats["75%"], new_token_stats["99%"], new_token_stats["max"],
                        average_tps_stats["std"], average_tps_stats["mean"], average_tps_stats["min"], average_tps_stats["1%"], average_tps_stats["25%"], average_tps_stats["50%"], average_tps_stats["75%"], average_tps_stats["99%"], average_tps_stats["max"],
                        error_count
                    ])

    # Create a DataFrame
    columns = [
        "Model", "Token In", "Token Out",
        "Token In / sec std", "Token In / sec mean", "Token In / sec min", "Token In / sec 1%", "Token In / sec 25%", "Token In / sec 50%", "Token In / sec 75%", "Token In / sec 99%", "Token In / sec max",
        "Token Out / sec std", "Token Out / sec mean", "Token Out / sec min", "Token Out / sec 1%", "Token Out / sec 25%", "Token Out / sec 50%", "Token Out / sec 75%", "Token Out / sec 99%", "Token Out / sec max",
        "Average Token / sec std", "Average Token / sec mean", "Average Token / sec min", "Average Token / sec 1%", "Average Token / sec 25%", "Average Token / sec 50%", "Average Token / sec 75%", "Average Token / sec 99%", "Average Token / sec max",
        "No of Fail"
    ]
    df = pd.DataFrame(statistics, columns=columns)

    # Create the statistics directory if it doesn't exist
    output_dir = "statistics"
    os.makedirs(output_dir, exist_ok=True)

    # Write to Excel
    output_file = os.path.join(output_dir, f"{model_name}_statistics.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Statistics written to {output_file}")

if __name__ == "__main__":
    main()
