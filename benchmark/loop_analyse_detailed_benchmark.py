import subprocess

model_names = [
    # model names
    
]


# Path to the ellm_benchmark.py script
analyse_detailed_benchmark_script = "analyse_detailed_benchmark.py"

for model_name in model_names:
    # Construct the command
    command = [
        "python", analyse_detailed_benchmark_script,
        "--model_name", model_name,
    ]

    # Execute the command
    subprocess.run(command)