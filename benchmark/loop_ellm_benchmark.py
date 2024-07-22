import subprocess
import os

# Define the models and token lengths
model_names = [
    "Phi-3-mini-4k-instruct-062024-int4-directml",
    "Phi-3-mini-128k-instruct-onnx-directml",
    "Phi-3-medium-4k-instruct-onnx-directml",
    "Phi-3-medium-128k-instruct-onnx-directml",
]

model_paths = [

]

token_in_out = [
    # (128, 128),
    # (128, 256),
    (128, 512),
    (128, 1024),
    # (256, 128),
    # (256, 256),
    (256, 512),
    (256, 1024),
    (512, 128),
    (512, 256),
]

# Choose backend
# backend = "cpu"
backend = "directml"

# Number of loops
loop_count = 50

# Path to the ellm_benchmark.py script
ellm_benchmark_script = "ellm_benchmark.py"

for model_name, model_path in zip(model_names, model_paths):
    for input_token_length, output_token_length in token_in_out:
        for i in range(loop_count):
            # Construct the command
            command = [
                "python", ellm_benchmark_script,
                "--backend", backend,
                "--model_name", model_name,
                "--model_path", model_path,
                "--token_in", str(input_token_length),
                "--token_out", str(output_token_length)
            ]

            # Execute the command
            subprocess.run(command)
