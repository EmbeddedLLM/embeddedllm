import subprocess
import os

# Define the models and token lengths
model_names = [
    # "Phi-3-medium-128k-instruct-onnx-directml",
    # "Phi-3-medium-4k-instruct-onnx-directml",
    # "Phi-3-mini-128k-instruct-onnx-directml",
    # "Phi-3-mini-4k-instruct-062024-int4-directml",
    # "Phi-3-mini-4k-instruct-onnx-directml",

    # "Phi-3-mini-4k-instruct-ipex",
    # "Phi-3-mini-128k-instruct-ipex",
    # "Phi-3-medium-4k-instruct-ipex",
    # "Phi-3-medium-128k-instruct-ipex",

    # "Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32",
    # "Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4",
    # "Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32",
    # "Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4",
]

model_paths = [
    # path to your model weight in the order with the model_names
]

token_in_out = [
    (1024, 1024),
    (1024, 512),
    (1024, 256),
    (1024, 128),
    (512, 1024),
    (512, 512),
    (512, 256),
    (512, 128),
    (256, 1024),
    (256, 512),
    (256, 256),
    (256, 128),
    (128, 1024),
    (128, 512),
    (128, 256),
    (128, 128),
]

# Choose backend
# backend = "ipex"
# backend = "directml"
# backend = "cpu"

# Number of loops
loop_count = 20

# Path to the ellm_benchmark.py script
ellm_benchmark_script = "ellm_benchmark.py"

for model_name, model_path in zip(model_names, model_paths):
    for input_token_length, output_token_length in token_in_out:
        # Construct the command
        command = [
            "python", ellm_benchmark_script,
            "--backend", backend,
            "--model_name", model_name,
            "--model_path", model_path,
            "--token_in", str(input_token_length),
            "--token_out", str(output_token_length),
            "--loop_count", str(loop_count)
        ]

        # Execute the command
        subprocess.run(command)
