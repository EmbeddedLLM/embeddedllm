import subprocess

# Define the models
model_names = [
    # model names
    "Phi-3-mini-4k-instruct-int8-ov"
]

# Define the model paths
model_paths = [
    # path to model in order to model names / model repo id
    "OpenVINO/Phi-3-mini-4k-instruct-int8-ov"
]

# Define the token length
token_in_out = [
    # (1024, 1024),
    # (1024, 512),
    # (1024, 256),
    # (1024, 128),
    # (512, 1024),
    # (512, 512),
    # (512, 256),
    # (512, 128),
    # (256, 1024),
    # (256, 512),
    # (256, 256),
    # (256, 128),
    # (128, 1024),
    # (128, 512),
    # (128, 256),
    (128, 128),
]

# Choose backend
# backend = "cpu"
# backend = "directml"
# backend = "ipex"
backend = "openvino"

# Number of loops
loop_count = 20

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
