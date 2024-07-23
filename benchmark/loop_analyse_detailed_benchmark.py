import subprocess

model_names = [
    "Phi-3-medium-128k-instruct-onnx-directml",
    "Phi-3-medium-4k-instruct-onnx-directml",
    "Phi-3-mini-128k-instruct-onnx-directml",
    "Phi-3-mini-4k-instruct-062024-int4-directml",
    "Phi-3-mini-4k-instruct-onnx-directml",

    # "Phi-3-mini-4k-instruct-ipex",
    # "Phi-3-mini-128k-instruct-ipex",
    # "Phi-3-medium-4k-instruct-ipex",
    # "Phi-3-medium-128k-instruct-ipex",

    # "Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32",
    # "Phi-3-mini-128k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4",
    # "Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32",
    # "Phi-3-mini-4k-instruct-onnx-cpu-int4-rtn-block-32-acc-level-4",
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