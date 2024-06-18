import sys
import os
import io
import platform
from setuptools import setup
import re
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from typing import List
ROOT_DIR = os.path.dirname(__file__)

# Custom function to check for DirectML support
def check_directml_support():
    if platform.system() != "Windows":
        raise RuntimeError("This package requires a Windows system with DirectML support.")
    # Add additional checks for DirectML support if necessary

# Run the check before proceeding with the setup
check_directml_support()

ELLM_TARGET_DEVICE='cpu'

def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""
    
def _is_directml() -> bool:
    return ELLM_TARGET_DEVICE == "directml"

def _is_cpu() -> bool:
    return ELLM_TARGET_DEVICE == "cpu"

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

def _read_requirements(filename: str) -> List[str]:
    with open(get_path(filename)) as f:
        requirements = f.read().strip().split("\n")
    resolved_requirements = []
    for line in requirements:
        if line.startswith("-r "):
            resolved_requirements += _read_requirements(line.split()[1])
        else:
            resolved_requirements.append(line)
    return resolved_requirements

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""


    if _is_directml():
        requirements = _read_requirements("requirements-directml.txt")
    elif _is_cpu():
        requirements = _read_requirements("requirements-cpu.txt")
    else:
        raise ValueError(
            "Unsupported platform, please use CUDA, ROCm, Neuron, or CPU.")
    return requirements

def get_ellm_version() -> str:
    version = find_version(get_path("src", "embeddedllm", "version.py"))
    
    if _is_directml():
        version += "+directml"
    elif _is_cpu():
        version += "+cpu"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version
setup(
    name="embeddedllm",
    version=get_ellm_version(),
    author="Embedded LLM Team",
    license="Apache 2.0",
    description="EmbeddedLLM: API server for Embedded Device Deployment. Currently support ONNX-DirectML.",
    python_requires=">=3.8",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/EmbeddedLLM/embeddedllm.git",
    project_urls={
        "Homepage": "https://github.com/EmbeddedLLM/embeddedllm.git",
    },
    packages=find_packages(where='src', exclude=("benchmarks", "docs", "scripts",
                                    "tests*")),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=get_requirements().extend(_read_requirements("requirements-common.txt")),
    # Add other metadata and dependencies as needed
    extras_require={
        'lint': _read_requirements("requirements-lint.txt"),
    },
)