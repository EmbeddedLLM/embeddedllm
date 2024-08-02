import io
import os
import platform
import re
from typing import List

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess


ROOT_DIR = os.path.dirname(__file__)


ELLM_TARGET_DEVICE = os.environ.get("ELLM_TARGET_DEVICE", "directml")


# Custom function to check for DirectML support
def check_directml_support():
    if platform.system() != "Windows":
        raise RuntimeError("This package requires a Windows system with DirectML support.")
    # Add additional checks for DirectML support if necessary


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def _is_directml() -> bool:
    # Run the check before proceeding with the setup
    check_directml_support()
    return ELLM_TARGET_DEVICE == "directml"


def _is_cpu() -> bool:
    return ELLM_TARGET_DEVICE == "cpu"


def _is_cuda() -> bool:
    return ELLM_TARGET_DEVICE == "cuda"


def _is_ipex() -> bool:
    return ELLM_TARGET_DEVICE == "ipex"


class ELLMInstallCommand(install):
    def run(self):
        install.run(self)
        print("is_ipex(): " + str(_is_ipex()))
        if _is_ipex():
            print("Install Ipex-LLM")
            result = subprocess.run([
                'pip', 'install', '--pre', '--upgrade', 'ipex-llm[xpu]',
                '--extra-index-url', 'https://pytorch-extension.intel.com/release-whl/stable/xpu/us/'
            ], capture_output=True, text=True)

            result = subprocess.run([
                'pip', 'install', '--upgrade', 'transformers==4.43.3'
            ], capture_output=True, text=True)

        if _is_directml():
            result = subprocess.run([
                'conda', 'install', 'conda-forge::vs2015_runtime' , '-y'
            ], capture_output=True, text=True)

class ELLMDevelopCommand(develop):
    def run(self):
        develop.run(self)
        print("is_ipex(): " + str(_is_ipex()))
        if _is_ipex():
            print("Install Ipex-LLM")
            result = subprocess.run([
                'pip', 'install', '--pre', '--upgrade', 'ipex-llm[xpu]',
                '--extra-index-url', 'https://pytorch-extension.intel.com/release-whl/stable/xpu/us/'
            ], capture_output=True, text=True)
            
            result = subprocess.run([
                'pip', 'install', '--upgrade', 'transformers==4.43.3'
            ], capture_output=True, text=True)

        if _is_directml():
            result = subprocess.run([
                'conda', 'install', 'conda-forge::vs2015_runtime' , '-y'
            ], capture_output=True, text=True)
            
            print(result)

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M)
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
    elif _is_cuda():
        requirements = _read_requirements("requirements-cuda.txt")
    elif _is_cpu():
        requirements = _read_requirements("requirements-cpu.txt")
    elif _is_ipex():
        requirements = _read_requirements("requirements-ipex.txt")
    else:
        raise ValueError("Unsupported platform, please use CUDA, ROCm, Neuron, or CPU.")
    return requirements


def get_ellm_version() -> str:
    version = find_version(get_path("src", "embeddedllm", "version.py"))

    if _is_directml():
        version += "+directml"
    elif _is_cuda():
        version += "+cuda"
    elif _is_cpu():
        version += "+cpu"
    elif _is_ipex():
        version += "+ipex"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version


print(get_requirements().extend(_read_requirements("requirements-common.txt")))

dependency_links = []
extra_install_requires = []

if(_is_directml() or _is_cuda() or _is_cpu()):
    dependency_links.extend(["https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/"])
# elif(_is_ipex()):
#     dependency_links.extend(["https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"])
    # extra_install_requires.extend(["torch==2.1.0a0 @ https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"])
    # extra_install_requires.extend(["ipex-llm[xpu]==2.1.0b20240702 @ https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"])

    # extra_install_requires = ['ipex-llm[xpu]==2.1.0b20240702 @ https://pytorch-extension.intel.com/release-whl/stable/xpu/us/']

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
    packages=find_packages(where="src", exclude=("benchmarks", "docs", "scripts", "tests*")),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=get_requirements()
    + _read_requirements("requirements-common.txt")
    + _read_requirements("requirements-build.txt") + extra_install_requires,
    # Add other metadata and dependencies as needed
    extras_require={
        "lint": _read_requirements("requirements-lint.txt"),
        "webui": _read_requirements("requirements-webui.txt"),
        "cuda": ["onnxruntime-genai-cuda==0.3.0rc2"],
        "ipex": [],
    },
    dependency_links=dependency_links,
    entry_points={
        "console_scripts": [
            "ellm_server=embeddedllm.entrypoints.api_server:main",
            "ellm_chatbot=embeddedllm.entrypoints.webui:main",
            "ellm_modelui=embeddedllm.entrypoints.modelui:main",
        ],
    },
    cmdclass={
        'install': ELLMInstallCommand,
        'develop': ELLMDevelopCommand,
    },
)
