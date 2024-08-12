# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files
import importlib.metadata
import re
import sys
import os

CONDA_PATH=Path(sys.executable)
print(dir(CONDA_PATH))
print(CONDA_PATH)
print(CONDA_PATH.parent)

excluded_modules = ['torch.distributions'] # <<< ADD THIS LINE

def get_embeddedllm_backend():
    try:
        # Get the version of embeddedllm
        version = importlib.metadata.version("embeddedllm")

        # Use regex to extract the backend
        match = re.search(r"\+(directml|cpu|cuda|ipex|openvino)$", version)

        if match:
            backend = match.group(1)
            return backend
        else:
            return "Unknown backend"

    except importlib.metadata.PackageNotFoundError:
        return "embeddedllm not installed"


backend = get_embeddedllm_backend()

binaries_list = []

datas_list = [
    (Path("src/embeddedllm/entrypoints/api_server.py").resolve().as_posix(), 'embeddedllm/entrypoints'),
]
datas_list.extend(collect_data_files('torch', include_py_files=True))

hiddenimports_list = ['multipart']

pathex = []

def add_package(package_name):
    datas, binaries, hiddenimports = collect_all(package_name)
    datas_list.extend(datas)
    binaries_list.extend(binaries)
    hiddenimports_list.extend(hiddenimports)

if backend in ('directml', 'cpu', 'cuda'):
    add_package('onnxruntime')
    add_package('onnxruntime_genai')

elif backend == 'ipex':
    print(f"Backend IPEX")
    add_package('ipex_llm')
    add_package('torch')
    add_package('torchvision')
    add_package('intel_extension_for_pytorch')
    add_package('trl')
    add_package('embeddedllm')
    add_package('numpy')
    binaries_list.append((f'{CONDA_PATH.parent}/Library/bin/*', '.'))

elif backend == 'openvino':
    print(f"Backend OpenVino")
    add_package('onnx')
    add_package('torch')
    add_package('torchvision')
    add_package('optimum')
    add_package('optimum.intel')
    add_package('embeddedllm')
    add_package('numpy')
    add_package('openvino')
    add_package('openvino-genai')
    add_package('openvino-telemetry')
    add_package('openvino-tokenizers')
    binaries_list.append((f'{CONDA_PATH.parent}/Library/bin/*', '.'))
    binaries_list.extend([
        (Path('C:\\Windows\\System32\\libomp140.x86_64.dll').as_posix(), '.'),
        (Path('C:\\Windows\\System32\\libomp140d.x86_64.dll').as_posix(), '.'),
    ])

print(binaries_list)

with open("binary.txt", 'w') as f:
    f.write(str(binaries_list))
block_cipher = None
a = Analysis(
    ['src\\embeddedllm\\entrypoints\\api_server.py'],             
    pathex=pathex,
    binaries=binaries_list,
    datas=datas_list,
    hiddenimports=hiddenimports_list,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    block_cipher=block_cipher,
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ellm_api_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ellm_api_server',
)
