# Changelog

## [Unrelease]

### CHANGES / FIXES

Engine
1. Re-structure the configuration to specify which backend and device to launch the `ipex-llm` model.
2. Fixed Non-Streaming Mode of ONNX is returning the Prompt in the Response #12

PyInstaller Executable
1. Update the `ellm_api_server.spec` to support compilation of `ipex-llm` into executable. #14 