# CHANGELOG

## [Unrelease]

### ADDED

DOC
- Update `README.md` to include usage of precompiled engine executable.

### CHANGES / FIXES

Engine
- Re-structure the configuration to specify which backend and device to launch the `ipex-llm` model.
- Fixed Non-Streaming Mode of ONNX is returning the Prompt in the Response #12

PyInstaller Executable
- Update the `ellm_api_server.spec` to support compilation of `ipex-llm` into executable. #14 