# CHANGELOG

## [Unreleased]

## ADDED

Engine

- Added OpenVINO support. #19

### CHANGES / FIXES

Ipex-LLM Engine

- Model generation does not adhere to the max_tokens params. #20

## [v0.2.0a]

### ADDED

DOC

- Update `README.md` to include usage of precompiled engine executable.

### CHANGES / FIXES

Installation

- Fixed the `ipex-llm` pypi library version.

Engine

- Re-structure the configuration to specify which backend and device to launch the `ipex-llm` model.
- Fixed Non-Streaming Mode of ONNX is returning the Prompt in the Response #12

PyInstaller Executable

- Update the `ellm_api_server.spec` to support compilation of `ipex-llm` into executable. #14
