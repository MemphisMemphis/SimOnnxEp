

# Simulator Execution Provider for ONNX Runtime

A custom execution provider (EP) that simulates on-device inference for testing
and development with ONNX Runtime.

## Repository Layout

- `src/` — plugin EP source (`simulator_ep.cc`)
- `test/` — gtest-based tests
- `third-parties/googletest/` — vendored GoogleTest v1.14.0 (CMake FetchContent)
- `onnxruntime/` — ONNX Runtime source tree (git submodule)

## Prerequisites

- CMake ≥ 3.18
- C/C++ compiler with C++17 support
- Python 3 (for generating test models)
- The `onnxruntime/` git submodule must be initialized.

## ONNX Runtime

The ONNX Runtime source is included as a git submodule at `onnxruntime/`.
CMake invokes its own `build.sh` to produce a shared library
(`libonnxruntime.so`). The build runs automatically on the first `cmake` configure
or when the library is missing; the first build is slow.

To force a rebuild after updating the submodule, delete
`onnxruntime/build/Linux/<config>/libonnxruntime.so`.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 4
```

This produces:

- `build/libsimulator_ep.so` — the plugin EP
- `build/run_sim` — standalone test host
- `build/test/simulator_ep_tests` — combined gtest binary

## Testing

Tests use GoogleTest (vendored in `third-parties/googletest/` and consumed via
CMake FetchContent). All test cases are compiled into a single
`simulator_ep_tests` executable and registered as one CTest entry.

```bash
cd build
ctest -R simulator_ep_tests --output-on-failure
```

Or run the binary directly:

```bash
./test/simulator_ep_tests
# Or filter individual cases:
./test/simulator_ep_tests --gtest_filter=SimulatorEP.AddModel
```

## Usage (Python)

```python
import onnxruntime as ort

# Load the custom EP shared library
session_options = ort.SessionOptions()
# Register the custom EP; the name must match the one registered inside the EP
session_options.register_custom_ops_library("./build/libsimulator_ep.so")

# Create the session
# Note: the custom EP name must be specified in providers
session = ort.InferenceSession(
    "model.onnx",
    sess_options=session_options,
    providers=["SimulatorEP"]
)

# Run inference
result = session.run(None, {"input": ...})
```

## Development Notes

1. **ABI compatibility**: Ensure the ONNX Runtime headers used to compile the EP match the version of the `onnxruntime.so` loaded at runtime.
2. **Memory management**: When handling tensor data in `Execute`, use zero-copy optimizations to avoid unnecessary memory copies.
3. **Operator support**: Declare supported operators accurately in `GetCapability`; unsupported operators will automatically fall back to the CPU EP.
