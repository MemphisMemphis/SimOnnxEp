cmake_minimum_required(VERSION 3.18)
project(SimulatorEP)

# Point this at the ONNX Runtime source tree (or install tree) that contains
# include/onnxruntime/core/session/onnxruntime_c_api.h.
# Override with: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime
if(NOT ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "/home/gangj/code/onnx/onnxruntime")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# onnxruntime_c_api.h lives at <root>/include/onnxruntime/core/session/onnxruntime_c_api.h
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
        ${ONNXRUNTIME_ROOT}/include/onnxruntime
        ${ONNXRUNTIME_ROOT}/include
        /usr/local/include/onnxruntime/core/session
        /usr/include/onnxruntime/core/session)

if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNX Runtime headers not found. Set ONNXRUNTIME_ROOT to the ORT source/install tree.")
endif()

add_library(simulator_ep SHARED
    simulator_ep.cc
)

target_include_directories(simulator_ep PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_ROOT}/include
)

# The plugin EP consumes ORT through the function-pointer tables supplied in
# CreateEpFactories, so it must NOT link against libonnxruntime.
set_target_properties(simulator_ep PROPERTIES
    PREFIX "lib"
)
