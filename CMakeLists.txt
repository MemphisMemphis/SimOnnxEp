cmake_minimum_required(VERSION 3.18)
project(SimulatorEP LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# ---------------------------------------------------------------------------
# Vendored ONNX Runtime (git submodule at <project>/onnxruntime).
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime" CACHE PATH
    "ONNX Runtime source tree")

set(ORT_BUILD_DIR "${ONNXRUNTIME_ROOT}/build/Linux/${CMAKE_BUILD_TYPE}")
set(ORT_LIB "${ORT_BUILD_DIR}/libonnxruntime.so")

# --- 1. Build ONNX Runtime via its own build.sh (shared lib). ---------------
# Runs once, when libonnxruntime.so is missing. Delete the .so to force a
# rebuild after updating the submodule.
add_custom_command(
  OUTPUT "${ORT_LIB}"
  COMMAND ./build.sh --config "${CMAKE_BUILD_TYPE}" --parallel 2 --build_shared_lib
  WORKING_DIRECTORY "${ONNXRUNTIME_ROOT}"
  COMMENT "Building ONNX Runtime shared library (first build is slow)"
  VERBATIM
)
add_custom_target(onnxruntime_build DEPENDS "${ORT_LIB}")

# --- 2. Import the built shared library. ------------------------------------
add_library(onnxruntime_shared SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime_shared PROPERTIES
  IMPORTED_LOCATION "${ORT_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES
    "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session;${ONNXRUNTIME_ROOT}/include"
)
add_dependencies(onnxruntime_shared onnxruntime_build)

# --- 3. Plugin EP (headers only; must NOT link libonnxruntime). -------------
add_library(simulator_ep SHARED src/simulator_ep.cc)
target_include_directories(simulator_ep PRIVATE
  "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
  "${ONNXRUNTIME_ROOT}/include"
)
set_target_properties(simulator_ep PROPERTIES PREFIX "lib")

# --- 4. Test host (links libonnxruntime, dlopens the plugin at runtime). ----
add_executable(run_sim test/main.cc)
target_link_libraries(run_sim PRIVATE onnxruntime_shared ${CMAKE_DL_LIBS})
add_dependencies(run_sim simulator_ep)

# --- 5. gtest-based tests ---------------------------------------------------
enable_testing()
add_subdirectory(test)
