include(FetchContent)
FetchContent_Declare(
  googletest
  SOURCE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/../third-parties/googletest
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(ORT_EP_LIB "${CMAKE_BINARY_DIR}/libsimulator_ep.so")

add_executable(simulator_ep_tests
  src/test_model_test.cc
  src/resnet18_test.cc
)
target_link_libraries(simulator_ep_tests PRIVATE gtest_main onnxruntime_shared)
target_compile_definitions(simulator_ep_tests PRIVATE
  ORT_EP_LIB="${ORT_EP_LIB}"
  TEST_MODEL_DIR="${CMAKE_SOURCE_DIR}/test/models/test_model"
  RESNET18_MODEL="${CMAKE_SOURCE_DIR}/test/models/resnet18/resnet18_classifier.onnx"
)
add_dependencies(simulator_ep_tests simulator_ep)
add_test(NAME simulator_ep_tests COMMAND simulator_ep_tests)
set_tests_properties(simulator_ep_tests PROPERTIES TIMEOUT 900)
