// Minimal host that loads SimulatorEP and runs any ONNX model through it.
//
// Inputs are fed as zero-filled tensors of the model's declared shapes
// (dynamic dims default to 1). Output shape and a few values are printed.
//
// Build (from the sim_ep directory):
//   g++ main.cc -o run_sim \
//     -I/home/gangj/code/onnx/onnxruntime/include/onnxruntime/core/session \
//     -I/home/gangj/code/onnx/onnxruntime/include \
//     -L/home/gangj/code/onnx/onnxruntime/build/Linux/Debug \
//     -lonnxruntime -Wl,-rpath,/home/gangj/code/onnx/onnxruntime/build/Linux/Debug
//
// Run:
//   ./run_sim <model.onnx> <path/to/libsimulator_ep.so>

#include "onnxruntime_c_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const OrtApi* g_api = nullptr;

static void Die(OrtStatus* status) {
  if (status == nullptr) return;
  const char* msg = g_api->GetErrorMessage(status);
  std::fprintf(stderr, "ORT error: %s\n", msg != nullptr ? msg : "(no message)");
  g_api->ReleaseStatus(status);
  std::exit(1);
}

static size_t ElementByteSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return 2;
    default:
      return 4;
  }
}

int main(int argc, char** argv) {
  const char* model_path = argc > 1 ? argv[1] : "resnet18.onnx";
  const char* ep_lib = argc > 2 ? argv[2] : "./build/libsimulator_ep.so";

  g_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (g_api == nullptr) {
    std::fprintf(stderr, "failed to get ORT API\n");
    return 1;
  }

  OrtEnv* env = nullptr;
  Die(g_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "sim_ep_test", &env));

  // 1) Register the plugin EP library.
  Die(g_api->RegisterExecutionProviderLibrary(env, "SimulatorEP", ep_lib));

  OrtSessionOptions* so = nullptr;
  Die(g_api->CreateSessionOptions(&so));
  Die(g_api->SetSessionLogSeverityLevel(so, 2));  // ERROR only, keep output small

  // 2) Discover the plugin EP's OrtEpDevice and append it.
  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_ep_devices = 0;
  Die(g_api->GetEpDevices(env, &ep_devices, &num_ep_devices));

  const OrtEpDevice* sim_device = nullptr;
  for (size_t i = 0; i < num_ep_devices; ++i) {
    const char* name = g_api->EpDevice_EpName(ep_devices[i]);
    if (name != nullptr && std::strcmp(name, "SimulatorEP") == 0) {
      sim_device = ep_devices[i];
      break;
    }
  }
  if (sim_device == nullptr) {
    std::fprintf(stderr, "SimulatorEP device not found after registration\n");
    return 1;
  }

  Die(g_api->SessionOptionsAppendExecutionProvider_V2(so, env, &sim_device, 1, nullptr, nullptr, 0));

  OrtSession* session = nullptr;
  Die(g_api->CreateSession(env, model_path, so, &session));

  OrtAllocator* allocator = nullptr;
  Die(g_api->GetAllocatorWithDefaultOptions(&allocator));

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  Die(g_api->SessionGetInputCount(session, &num_inputs));
  Die(g_api->SessionGetOutputCount(session, &num_outputs));

  std::vector<char*> input_names(num_inputs);
  std::vector<char*> output_names(num_outputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    Die(g_api->SessionGetInputName(session, i, allocator, &input_names[i]));
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    Die(g_api->SessionGetOutputName(session, i, allocator, &output_names[i]));
  }

  // 3) Allocate zero-filled inputs using the model's declared type/shape.
  std::vector<OrtValue*> input_values(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    OrtTypeInfo* type_info = nullptr;
    Die(g_api->SessionGetInputTypeInfo(session, i, &type_info));

    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    Die(g_api->CastTypeInfoToTensorInfo(type_info, &tensor_info));

    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    Die(g_api->GetTensorElementType(tensor_info, &type));

    size_t rank = 0;
    Die(g_api->GetDimensionsCount(tensor_info, &rank));

    std::vector<int64_t> dims(rank);
    if (rank > 0) {
      Die(g_api->GetDimensions(tensor_info, dims.data(), rank));
    }
    for (int64_t& d : dims) {
      if (d <= 0) d = 1;  // resolve dynamic/symbolic dims to 1
    }

    size_t num_elems = 1;
    for (int64_t d : dims) num_elems *= static_cast<size_t>(d);

    OrtValue* v = nullptr;
    Die(g_api->CreateTensorAsOrtValue(allocator, dims.data(), rank, type, &v));
    void* data = nullptr;
    Die(g_api->GetTensorMutableData(v, &data));
    std::memset(data, 0, num_elems * ElementByteSize(type));

    std::printf("input[%zu] '%s' shape=[", i, input_names[i]);
    for (size_t d = 0; d < rank; ++d) std::printf("%lld%s", (long long)dims[d], d + 1 < rank ? "," : "");
    std::printf("] (%zu elems, zero-filled)\n", num_elems);

    g_api->ReleaseTypeInfo(type_info);
    input_values[i] = v;
  }

  // 4) Run.
  std::vector<OrtValue*> outputs(num_outputs, nullptr);
  Die(g_api->Run(session, nullptr,
                 const_cast<const char* const*>(input_names.data()),
                 const_cast<const OrtValue* const*>(input_values.data()), num_inputs,
                 const_cast<const char* const*>(output_names.data()), num_outputs,
                 outputs.data()));

  // 5) Print output shape + first values (assume float32 output).
  for (size_t i = 0; i < num_outputs; ++i) {
    OrtTensorTypeAndShapeInfo* out_info = nullptr;
    Die(g_api->GetTensorTypeAndShape(outputs[i], &out_info));

    size_t out_rank = 0;
    Die(g_api->GetDimensionsCount(out_info, &out_rank));
    std::vector<int64_t> out_dims(out_rank);
    if (out_rank > 0) Die(g_api->GetDimensions(out_info, out_dims.data(), out_rank));

    size_t out_count = 1;
    for (int64_t d : out_dims) out_count *= static_cast<size_t>(d);

    float* out_data = nullptr;
    Die(g_api->GetTensorMutableData(outputs[i], reinterpret_cast<void**>(&out_data)));

    std::printf("output[%zu] '%s' shape=[", i, output_names[i]);
    for (size_t d = 0; d < out_rank; ++d) std::printf("%lld%s", (long long)out_dims[d], d + 1 < out_rank ? "," : "");
    std::printf("]\n  first %zu values: ", out_count < 10 ? out_count : 10);
    for (size_t k = 0; k < out_count && k < 10; ++k) std::printf("%.4f ", out_data[k]);
    std::printf("\n");

    g_api->ReleaseTensorTypeAndShapeInfo(out_info);
  }

  // Cleanup.
  for (OrtValue* v : outputs) g_api->ReleaseValue(v);
  for (OrtValue* v : input_values) g_api->ReleaseValue(v);
  for (char* n : input_names) g_api->AllocatorFree(allocator, n);
  for (char* n : output_names) g_api->AllocatorFree(allocator, n);
  g_api->ReleaseSession(session);
  g_api->ReleaseSessionOptions(so);
  g_api->ReleaseEnv(env);
  return 0;
}
