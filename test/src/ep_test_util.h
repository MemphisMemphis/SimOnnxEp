#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <string>
#include <vector>

namespace sim_ep_test {

// One float32 tensor: data + shape.
struct FloatTensor {
  std::vector<float> data;
  std::vector<int64_t> shape;
};

// The Env must outlive the Session (it keeps the plugin EP library loaded).
// env is declared before session so it is destroyed after it.
struct SimSession {
  Ort::Env env{nullptr};
  Ort::Session session{nullptr};
};

// Register SimulatorEP and load the given model.
inline SimSession MakeSimulatorSession(const std::string& model_path,
                                       const std::string& ep_lib_path) {
  SimSession result;
  result.env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SimulatorEP_test");
  const OrtApi& api = Ort::GetApi();

  Ort::ThrowOnError(
      api.RegisterExecutionProviderLibrary(result.env, "SimulatorEP", ep_lib_path.c_str()));

  const OrtEpDevice* const* devices = nullptr;
  size_t num_devices = 0;
  Ort::ThrowOnError(api.GetEpDevices(result.env, &devices, &num_devices));

  const OrtEpDevice* sim_device = nullptr;
  for (size_t i = 0; i < num_devices; ++i) {
    const char* name = api.EpDevice_EpName(devices[i]);
    if (name != nullptr && std::strcmp(name, "SimulatorEP") == 0) {
      sim_device = devices[i];
      break;
    }
  }
  if (sim_device == nullptr) {
    throw std::runtime_error("SimulatorEP device not found after registration");
  }

  Ort::SessionOptions so;
  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_V2(
      so, result.env, &sim_device, 1, nullptr, nullptr, 0));

  result.session = Ort::Session(result.env, model_path.c_str(), so);
  return result;
}

// Run a model with the given float32 inputs; return the (flattened) float32
// output of the first output.
inline std::vector<float> RunFloatModel(SimSession& sim,
                                        const std::vector<FloatTensor>& inputs) {
  Ort::Session& session = sim.session;
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const size_t num_inputs = inputs.size();
  std::vector<std::string> in_name_storage(num_inputs);
  std::vector<const char*> in_names(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    auto n = session.GetInputNameAllocated(i, alloc);
    in_name_storage[i] = n.get();
    in_names[i] = in_name_storage[i].c_str();
  }

  std::vector<Ort::Value> in_values;
  in_values.reserve(num_inputs);
  for (const FloatTensor& t : inputs) {
    in_values.push_back(Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(t.data.data()), t.data.size(), t.shape.data(), t.shape.size()));
  }

  auto out_name = session.GetOutputNameAllocated(0, alloc);
  const char* out_names[1] = {out_name.get()};

  auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names.data(),
                             in_values.data(), num_inputs, out_names, 1);

  const Ort::Value& out = outputs[0];
  const auto out_info = out.GetTensorTypeAndShapeInfo();
  const size_t count = out_info.GetElementCount();
  const float* out_data = out.GetTensorData<float>();
  return std::vector<float>(out_data, out_data + count);
}

}  // namespace sim_ep_test
