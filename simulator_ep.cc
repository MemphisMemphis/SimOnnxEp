// SimulatorEP: an out-of-tree ONNX Runtime execution provider (plugin EP).
//
// Implements the EP Factory API (OrtEpFactory / OrtEp / OrtNodeComputeInfo)
// declared in onnxruntime_ep_c_api.h, which is pulled in by onnxruntime_c_api.h.
//
// The library exports CreateEpFactories / ReleaseEpFactory; ONNX Runtime
// dlopen()s it at registration time and calls CreateEpFactories to obtain the
// OrtEpFactory. The plugin consumes ORT only through the function-pointer
// tables captured here, so it must NOT link against libonnxruntime.

#include "onnxruntime_c_api.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

const OrtApi* g_ort = nullptr;
const OrtEpApi* g_ep_api = nullptr;

// Returns an OrtStatus (non-null) with ORT_FAIL. The C API convention is that a
// NULL return from these callbacks means "success", so every real error must
// return a non-NULL OrtStatus.
OrtStatus* MakeErrorStatus(const char* message) {
  return (g_ort != nullptr) ? g_ort->CreateStatus(ORT_FAIL, message) : nullptr;
}

// Gemm attributes (read once at Compile time, stored in the compute state).
struct GemmParams {
  float alpha = 1.0f;
  float beta = 1.0f;
  int64_t transA = 0;
  int64_t transB = 0;
};

// Conv attributes (2D only). pads is [top, left, bottom, right].
struct ConvParams {
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilations;
  int64_t group = 1;
};

// ---------------------------------------------------------------------------
// Simulator backend hooks. Replace these with calls into the real simulator.
// ---------------------------------------------------------------------------
void* Sim_Init() { return nullptr; }
void Sim_Run(void* /*ctx*/) { std::fprintf(stderr, "[Sim] Sim_Run called\n"); }
void Sim_Destroy(void* /*ctx*/) {}

// ---------------------------------------------------------------------------
// Tensor helpers (float32 only, to keep the demo EP small and self-contained).
// ---------------------------------------------------------------------------
OrtStatus* GetFloat32Tensor(const OrtValue* value, const float** data,
                            std::vector<int64_t>& shape) {
  OrtTensorTypeAndShapeInfo* info = nullptr;
  OrtStatus* status = g_ort->GetTensorTypeAndShape(value, &info);
  if (status != nullptr) {
    return status;
  }

  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  status = g_ort->GetTensorElementType(info, &type);
  if (status == nullptr && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    status = MakeErrorStatus("SimulatorEP only supports float32 tensors");
  }

  size_t rank = 0;
  if (status == nullptr) {
    status = g_ort->GetDimensionsCount(info, &rank);
  }
  if (status == nullptr) {
    shape.resize(rank);
    if (rank > 0) {
      status = g_ort->GetDimensions(info, shape.data(), rank);
    }
  }

  if (status == nullptr) {
    const void* raw_data = nullptr;
    status = g_ort->GetTensorData(value, &raw_data);
    *data = static_cast<const float*>(raw_data);
  }

  g_ort->ReleaseTensorTypeAndShapeInfo(info);
  return status;
}

// Row-major strides for a contiguous tensor with the given shape.
std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Strides aligned to `rank` output dimensions. A dimension of size 1 gets a
// stride of 0 so it broadcasts.
std::vector<int64_t> AlignedBroadcastStrides(const std::vector<int64_t>& shape,
                                             size_t rank) {
  const std::vector<int64_t> own = ContiguousStrides(shape);
  std::vector<int64_t> aligned(rank, 0);
  const size_t offset = rank - shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    aligned[offset + i] = (shape[i] == 1) ? 0 : own[i];
  }
  return aligned;
}

// NumPy-style right-aligned broadcast shape. Empty means "not broadcastable".
std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a,
                                    const std::vector<int64_t>& b) {
  const size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
    const int64_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
    if (da != db && da != 1 && db != 1) {
      return {};
    }
    out[rank - 1 - i] = (da > db) ? da : db;
  }
  return out;
}

OrtStatus* ComputeAdd(OrtKernelContext* context) {
  const OrtValue* a_value = nullptr;
  const OrtValue* b_value = nullptr;
  OrtStatus* status = g_ort->KernelContext_GetInput(context, 0, &a_value);
  if (status == nullptr) {
    status = g_ort->KernelContext_GetInput(context, 1, &b_value);
  }

  const float* a = nullptr;
  const float* b = nullptr;
  std::vector<int64_t> a_shape, b_shape;
  if (status == nullptr) {
    status = GetFloat32Tensor(a_value, &a, a_shape);
  }
  if (status == nullptr) {
    status = GetFloat32Tensor(b_value, &b, b_shape);
  }

  std::vector<int64_t> out_shape;
  if (status == nullptr) {
    out_shape = BroadcastShape(a_shape, b_shape);
    if (out_shape.empty()) {
      status = MakeErrorStatus("SimulatorEP Add: incompatible input shapes");
    }
  }

  OrtValue* out_value = nullptr;
  if (status == nullptr) {
    status = g_ort->KernelContext_GetOutput(context, 0, out_shape.data(),
                                            out_shape.size(), &out_value);
  }

  if (status == nullptr) {
    float* out = nullptr;
    status = g_ort->GetTensorMutableData(out_value, reinterpret_cast<void**>(&out));
    if (status != nullptr) return status;
    size_t total = 1;
    for (const int64_t d : out_shape) {
      total *= static_cast<size_t>(d);
    }

    const size_t rank = out_shape.size();
    const std::vector<int64_t> out_strides = ContiguousStrides(out_shape);
    const std::vector<int64_t> a_strides = AlignedBroadcastStrides(a_shape, rank);
    const std::vector<int64_t> b_strides = AlignedBroadcastStrides(b_shape, rank);

    for (size_t lin = 0; lin < total; ++lin) {
      size_t rem = lin;
      int64_t a_idx = 0;
      int64_t b_idx = 0;
      for (size_t d = 0; d < rank; ++d) {
        const int64_t coord = static_cast<int64_t>(rem / out_strides[d]);
        rem %= out_strides[d];
        a_idx += coord * a_strides[d];
        b_idx += coord * b_strides[d];
      }
      out[lin] = a[a_idx] + b[b_idx];
    }
  }

  return status;
}

OrtStatus* ComputeMatMul2D(OrtKernelContext* context) {
  const OrtValue* a_value = nullptr;
  const OrtValue* b_value = nullptr;
  OrtStatus* status = g_ort->KernelContext_GetInput(context, 0, &a_value);
  if (status == nullptr) {
    status = g_ort->KernelContext_GetInput(context, 1, &b_value);
  }

  const float* a = nullptr;
  const float* b = nullptr;
  std::vector<int64_t> a_shape, b_shape;
  if (status == nullptr) {
    status = GetFloat32Tensor(a_value, &a, a_shape);
  }
  if (status == nullptr) {
    status = GetFloat32Tensor(b_value, &b, b_shape);
  }

  if (status == nullptr && (a_shape.size() != 2 || b_shape.size() != 2)) {
    status = MakeErrorStatus("SimulatorEP MatMul: only 2D inputs are supported");
  }
  if (status == nullptr && a_shape[1] != b_shape[0]) {
    status = MakeErrorStatus("SimulatorEP MatMul: inner dimensions must match");
  }

  const int64_t M = a_shape[0];
  const int64_t K = a_shape[1];
  const int64_t N = b_shape[1];

  OrtValue* out_value = nullptr;
  const int64_t out_shape[2] = {M, N};
  if (status == nullptr) {
    status = g_ort->KernelContext_GetOutput(context, 0, out_shape, 2, &out_value);
  }

  if (status == nullptr) {
    float* out = nullptr;
    status = g_ort->GetTensorMutableData(out_value, reinterpret_cast<void**>(&out));
    if (status != nullptr) return status;
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += a[m * K + k] * b[k * N + n];
        }
        out[m * N + n] = sum;
      }
    }
  }

  return status;
}

// Read Gemm's alpha/beta/transA/transB attributes from a node.
OrtStatus* ReadGemmParams(const OrtNode* node, GemmParams& params) {
  size_t num_attrs = 0;
  OrtStatus* status = g_ort->Node_GetNumAttributes(node, &num_attrs);
  if (status != nullptr) {
    return status;
  }

  std::vector<const OrtOpAttr*> attrs(num_attrs);
  if (num_attrs > 0) {
    status = g_ort->Node_GetAttributes(node, attrs.data(), num_attrs);
    if (status != nullptr) {
      return status;
    }
  }

  for (const OrtOpAttr* attr : attrs) {
    const char* name = nullptr;
    status = g_ort->OpAttr_GetName(attr, &name);
    if (status != nullptr) {
      return status;
    }
    if (name == nullptr) {
      continue;
    }

    size_t out = 0;
    if (std::strcmp(name, "alpha") == 0) {
      status = g_ort->ReadOpAttr(attr, ORT_OP_ATTR_FLOAT, &params.alpha, sizeof(float), &out);
    } else if (std::strcmp(name, "beta") == 0) {
      status = g_ort->ReadOpAttr(attr, ORT_OP_ATTR_FLOAT, &params.beta, sizeof(float), &out);
    } else if (std::strcmp(name, "transA") == 0) {
      status = g_ort->ReadOpAttr(attr, ORT_OP_ATTR_INT, &params.transA, sizeof(params.transA), &out);
    } else if (std::strcmp(name, "transB") == 0) {
      status = g_ort->ReadOpAttr(attr, ORT_OP_ATTR_INT, &params.transB, sizeof(params.transB), &out);
    }
    if (status != nullptr) {
      return status;
    }
  }

  return nullptr;
}

// Y = alpha * (A' @ B') + beta * C, where A'/B' are optionally transposed and C
// is an optional broadcastable bias.
OrtStatus* ComputeGemm(OrtKernelContext* context, const GemmParams& params) {
  const OrtValue* a_value = nullptr;
  const OrtValue* b_value = nullptr;
  const OrtValue* c_value = nullptr;
  OrtStatus* status = g_ort->KernelContext_GetInput(context, 0, &a_value);
  if (status == nullptr) {
    status = g_ort->KernelContext_GetInput(context, 1, &b_value);
  }

  size_t num_inputs = 0;
  if (status == nullptr) {
    status = g_ort->KernelContext_GetInputCount(context, &num_inputs);
  }
  if (status == nullptr && num_inputs >= 3) {
    status = g_ort->KernelContext_GetInput(context, 2, &c_value);
  }

  const float* a = nullptr;
  const float* b = nullptr;
  const float* c = nullptr;
  std::vector<int64_t> a_shape, b_shape, c_shape;
  if (status == nullptr) {
    status = GetFloat32Tensor(a_value, &a, a_shape);
  }
  if (status == nullptr) {
    status = GetFloat32Tensor(b_value, &b, b_shape);
  }
  if (status == nullptr && c_value != nullptr) {
    status = GetFloat32Tensor(c_value, &c, c_shape);
  }

  if (status == nullptr && (a_shape.size() != 2 || b_shape.size() != 2)) {
    status = MakeErrorStatus("SimulatorEP Gemm: A and B must be 2D");
  }

  const int64_t M = a_shape.size() == 2 ? (params.transA ? a_shape[1] : a_shape[0]) : 0;
  const int64_t K = a_shape.size() == 2 ? (params.transA ? a_shape[0] : a_shape[1]) : 0;
  const int64_t Kb = b_shape.size() == 2 ? (params.transB ? b_shape[1] : b_shape[0]) : 0;
  const int64_t N = b_shape.size() == 2 ? (params.transB ? b_shape[0] : b_shape[1]) : 0;

  if (status == nullptr && K != Kb) {
    status = MakeErrorStatus("SimulatorEP Gemm: inner dimensions must match");
  }

  OrtValue* out_value = nullptr;
  const int64_t out_shape[2] = {M, N};
  if (status == nullptr) {
    status = g_ort->KernelContext_GetOutput(context, 0, out_shape, 2, &out_value);
  }

  if (status == nullptr) {
    float* out = nullptr;
    status = g_ort->GetTensorMutableData(out_value, reinterpret_cast<void**>(&out));
    if (status != nullptr) return status;

    const std::vector<int64_t> c_strides = (c != nullptr) ? AlignedBroadcastStrides(c_shape, 2)
                                                          : std::vector<int64_t>(2, 0);

    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          const int64_t a_idx = params.transA ? (k * M + m) : (m * K + k);
          const int64_t b_idx = params.transB ? (n * K + k) : (k * N + n);
          sum += a[a_idx] * b[b_idx];
        }
        sum *= params.alpha;
        if (c != nullptr) {
          const int64_t c_idx = m * c_strides[0] + n * c_strides[1];
          sum += params.beta * c[c_idx];
        }
        out[m * N + n] = sum;
      }
    }
  }

  return status;
}

// Read an INTS attribute (list of int64) using the two-call pattern.
OrtStatus* ReadIntsAttr(const OrtOpAttr* attr, std::vector<int64_t>& values) {
  size_t required = 0;
  OrtStatus* status = g_ort->ReadOpAttr(attr, ORT_OP_ATTR_INTS, nullptr, 0, &required);
  if (status != nullptr) {
    g_ort->ReleaseStatus(status);  // expected: buffer too small; 'required' is now set
  }
  if (required == 0) {
    values.clear();
    return nullptr;
  }
  values.resize(required / sizeof(int64_t));
  size_t written = 0;
  status = g_ort->ReadOpAttr(attr, ORT_OP_ATTR_INTS, values.data(), required, &written);
  if (status != nullptr) {
    return status;
  }
  values.resize(written / sizeof(int64_t));
  return nullptr;
}

// Read Conv's kernel_shape/strides/pads/dilations/group attributes.
OrtStatus* ReadConvParams(const OrtNode* node, ConvParams& params) {
  size_t num_attrs = 0;
  OrtStatus* status = g_ort->Node_GetNumAttributes(node, &num_attrs);
  if (status != nullptr) {
    return status;
  }

  std::vector<const OrtOpAttr*> attrs(num_attrs);
  if (num_attrs > 0) {
    status = g_ort->Node_GetAttributes(node, attrs.data(), num_attrs);
    if (status != nullptr) {
      return status;
    }
  }

  for (const OrtOpAttr* attr : attrs) {
    const char* name = nullptr;
    status = g_ort->OpAttr_GetName(attr, &name);
    if (status != nullptr) {
      return status;
    }
    if (name == nullptr) {
      continue;
    }

    if (std::strcmp(name, "kernel_shape") == 0) {
      status = ReadIntsAttr(attr, params.kernel_shape);
    } else if (std::strcmp(name, "strides") == 0) {
      status = ReadIntsAttr(attr, params.strides);
    } else if (std::strcmp(name, "pads") == 0) {
      status = ReadIntsAttr(attr, params.pads);
    } else if (std::strcmp(name, "dilations") == 0) {
      status = ReadIntsAttr(attr, params.dilations);
    } else if (std::strcmp(name, "group") == 0) {
      size_t out = 0;
      status = g_ort->ReadOpAttr(attr, ORT_OP_ATTR_INT, &params.group, sizeof(params.group), &out);
    }
    // auto_pad is ignored here; models like torchvision ResNet export explicit
    // pads with auto_pad=NOTSET.
    if (status != nullptr) {
      return status;
    }
  }

  const size_t nd = params.kernel_shape.size();
  if (nd > 0) {
    if (params.strides.empty()) params.strides.assign(nd, 1);
    if (params.dilations.empty()) params.dilations.assign(nd, 1);
    if (params.pads.empty()) params.pads.assign(2 * nd, 0);
  }

  return nullptr;
}

// 2D convolution (NCHW), group == 1, with stride/pad/dilation.
OrtStatus* ComputeConv2D(OrtKernelContext* context, const ConvParams& params) {
  const OrtValue* x_value = nullptr;
  const OrtValue* w_value = nullptr;
  const OrtValue* b_value = nullptr;
  OrtStatus* status = g_ort->KernelContext_GetInput(context, 0, &x_value);
  if (status == nullptr) {
    status = g_ort->KernelContext_GetInput(context, 1, &w_value);
  }
  size_t num_inputs = 0;
  if (status == nullptr) {
    status = g_ort->KernelContext_GetInputCount(context, &num_inputs);
  }
  if (status == nullptr && num_inputs >= 3) {
    status = g_ort->KernelContext_GetInput(context, 2, &b_value);
  }

  const float* x = nullptr;
  const float* w = nullptr;
  const float* b = nullptr;
  std::vector<int64_t> x_shape, w_shape, b_shape;
  if (status == nullptr) status = GetFloat32Tensor(x_value, &x, x_shape);
  if (status == nullptr) status = GetFloat32Tensor(w_value, &w, w_shape);
  if (status == nullptr && b_value != nullptr) status = GetFloat32Tensor(b_value, &b, b_shape);

  if (status == nullptr && (x_shape.size() != 4 || w_shape.size() != 4)) {
    status = MakeErrorStatus("SimulatorEP Conv: only 2D (NCHW) convolution is supported");
  }
  if (status == nullptr && params.group != 1) {
    status = MakeErrorStatus("SimulatorEP Conv: only group=1 is supported");
  }

  const int64_t N = x_shape.size() == 4 ? x_shape[0] : 0;
  const int64_t C = x_shape.size() == 4 ? x_shape[1] : 0;
  const int64_t H = x_shape.size() == 4 ? x_shape[2] : 0;
  const int64_t W = x_shape.size() == 4 ? x_shape[3] : 0;

  const int64_t M = w_shape.size() == 4 ? w_shape[0] : 0;
  const int64_t kH = w_shape.size() == 4 ? w_shape[2] : 0;
  const int64_t kW = w_shape.size() == 4 ? w_shape[3] : 0;

  const int64_t strideH = params.strides.size() >= 2 ? params.strides[0] : 1;
  const int64_t strideW = params.strides.size() >= 2 ? params.strides[1] : 1;
  const int64_t dilH = params.dilations.size() >= 2 ? params.dilations[0] : 1;
  const int64_t dilW = params.dilations.size() >= 2 ? params.dilations[1] : 1;
  const int64_t padT = params.pads.size() >= 4 ? params.pads[0] : 0;
  const int64_t padL = params.pads.size() >= 4 ? params.pads[1] : 0;
  const int64_t padB = params.pads.size() >= 4 ? params.pads[2] : 0;
  const int64_t padR = params.pads.size() >= 4 ? params.pads[3] : 0;

  const int64_t outH = (H + padT + padB - dilH * (kH - 1) - 1) / strideH + 1;
  const int64_t outW = (W + padL + padR - dilW * (kW - 1) - 1) / strideW + 1;

  OrtValue* out_value = nullptr;
  const int64_t out_shape[4] = {N, M, outH, outW};
  if (status == nullptr) {
    status = g_ort->KernelContext_GetOutput(context, 0, out_shape, 4, &out_value);
  }

  if (status == nullptr) {
    float* out = nullptr;
    status = g_ort->GetTensorMutableData(out_value, reinterpret_cast<void**>(&out));
    if (status != nullptr) return status;

    for (int64_t n = 0; n < N; ++n) {
      for (int64_t m = 0; m < M; ++m) {
        const float bias = (b != nullptr) ? b[m] : 0.0f;
        for (int64_t oh = 0; oh < outH; ++oh) {
          for (int64_t ow = 0; ow < outW; ++ow) {
            float sum = bias;
            for (int64_t c = 0; c < C; ++c) {
              for (int64_t kh = 0; kh < kH; ++kh) {
                const int64_t ih = oh * strideH + kh * dilH - padT;
                if (ih < 0 || ih >= H) continue;
                for (int64_t kw = 0; kw < kW; ++kw) {
                  const int64_t iw = ow * strideW + kw * dilW - padL;
                  if (iw < 0 || iw >= W) continue;
                  sum += x[((n * C + c) * H + ih) * W + iw] *
                         w[((m * C + c) * kH + kh) * kW + kw];
                }
              }
            }
            out[((n * M + m) * outH + oh) * outW + ow] = sum;
          }
        }
      }
    }
  }

  return status;
}

// ---------------------------------------------------------------------------
// Compute state and node compute info.
//
// SimulatorNodeComputeInfo embeds OrtNodeComputeInfo as its first member so a
// OrtNodeComputeInfo* can be recovered from the this_ptr passed to the
// CreateState/Compute/ReleaseState callbacks.
// ---------------------------------------------------------------------------
struct SimulatorComputeState {
  std::string op_type;
  GemmParams gemm;
  ConvParams conv;
};

struct SimulatorNodeComputeInfo {
  OrtNodeComputeInfo base;  // must stay the first member
  std::string op_type;
  GemmParams gemm;
  ConvParams conv;
};

OrtStatus* ORT_API_CALL SimulatorEp_CreateState(OrtNodeComputeInfo* this_ptr,
                                                OrtNodeComputeContext* /*context*/,
                                                void** compute_state) noexcept {
  const auto* info = reinterpret_cast<const SimulatorNodeComputeInfo*>(this_ptr);
  *compute_state = new SimulatorComputeState{info->op_type, info->gemm, info->conv};
  return nullptr;
}

OrtStatus* ORT_API_CALL SimulatorEp_Compute(OrtNodeComputeInfo* /*this_ptr*/,
                                            void* compute_state,
                                            OrtKernelContext* context) noexcept {
  const auto* state = static_cast<const SimulatorComputeState*>(compute_state);
  if (state == nullptr) {
    return MakeErrorStatus("SimulatorEP: null compute state");
  }

  Sim_Run(nullptr);  // optional simulator hook

  if (state->op_type == "Add") {
    return ComputeAdd(context);
  }
  if (state->op_type == "MatMul") {
    return ComputeMatMul2D(context);
  }
  if (state->op_type == "Gemm") {
    return ComputeGemm(context, state->gemm);
  }
  if (state->op_type == "Conv") {
    return ComputeConv2D(context, state->conv);
  }
  return MakeErrorStatus("SimulatorEP: unsupported operator type");
}

void ORT_API_CALL SimulatorEp_ReleaseState(OrtNodeComputeInfo* /*this_ptr*/,
                                           void* compute_state) noexcept {
  delete static_cast<SimulatorComputeState*>(compute_state);
}

// ---------------------------------------------------------------------------
// OrtEp
// ---------------------------------------------------------------------------
const char* ORT_API_CALL SimulatorEp_GetName(const OrtEp* /*ep*/) noexcept {
  return "SimulatorEP";
}

OrtStatus* ORT_API_CALL SimulatorEp_GetCapability(
    OrtEp* /*ep*/, const OrtGraph* graph,
    OrtEpGraphSupportInfo* graph_support_info) noexcept {
  if (g_ort == nullptr || g_ep_api == nullptr) {
    return nullptr;  // success, but claim no nodes
  }

  size_t num_nodes = 0;
  OrtStatus* status = g_ort->Graph_GetNumNodes(graph, &num_nodes);
  if (status != nullptr) {
    return status;
  }

  std::vector<const OrtNode*> nodes(num_nodes);
  status = g_ort->Graph_GetNodes(graph, nodes.data(), num_nodes);
  if (status != nullptr) {
    return status;
  }

  for (const OrtNode* node : nodes) {
    const char* op_type = nullptr;
    status = g_ort->Node_GetOperatorType(node, &op_type);
    if (status != nullptr) {
      return status;
    }
    if (op_type == nullptr) {
      continue;
    }

    if (std::strcmp(op_type, "Add") == 0 || std::strcmp(op_type, "MatMul") == 0 ||
        std::strcmp(op_type, "Gemm") == 0 || std::strcmp(op_type, "Conv") == 0) {
      const OrtNode* nodes_to_fuse[1] = {node};
      status = g_ep_api->EpGraphSupportInfo_AddNodesToFuse(graph_support_info, nodes_to_fuse, 1, nullptr);
      if (status != nullptr) {
        return status;
      }
    }
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL SimulatorEp_Compile(
    OrtEp* /*ep*/, const OrtGraph** graphs, const OrtNode** /*fused_nodes*/,
    size_t count, OrtNodeComputeInfo** node_compute_infos,
    OrtNode** /*ep_context_nodes*/) noexcept {
  for (size_t i = 0; i < count; ++i) {
    std::string op_type;
    GemmParams gemm;
    ConvParams conv;

    size_t num_nodes = 0;
    OrtStatus* status = g_ort->Graph_GetNumNodes(graphs[i], &num_nodes);
    const OrtNode* node = nullptr;
    if (status == nullptr && num_nodes > 0) {
      std::vector<const OrtNode*> nodes(num_nodes);
      status = g_ort->Graph_GetNodes(graphs[i], nodes.data(), num_nodes);
      if (status == nullptr) {
        node = nodes[0];
        const char* op = nullptr;
        status = g_ort->Node_GetOperatorType(node, &op);
        if (status == nullptr && op != nullptr) {
          op_type = op;
        }
      }
    }

    if (status == nullptr && op_type == "Gemm" && node != nullptr) {
      status = ReadGemmParams(node, gemm);
    }

    if (status == nullptr && op_type == "Conv" && node != nullptr) {
      status = ReadConvParams(node, conv);
    }

    if (status != nullptr) {
      for (size_t j = 0; j < i; ++j) {
        delete reinterpret_cast<SimulatorNodeComputeInfo*>(node_compute_infos[j]);
        node_compute_infos[j] = nullptr;
      }
      return status;
    }

    auto* info = new SimulatorNodeComputeInfo();
    info->op_type = op_type;
    info->gemm = gemm;
    info->conv = conv;
    info->base.ort_version_supported = ORT_API_VERSION;
    info->base.CreateState = SimulatorEp_CreateState;
    info->base.Compute = SimulatorEp_Compute;
    info->base.ReleaseState = SimulatorEp_ReleaseState;
    node_compute_infos[i] = &info->base;
  }
  return nullptr;
}

void ORT_API_CALL SimulatorEp_ReleaseNodeComputeInfos(
    OrtEp* /*ep*/, OrtNodeComputeInfo** node_compute_infos,
    size_t num_node_compute_infos) noexcept {
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    delete reinterpret_cast<SimulatorNodeComputeInfo*>(node_compute_infos[i]);
  }
}

// ---------------------------------------------------------------------------
// OrtEpFactory
// ---------------------------------------------------------------------------
const char* ORT_API_CALL SimulatorFactory_GetName(const OrtEpFactory*) noexcept {
  return "SimulatorEP";
}

const char* ORT_API_CALL SimulatorFactory_GetVendor(const OrtEpFactory*) noexcept {
  return "Local";
}

uint32_t ORT_API_CALL SimulatorFactory_GetVendorId(const OrtEpFactory*) noexcept {
  return 0;
}

const char* ORT_API_CALL SimulatorFactory_GetVersion(const OrtEpFactory*) noexcept {
  return "1.0.0";
}

bool ORT_API_CALL SimulatorFactory_IsStreamAware(const OrtEpFactory*) noexcept {
  return false;
}

OrtStatus* ORT_API_CALL SimulatorFactory_CreateDataTransfer(OrtEpFactory* /*factory*/,
                                                            OrtDataTransferImpl** data_transfer) noexcept {
  *data_transfer = nullptr;  // no custom data transfer; use CPU
  return nullptr;
}

OrtStatus* ORT_API_CALL SimulatorFactory_GetSupportedDevices(
    OrtEpFactory* factory, const OrtHardwareDevice* const* devices,
    size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
    size_t* num_ep_devices) noexcept {
  if (g_ort == nullptr || g_ep_api == nullptr || num_ep_devices == nullptr) {
    return nullptr;
  }

  *num_ep_devices = 0;
  for (size_t i = 0; i < num_devices; ++i) {
    if (g_ort->HardwareDevice_Type(devices[i]) != OrtHardwareDeviceType_CPU) {
      continue;
    }
    if (*num_ep_devices >= max_ep_devices) {
      break;
    }

    OrtStatus* status = g_ep_api->CreateEpDevice(
        factory, devices[i], nullptr, nullptr, &ep_devices[*num_ep_devices]);
    if (status != nullptr) {
      return status;
    }
    ++(*num_ep_devices);
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL SimulatorFactory_CreateEp(
    OrtEpFactory* /*factory*/, const OrtHardwareDevice* const* /*devices*/,
    const OrtKeyValuePairs* const* /*ep_metadata_pairs*/, size_t /*num_devices*/,
    const OrtSessionOptions* /*session_options*/, const OrtLogger* /*logger*/,
    OrtEp** ep) noexcept {
  auto* simulator_ep = new OrtEp();
  simulator_ep->ort_version_supported = ORT_API_VERSION;
  simulator_ep->GetName = SimulatorEp_GetName;
  simulator_ep->GetCapability = SimulatorEp_GetCapability;
  simulator_ep->Compile = SimulatorEp_Compile;
  simulator_ep->ReleaseNodeComputeInfos = SimulatorEp_ReleaseNodeComputeInfos;
  // All other optional OrtEp callbacks are left NULL (value-initialized).
  *ep = simulator_ep;
  return nullptr;
}

void ORT_API_CALL SimulatorFactory_ReleaseEp(OrtEpFactory* /*factory*/,
                                             OrtEp* ep) noexcept {
  delete ep;
}

}  // namespace

// ---------------------------------------------------------------------------
// Library entry points.
// ---------------------------------------------------------------------------
extern "C" ORT_EXPORT OrtStatus* ORT_API_CALL CreateEpFactories(
    const char* /*registered_name*/, const OrtApiBase* ort_api_base,
    const OrtLogger* /*default_logger*/, OrtEpFactory** factories,
    size_t max_factories, size_t* num_factories) noexcept {
  if (num_factories != nullptr) {
    *num_factories = 0;
  }
  if (ort_api_base == nullptr || factories == nullptr ||
      num_factories == nullptr || max_factories < 1) {
    return nullptr;
  }

  g_ort = ort_api_base->GetApi(ORT_API_VERSION);
  if (g_ort == nullptr) {
    return nullptr;
  }
  g_ep_api = g_ort->GetEpApi();

  auto* factory = new OrtEpFactory();
  factory->ort_version_supported = ORT_API_VERSION;
  factory->GetName = SimulatorFactory_GetName;
  factory->GetVendor = SimulatorFactory_GetVendor;
  factory->GetSupportedDevices = SimulatorFactory_GetSupportedDevices;
  factory->CreateEp = SimulatorFactory_CreateEp;
  factory->ReleaseEp = SimulatorFactory_ReleaseEp;
  factory->GetVendorId = SimulatorFactory_GetVendorId;
  factory->GetVersion = SimulatorFactory_GetVersion;
  factory->IsStreamAware = SimulatorFactory_IsStreamAware;
  factory->CreateDataTransfer = SimulatorFactory_CreateDataTransfer;

  factories[0] = factory;
  *num_factories = 1;
  return nullptr;
}

extern "C" ORT_EXPORT OrtStatus* ORT_API_CALL ReleaseEpFactory(
    OrtEpFactory* factory) noexcept {
  delete factory;
  return nullptr;
}
