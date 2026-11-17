

# Simulator Execution Provider for ONNX Runtime

## 构建前提

1. 安装 ONNX Runtime 开发包。
2. 设置环境变量 `ONNXRUNTIME_ROOT` 指向 ORT 安装目录。
3. 确保你的模拟器库 (`libmysim.so`/`.dll`) 已在系统路径中。

## 编译

```bash
mkdir build && cd build
cmake ..
make -j4
```

## 使用方法 (Python)

```python
import onnxruntime as ort

# 加载自定义 EP 动态库
session_options = ort.SessionOptions()
# 注册自定义 EP，名称需与 EP 内部注册一致
session_options.register_custom_ops_library("./build/libsimulator_ep.so")

# 创建会话
# 注意：需要在 providers 中指定自定义 EP 的名称
session = ort.InferenceSession(
    "model.onnx", 
    sess_options=session_options, 
    providers=["SimulatorEP"] 
)

# 运行推理
result = session.run(None, {"input": ...})
```

## 开发注意事项

1. **ABI 兼容性**: 确保编译 EP 使用的 ONNX Runtime 头文件版本与运行时加载的 `onnxruntime.so` 版本一致。
2. **内存管理**: 在 `Execute` 中处理张量数据时，注意零拷贝（Zero-Copy）优化，避免不必要的内存复制。
3. **算子支持**: 在 `GetCapability` 中准确声明支持的算子，不支持的算子会自动回退到 CPU EP。
