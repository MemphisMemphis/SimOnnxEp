#!/usr/bin/env python3
"""Create a tiny ONNX model (Add of two [2,3] float tensors) to exercise SimulatorEP."""
import onnx
from onnx import TensorProto, helper

A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])
C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 3])

node = helper.make_node("Add", ["A", "B"], ["C"], name="add_node")
graph = helper.make_graph([node], "add_graph", [A, B], [C])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8

onnx.checker.check_model(model)
onnx.save(model, "add.onnx")
print("wrote add.onnx")
