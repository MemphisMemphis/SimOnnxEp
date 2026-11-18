#!/usr/bin/env python3
"""Export torchvision ResNet18 to ONNX (resnet18.onnx)."""
import torch
import torchvision

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy,
    "resnet18.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
)
print("wrote resnet18.onnx")
