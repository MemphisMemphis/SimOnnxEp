#include <gtest/gtest.h>

#include <cmath>

#include "ep_test_util.h"

namespace {

// resnet18_classifier.onnx: [1,3,224,224] -> [1,1000] class logits.
// NOTE: the naive Conv kernels make this test slow (minutes).
TEST(SimulatorEP, ResNet18Classifier) {
  auto session = sim_ep_test::MakeSimulatorSession(RESNET18_MODEL, ORT_EP_LIB);

  sim_ep_test::FloatTensor image{};
  image.shape = {1, 3, 224, 224};
  image.data.assign(1 * 3 * 224 * 224, 0.0f);  // zero image -> bias-only logits

  const std::vector<float> out = sim_ep_test::RunFloatModel(session, {image});

  ASSERT_EQ(out.size(), 1000u);  // 1000 ImageNet classes
  for (float v : out) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

}  // namespace
