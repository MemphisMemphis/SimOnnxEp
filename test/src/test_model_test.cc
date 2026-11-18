#include <gtest/gtest.h>

#include "ep_test_util.h"

namespace {

// add.onnx: A[2,3] + B[2,3] -> C[2,3], executed by SimulatorEP.
TEST(SimulatorEP, AddModel) {
  const std::string model = std::string(TEST_MODEL_DIR) + "/add.onnx";
  auto session = sim_ep_test::MakeSimulatorSession(model, ORT_EP_LIB);

  sim_ep_test::FloatTensor a{{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}};
  sim_ep_test::FloatTensor b{{10.f, 20.f, 30.f, 40.f, 50.f, 60.f}, {2, 3}};

  const std::vector<float> out = sim_ep_test::RunFloatModel(session, {a, b});

  const std::vector<float> expected = {11.f, 22.f, 33.f, 44.f, 55.f, 66.f};
  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-4f) << "at index " << i;
  }
}

}  // namespace
