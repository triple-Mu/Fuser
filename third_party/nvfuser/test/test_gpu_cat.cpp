#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor_utils.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

TEST_F(NVFuserTest, FusionCat1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = cat(tv0, tv1, 0);
  fusion.addOutput(tv2);

  fusion.printMath();

  GpuLower gpulw(&fusion);
  kir::Kernel* kernel = gpulw.kernel();
  for (auto expr : kernel->topLevelExprs()) {
    std::cerr << "Kernel expr: " << expr->toString();
  }

  fusion.printKernel();
}

} // namespace jit
} // namespace torch
