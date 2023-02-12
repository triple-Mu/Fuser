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

TEST_F(NVFuserTest, FusionIterDomainExpand1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  fusion.printMath();

  tv1->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));

  tv1->split(0, 4);

  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({9}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionIterDomainExpand2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  fusion.printMath();

  tv1->split(0, 3);

  tv1->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));

  // When expand is done on a non-root domain, the input domain of the
  // expand needs to be predicated

  fusion.printKernel();
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({19}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionIterDomainExpand3_1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));
  tv1->merge(0);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({19, 7}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionIterDomainExpand3_2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->expand(1, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));
  tv1->merge(0);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({19, 7}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionIterDomainExpand4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->merge(0);

  tv1->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({19, 7}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionIterDomainExpand5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // auto tv0 = makeSymbolicTensor(1);
  auto tv0 = makeConcreteTensor({7});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv2->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({7}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionIterDomainExpand6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // auto tv0 = makeSymbolicTensor(1);
  auto tv0 = makeConcreteTensor({7, 11});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->split(0, 3);

  tv2->merge(0);
  tv2->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));
  tv2->split(0, 4);

  tv3->split(1, 5);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({7, 11}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionIterDomainExpand7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // auto tv0 = makeSymbolicTensor(1);
  auto tv0 = makeConcreteTensor({11});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->split(0, 3);
  tv1->expand(1, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));

  tv2->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));

  tv3->split(0, 5);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({11}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionIterDomainExpand8_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // auto tv0 = makeSymbolicTensor(1);
  auto tv0 = makeConcreteTensor({11});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->split(0, 3);
  tv1->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));
  tv1->expand(1, IrBuilder::create<Int>(3), IrBuilder::create<Int>(4));

  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({11}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionPad1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  fusion.addOutput(tv1);

  std::cerr << "tv1: " << tv1->toString() << std::endl;
  fusion.printMath();

  std::cerr << tv1->definition()->as<PadOp>()->getPaddedAxes() << std::endl;
  PairwiseRootDomainMap map(tv0, tv1);
  for (auto kv : map.mapProducerToConsumer(tv0->domain(), tv1->domain())) {
    std::cerr << kv.first->toString() << ", " << kv.second->toString()
              << std::endl;
  }

  GpuLower gpulw(&fusion);
  kir::Kernel* kernel = gpulw.kernel();
  for (auto expr : kernel->topLevelExprs()) {
    std::cerr << "Kernel expr: " << expr->toString();
  }

  fusion.printKernel();
}

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
