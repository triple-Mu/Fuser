#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor_utils.h>
#include <fusion.h>
#include <inlining.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>
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

TEST_F(NVFuserTest, FusionIterDomainExpand9_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // auto tv0 = makeSymbolicTensor(1);
  auto tv0 = makeConcreteTensor({11});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // tv1->split(0, 3);
  tv2->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));

  tv1->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));

  inlineMost();

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

TEST_F(NVFuserTest, FusionIterDomainExpand10_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // auto tv0 = makeSymbolicTensor(1);
  auto tv0 = makeConcreteTensor({11});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // tv1->split(0, 3);
  tv2->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));
  tv2->split(0, 3);
  // tv1->expand(1, IrBuilder::create<Int>(3), IrBuilder::create<Int>(4));

  tv1->expand(0, IrBuilder::create<Int>(1), IrBuilder::create<Int>(2));
  tv1->split(0, 3);

  tv1->inlineAt(1);

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

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  fusion.addOutput(tv1);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionPad2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  fusion.addOutput(tv1);

  tv1->split(0, 4);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionPad3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9, 11});
  std::vector<int64_t> padded_shape({9, 11 + 2});

  auto tv0 = makeSymbolicTensor(2);
  // auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  // auto tv1 = makeConcreteTensor(padded_shape);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = pad(tv2, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  fusion.printMath();

  std::cerr << "Padded axes: "
            << tv3->definition()->as<PadOp>()->getPaddedAxes() << std::endl;

#if 1
  tv4->merge(0);
  tv4->split(0, 32);

  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);
#endif

  inlineMost();

  fusion.printMath();
  fusion.print();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  auto t1 = at::randn(padded_shape, options);
  std::vector<IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t3 = at::pad(t0, {1, 1});
  auto ref = t3 + t1;

#if 0
  std::cerr << "t0: " << t0 << std::endl;
  std::cerr << "t1: " << t1 << std::endl;
  std::cerr << "ref: " << ref << std::endl;
  std::cerr << "cg: " << cg_outputs[0] << std::endl;
#endif

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPad4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  fusion.addOutput(tv1);

  tv1->axis(0)->parallelize(ParallelType::TIDx);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionPad5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = pad(tv1, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  fusion.addOutput(tv2);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  tv1->setMemoryType(MemoryType::Shared);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionPad6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({99, 111});
  std::vector<int64_t> padded_shape({shape[0], shape[1] + 2});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor(padded_shape);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = pad(tv2, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  fusion.printMath();

  tv4->merge(0);
  tv4->split(0, 32);

  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);

  fusion.printMath();
  fusion.print();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  auto t1 = at::randn(padded_shape, options);
  std::vector<IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t2 = t0 + 1;
  auto t3 = at::pad(t2, {1, 1});
  auto ref = t3 + t1;

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCat1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({2});
  std::vector<int64_t> shape1({3});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);
  fusion.addOutput(tv2);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 0);

  std::cout << "In0: " << t0 << std::endl;
  std::cout << "In1: " << t1 << std::endl;
  std::cout << "Ref: " << ref << std::endl;
  std::cout << "CG: " << cg_outputs[0] << std::endl;

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionCat2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({2, 4});
  std::vector<int64_t> shape1({3, 4});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);
  fusion.addOutput(tv2);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 0);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionCat3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({4, 2});
  std::vector<int64_t> shape1({4, 3});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 1);
  fusion.addOutput(tv2);

  tv2->merge(0);
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 1);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionCat4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({11, 12});
  std::vector<int64_t> shape1({11, 13});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 1);
  fusion.addOutput(tv2);

  tv2->merge(0);
  tv2->split(0, 128);

  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  inlineMost();

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 1);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionCat5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1}, 1);
  auto tv4 = add(tv3, tv2);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 128);

  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  fusion.printMath();
  fusion.printKernel();

  std::vector<int64_t> shape0({11, 12});
  std::vector<int64_t> shape1({shape0[0], 13});
  std::vector<int64_t> shape2({shape0[0], shape0[1] + shape1[1]});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 1) + t2;

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionCat6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({2, 4});
  std::vector<int64_t> shape1({5, 4});
  std::vector<int64_t> shape2({3, 4});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);
  auto tv2 = makeConcreteTensor(shape2);
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1, tv2}, 0);
  fusion.addOutput(tv3);

  tv3->merge(0);
  tv3->split(0, 4);
  TransformPropagator propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  inlineMost();

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1, t2}, 0);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionCat7_CUDA) {
  int num_tensors_to_concat = 10;

  for (int concat_dim : {0, 1}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    std::vector<TensorView*> inputs;
    for (const auto i : c10::irange(num_tensors_to_concat)) {
      (void)i;
      auto tv = makeSymbolicTensor(2);
      fusion.addInput(tv);
      inputs.push_back(tv);
    }

    auto concat_tv = cat(inputs, concat_dim);
    fusion.addOutput(concat_tv);

    fusion.printMath();

    concat_tv->merge(0);
    concat_tv->split(0, 128);

    TransformPropagator propagator(concat_tv);
    MaxRootDomainInfoSpanningTree(concat_tv).traverse(&propagator);

    inlineMost();

    concat_tv->axis(0)->parallelize(ParallelType::BIDx);
    concat_tv->axis(1)->parallelize(ParallelType::TIDx);
    scheduler_utils::parallelizeAllLike(concat_tv);

    fusion.printMath();
    fusion.printKernel();

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::manual_seed(0);

    std::vector<int64_t> base_shape({11, 13});
    std::vector<at::Tensor> aten_inputs;
    for (const auto i : c10::irange(num_tensors_to_concat)) {
      auto shape = base_shape;
      shape[concat_dim] = 10 + (i % 5);
      aten_inputs.emplace_back(at::randn(shape, options));
    }

    std::vector<IValue> aten_inputs_ivalue(
        {aten_inputs.begin(), aten_inputs.end()});

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs_ivalue);
    auto cg_outputs = fe.runFusion(aten_inputs_ivalue);

    auto ref = at::cat(aten_inputs, concat_dim);

    TORCH_CHECK(ref.equal(cg_outputs[0]));
  }
}

TEST_F(NVFuserTest, FusionSlice1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Int>(1),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Int>(1))}});
  fusion.addOutput(tv1);

  fusion.printMath();

  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = t0.index({at::indexing::Slice(1, shape[0] - 1)});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionSlice2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({11, 30});

  TORCH_CHECK(shape[1] % 2 == 0);

  // auto tv0 = makeSymbolicTensor(2);
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  // This results in float
  // auto mid_point = div(tv0->axis(1)->extent(), IrBuilder::create<Int>(2));
  auto mid_point =
      IrBuilder::divExpr(tv0->axis(1)->extent(), IrBuilder::create<Int>(2));

  std::cerr << "Mid point: " << mid_point->toString()
            << ", type: " << mid_point->getDataType().value() << std::endl;
  auto tv1 = slice(tv0, {Slice(), {IrBuilder::create<Int>(0), mid_point}});
  auto tv2 = slice(tv0, {Slice(), {mid_point}});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  fusion.printMath();

  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t1 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(0, shape[1] / 2)});
  auto t2 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(shape[1] / 2)});
  auto ref = t1 + t2;

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionSlice3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // These should result in unary set op
  auto tv1 = slice(tv0, {{nullptr, tv0->axis(0)->extent()}});
  auto tv2 = slice(tv0, {Slice()});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  fusion.printMath();

  TORCH_CHECK(
      tv1->definition()->isA<UnaryOp>() &&
      tv1->definition()->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Set);
  TORCH_CHECK(
      tv2->definition()->isA<UnaryOp>() &&
      tv2->definition()->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Set);
}

TEST_F(NVFuserTest, FusionSlice4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({5, 100});

  // auto tv0 = makeSymbolicTensor(2);
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  // Consider a fusion of:
  // auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  // auto tv2 = sum(tv1, {1});

  // Reproduce the above fusion with split tensors

  // Split the input to [0:2, :] and [2:, :]
  auto tv1 = slice(
      tv0, {{IrBuilder::create<Int>(0), IrBuilder::create<Int>(2)}, Slice()});
  auto tv2 = slice(tv0, {{IrBuilder::create<Int>(2)}, Slice()});

  auto tv3 = add(tv1, IrBuilder::create<Double>(1));
  auto tv4 = add(tv2, IrBuilder::create<Double>(1));

  auto tv5 = sum(tv3, {1});
  auto tv6 = sum(tv4, {1});
  auto tv7 = cat({tv5, tv6}, 0);
  fusion.addOutput(tv7);

  fusion.printMath();

  // Schedule the two reductions separately
  tv5->split(-1, 32);
  auto tv5_rf = tv5->rFactor({-2});
  tv5_rf->reorder({{-1, -2}});
  auto tv5_cache = tv5->cacheBefore();
  tv5->setMemoryType(MemoryType::Global);
  SetSelector tv5_rf_selector({tv1, tv3, tv5, tv5_cache});
  TransformPropagator tv5_rf_tp(tv5_rf);
  MaxRootDomainInfoSpanningTree(tv5_rf, &tv5_rf_selector).traverse(&tv5_rf_tp);
  inlineMost(std::vector<TensorView*>{tv1, tv3, tv5_rf});
  tv5_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv5_rf->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv5_rf, {tv1, tv3, tv5, tv5_cache});

  tv6->split(-1, 32);
  auto tv6_rf = tv6->rFactor({-2});
  tv6_rf->reorder({{-1, -2}});
  auto tv6_cache = tv6->cacheBefore();
  tv6->setMemoryType(MemoryType::Global);
  SetSelector tv6_rf_selector({tv2, tv4, tv6, tv6_cache});
  TransformPropagator tv6_rf_tp(tv6_rf);
  MaxRootDomainInfoSpanningTree(tv6_rf, &tv6_rf_selector).traverse(&tv6_rf_tp);
  inlineMost(std::vector<TensorView*>{tv2, tv4, tv6_rf});
  tv6_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv6_rf->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv6_rf, {tv2, tv4, tv6, tv6_cache});

  // cat consits of a PadOp and a CatOp. Fully inline the PadOp
  for (auto tv7_inp :
       ir_utils::filterByType<TensorView>(tv7->definition()->inputs())) {
    tv7_inp->inlineAt(-1);
  }

  // Use just one block to concat the two results
  // This doesn't work due to thread predicates (bug?)
  // tv7->axis(0)->parallelize(ParallelType::TIDx);
  tv7->axis(0)->parallelize(ParallelType::BIDx);

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = (t0 + 1).to(at::kDouble).sum({1});

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, TMP1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // These should result in unary set op
  auto tv1 = sum(tv0, {1});
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->setMemoryType(MemoryType::Shared);

  tv3->axis(0)->parallelize(ParallelType::TIDx);

  fusion.printMath();
  fusion.printKernel();
}

} // namespace jit
} // namespace torch
