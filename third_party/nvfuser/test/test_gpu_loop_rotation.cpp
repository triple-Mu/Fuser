#include <gtest/gtest.h>

#include <inlining.h>
#include <ops/arith.h>
#include <scheduler/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

namespace nvfuser {

TEST_F(NVFuserTest, FusionLoopRotation1Inner_CUDA) {
  // Please see note [Limitation of boundary assert]
  EnableOutOfBoundAssert guard;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineMost();
  scheduler_utils::rotateLoop(tv4, -1, {tv1, tv2});

  // TODO: b76 is trivially true, we should eliminate it
  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i21 = 0; i21 < T0.size[0]; ++i21) {
    int64_t i30;
    i30 = T0.stride[0] * i21;
    int64_t i44;
    i44 = 3 * i21;
    bool b82;
    b82 = 0 < (T0.size[0] - i21);
    float T1[1];
    float T2[1];
    T1[0] = 0;
    if (b82) {
      T1[0]
         = T0[i30];
    }
    T2[0]
       = T1[0];
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      int64_t i37;
      i37 = i22 + nvfuser_zero;
      int64_t i61;
      i61 = (1 + i22) + nvfuser_zero;
      float T3[1];
      T3[0]
         = T2[0];
      if ((b82 && (i37 < 3))) {
        T4[(i44 + i37)]
           = T3[0];
      }
      T1[0] = 0;
      if ((b82 && (i61 < 3))) {
        T1[0]
           = T0[(i30 + (T0.stride[1] * i61))];
      }
      T2[0]
         = T1[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionLoopRotation1Outer_CUDA) {
  // Please see note [Limitation of boundary assert]
  EnableOutOfBoundAssert guard;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineAllAt(tv4, 1);
  scheduler_utils::rotateLoop(tv4, 0, {tv1, tv2});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  bool b79;
  b79 = 0 < T0.size[0];
  int64_t i128;
  i128 = -T0.size[0];
  float T1[3];
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[i21] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    int64_t i29;
    i29 = i21 + nvfuser_zero;
    if ((b79 && (i29 < 3))) {
      T1[i21]
         = T0[(T0.stride[1] * i29)];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
    T2[i22]
       = T1[i22];
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i24 = 0; i24 < T0.size[0]; ++i24) {
    int64_t i48;
    i48 = 3 * i24;
    int64_t i69;
    i69 = T0.stride[0] + (T0.stride[0] * i24);
    bool b107;
    b107 = 0 < (T0.size[0] - i24);
    bool b136;
    b136 = (i128 + i24) < -1;
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i23 = 0; i23 < 3; ++i23) {
      T3[i23]
         = T2[i23];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i25 = 0; i25 < 3; ++i25) {
      int64_t i41;
      i41 = i25 + nvfuser_zero;
      if ((b107 && (i41 < 3))) {
        T4[(i48 + i41)]
           = T3[i25];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[i21] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i52;
      i52 = i21 + nvfuser_zero;
      if ((b136 && (i52 < 3))) {
        T1[i21]
           = T0[(i69 + (T0.stride[1] * i52))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[i22];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionLoopRotation2_CUDA) {
  // Please see note [Limitation of boundary assert]
  EnableOutOfBoundAssert guard;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, -1});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  for (auto tv : {tv0, tv1, tv2, tv3, tv4}) {
    tv->merge(0);
    tv->split(0, 5);
  }
  inlineAllAt(tv4, 1);
  scheduler_utils::rotateLoop(tv4, 0, {tv1, tv2});

  // TODO: 0 < (i279 - i44) looks ugly, should be i144 < i279. In fact:
  // i36 -> immediate, so i44 -> uniform register
  // also, i279 -> uniform register
  // changing from uniform < uniform into 0 < uniform - uniform does not help
  // anything
  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  int64_t i279;
  i279 = T0.size[0] * T0.size[1];
  int64_t i380;
  i380 = -i279;
  float T1[5];
  float T2[5];
  #pragma unroll
  for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
    T1[i36] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
    int64_t i44;
    i44 = i36 + nvfuser_zero;
    if ((0 < (i279 - i44))) {
      T1[i36]
         = T0[((T0.stride[0] * (i44 / T0.size[1])) + (T0.stride[1] * (i44 % T0.size[1])))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i37 = 0; i37 < 5; ++i37) {
    T2[i37]
       = T1[i37];
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i39 = 0; i39 < (ceilDiv((T0.size[0] * T0.size[1]), 5)); ++i39) {
    int64_t i98;
    i98 = 5 * i39;
    int64_t i246;
    i246 = 5 + i98;
    int64_t i321;
    i321 = i279 - i98;
    int64_t i381;
    i381 = i380 + i98;
    // Alias Allocation - register
    auto& T3 = T1;
    #pragma unroll
    for(nvfuser_index_t i38 = 0; i38 < 5; ++i38) {
      T3[i38]
         = T2[i38];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i40 = 0; i40 < 5; ++i40) {
      int64_t i82;
      i82 = i40 + nvfuser_zero;
      if ((0 < (i321 - i82))) {
        T4[(i98 + i82)]
           = T3[i40];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
      T1[i36] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i36 = 0; i36 < 5; ++i36) {
      int64_t i106;
      i106 = i36 + nvfuser_zero;
      int64_t i247;
      i247 = i246 + i106;
      if (((i381 + i106) < -5)) {
        T1[i36]
           = T0[((T0.stride[0] * (i247 / T0.size[1])) + (T0.stride[1] * (i247 % T0.size[1])))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i37 = 0; i37 < 5; ++i37) {
      T2[i37]
         = T1[i37];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionLoopRotationDoubleBuffered_CUDA) {
  // Please see note [Limitation of boundary assert]
  EnableOutOfBoundAssert guard;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineAllAt(tv4, 1);
  tv1->circularBuffer(5);
  scheduler_utils::rotateLoop(tv4, 0, {tv2});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  int64_t i111;
  i111 = T0.stride[0] * 4;
  int64_t i214;
  i214 = -T0.size[0];
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i24 = 0; i24 < 4; ++i24) {
    int64_t i40;
    i40 = 3 * i24;
    int64_t i51;
    i51 = T0.stride[0] * i24;
    bool b193;
    b193 = 0 < (T0.size[0] - (i24 + nvfuser_zero));
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i40 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i42;
      i42 = i21 + nvfuser_zero;
      if ((b193 && (i42 < 3))) {
        T1[(i40 + i21)]
           = T0[(i51 + (T0.stride[1] * i42))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
    T2[i22]
       = T1[i22];
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i25 = 0; i25 < T0.size[0]; ++i25) {
    int64_t i91;
    i91 = 3 * ((4 + i25) % 5);
    int64_t i113;
    i113 = i111 + (T0.stride[0] * i25);
    int64_t i150;
    i150 = 3 * i25;
    int64_t i173;
    i173 = 3 * ((1 + i25) % 5);
    bool b222;
    b222 = (i214 + i25) < -4;
    bool b243;
    b243 = 0 < (T0.size[0] - i25);
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i91 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i93;
      i93 = i21 + nvfuser_zero;
      if ((b222 && (i93 < 3))) {
        T1[(i91 + i21)]
           = T0[(i113 + (T0.stride[1] * i93))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i23 = 0; i23 < 3; ++i23) {
      T3[i23]
         = T2[i23];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i27 = 0; i27 < 3; ++i27) {
      int64_t i143;
      i143 = i27 + nvfuser_zero;
      if ((b243 && (i143 < 3))) {
        T4[(i150 + i143)]
           = T3[i27];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[(i173 + i22)];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionLoopRotationSelectDoubleBufferLoad_CUDA) {
  // Please see note [Limitation of boundary assert]
  EnableOutOfBoundAssert guard;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  inlineAllAt(tv4, 1);
  tv1->circularBuffer(5);
  scheduler_utils::rotateLoop(tv4, 0, {tv1, tv2});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  NVFUSER_DEFINE_MAGIC_ZERO
  int64_t i119;
  i119 = 4 * T0.stride[0];
  int64_t i220;
  i220 = T0.stride[0] * 5;
  bool b295;
  b295 = 0 < T0.size[0];
  int64_t i357;
  i357 = -T0.size[0];
  bool b361;
  b361 = i357 < -4;
  float T1[15];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[i21] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    int64_t i35;
    i35 = i21 + nvfuser_zero;
    if ((b295 && (i35 < 3))) {
      T1[i21]
         = T0[(T0.stride[1] * i35)];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i24 = 0; i24 < 4; ++i24) {
    int64_t i57;
    i57 = 3 + (3 * i24);
    int64_t i78;
    i78 = T0.stride[0] + (T0.stride[0] * i24);
    bool b333;
    b333 = 0 < (T0.size[0] - ((1 + i24) + nvfuser_zero));
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i57 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i61;
      i61 = i21 + nvfuser_zero;
      if ((b333 && (i61 < 3))) {
        T1[(i57 + i21)]
           = T0[(i78 + (T0.stride[1] * i61))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  float T2[3];
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    T1[(12 + i21)] = 0;
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
    int64_t i109;
    i109 = i21 + nvfuser_zero;
    if ((b361 && (i109 < 3))) {
      T1[(12 + i21)]
         = T0[(i119 + (T0.stride[1] * i109))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll
  for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
    T2[i22]
       = T1[i22];
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  #pragma unroll 1
  for(nvfuser_index_t i25 = 0; i25 < T0.size[0]; ++i25) {
    int64_t i151;
    i151 = 3 * i25;
    int64_t i192;
    i192 = 3 * (i25 % 5);
    int64_t i222;
    i222 = i220 + (T0.stride[0] * i25);
    int64_t i288;
    i288 = 3 * ((1 + i25) % 5);
    bool b383;
    b383 = 0 < (T0.size[0] - i25);
    bool b426;
    b426 = (i357 + i25) < -5;
    float T3[3];
    #pragma unroll
    for(nvfuser_index_t i23 = 0; i23 < 3; ++i23) {
      T3[i23]
         = T2[i23];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i27 = 0; i27 < 3; ++i27) {
      int64_t i144;
      i144 = i27 + nvfuser_zero;
      if ((b383 && (i144 < 3))) {
        T4[(i151 + i144)]
           = T3[i27];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i192 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i196;
      i196 = i21 + nvfuser_zero;
      if ((b426 && (i196 < 3))) {
        T1[(i192 + i21)]
           = T0[(i222 + (T0.stride[1] * i196))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i22 = 0; i22 < 3; ++i22) {
      T2[i22]
         = T1[(i288 + i22)];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

// This is a case similar to matmul, where we have
// tv1 = set(tv0) // cp.async for matmul
// tv2 = set(tv1) // ld.matrix for matmul
// and both are double buffered
TEST_F(NVFuserTest, FusionLoopRotationMultipleDoubleBuffer_CUDA) {
  // Please see note [Limitation of boundary assert]
  EnableOutOfBoundAssert guard;
  Fusion fusion;

  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  tv1->setMemoryType(MemoryType::Shared);

  inlineAllAt(tv4, 1);
  inlineSelectedAt({tv2, tv3, tv4}, tv4, 2);

  tv1->circularBuffer(5);
  tv2->doubleBuffer();
  scheduler_utils::rotateLoop(tv4, 0, {tv2});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T4) {
  alignas(16) extern __shared__ char array[];
  unsigned smem_offset = 0;
  NVFUSER_DEFINE_MAGIC_ZERO
  int64_t i116;
  i116 = T0.stride[0] * 4;
  int64_t i275;
  i275 = -T0.size[0];
  smem_offset = alignBufferSize(smem_offset, 16);
  float* T1 = reinterpret_cast<float*>(array + smem_offset);
  smem_offset += (15 * sizeof(float));
  #pragma unroll
  for(nvfuser_index_t i22 = 0; i22 < 4; ++i22) {
    int64_t i47;
    i47 = 3 * i22;
    int64_t i58;
    i58 = T0.stride[0] * i22;
    bool b254;
    b254 = 0 < (T0.size[0] - (i22 + nvfuser_zero));
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i47 + i21)] = 0;
    }
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i49;
      i49 = i21 + nvfuser_zero;
      if ((b254 && (i49 < 3))) {
        T1[(i47 + i21)]
           = T0[(i58 + (T0.stride[1] * i49))];
      }
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO
  float T2[2];
  T2[0]
     = T1[0];
  #pragma unroll 1
  for(nvfuser_index_t i23 = 0; i23 < T0.size[0]; ++i23) {
    int64_t i96;
    i96 = 3 * ((4 + i23) % 5);
    int64_t i118;
    i118 = i116 + (T0.stride[0] * i23);
    int64_t i161;
    i161 = 1 + (3 * (i23 % 5));
    int64_t i197;
    i197 = 3 * i23;
    bool b283;
    b283 = (i275 + i23) < -4;
    bool b307;
    b307 = 0 < (T0.size[0] - i23);
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      T1[(i96 + i21)] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i21 = 0; i21 < 3; ++i21) {
      int64_t i98;
      i98 = i21 + nvfuser_zero;
      if ((b283 && (i98 < 3))) {
        T1[(i96 + i21)]
           = T0[(i118 + (T0.stride[1] * i98))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(nvfuser_index_t i26 = 0; i26 < 2; ++i26) {
      int64_t i189;
      i189 = i26 + nvfuser_zero;
      T2[((1 + i26) % 2)]
         = T1[(i161 + i26)];
      float T3[1];
      T3[0]
         = T2[(i26 % 2)];
      if ((b307 && (i189 < 3))) {
        T4[(i197 + i189)]
           = T3[0];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T3[1];
    T3[0]
       = T2[0];
    if (b307) {
      T4[(2 + i197)]
         = T3[0];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    T2[0]
       = T1[(3 * ((1 + i23) % 5))];
  }
}
)";
  assertCUDAKernel(&fusion, expected_kernel);

  for (auto n : {0, 1, 99}) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n, 3}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});
    testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}
} // namespace nvfuser
