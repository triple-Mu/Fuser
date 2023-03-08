#include <gtest/gtest.h>

#include <expr_simplifier.h>
#include <ops/all_ops.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

namespace nvfuser {
namespace {

// check if x and y are equivalent expressions by checking that x == y
// simplifies to true. We don't use x->sameAs(y), because we want to consider
// a(b+c), ab+ac, (b+c)a, ba+ca as equivalent, but `sameAs` can not do this job.
bool isEquivalent(Val* x, Val* y) {
  return simplifyExpr(eq(x, y))->getBool() == true;
}

// assert that x/y -> z
void assertSimplifiedDiv(Val* x, Val* y, Val* z) {
  auto simplified = simplifyExpr(cpp_div(x, y));
  TORCH_CHECK(
      isEquivalent(simplified, z),
      "Expect ",
      x->toInlineString(),
      " / ",
      y->toInlineString(),
      " to be simplified to ",
      z->toInlineString(),
      ", but get ",
      simplified->toInlineString());
}

// assert that x % y -> z
void assertSimplifiedMod(Val* x, Val* y, Val* z) {
  auto simplified = simplifyExpr(mod(x, y));
  TORCH_CHECK(
      isEquivalent(simplified, z),
      "Expect ",
      x->toInlineString(),
      " % ",
      y->toInlineString(),
      " to be simplified to ",
      z->toInlineString(),
      ", but get ",
      simplified->toInlineString());
}

} // namespace

class ExprSimplifierTest : public NVFuserTest {};

TEST_F(ExprSimplifierTest, AssociativeAndCommutativeReordering_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto a = IrBuilder::create<NamedScalar>("a", DataType::Int);
  auto b = IrBuilder::create<NamedScalar>("b", DataType::Int);
  auto c = IrBuilder::create<NamedScalar>("c", DataType::Int);
  auto d = IrBuilder::create<NamedScalar>("d", DataType::Int);
  auto e = IrBuilder::create<NamedScalar>("e", DataType::Int);
  auto f = IrBuilder::create<NamedScalar>("f", DataType::Int);
  auto three = IrBuilder::create<Int>(3);
  auto five = IrBuilder::create<Int>(5);
  auto six = IrBuilder::create<Int>(6);
  auto eight = IrBuilder::create<Int>(8);

  {
    // ((c * b) + d) + (a + 3)
    auto val = add(add(mul(c, b), d), add(a, three));
    std::vector<VarInfo> variables(4);
    variables[0].variable = a;
    variables[1].variable = b;
    variables[2].variable = c;
    variables[3].variable = d;
    auto simplified = simplifyExpr(val, {variables.begin(), variables.end()});
    // simplify it, expecting to get
    // (((3 + a) + (b * c)) + d)
    auto expect = add(add(add(three, a), mul(b, c)), d);
    TORCH_CHECK(
        expect->sameAs(simplified) && simplified->sameAs(expect),
        "Expect the simplified expression",
        simplified->toInlineString(),
        " to be the same as ",
        expect->toInlineString());
  }

  {
    // ((((c * d) + ((e + f) + 3)) + 3) * ((((a + b) + 3) + 5) + c)) * a
    auto val =
        mul(mul(add(add(mul(c, d), add(add(e, f), three)), three),
                add(add(add(add(a, b), three), five), c)),
            a);
    std::vector<VarInfo> variables(6);
    variables[0].variable = a;
    variables[1].variable = b;
    variables[2].variable = c;
    variables[3].variable = d;
    variables[4].variable = e;
    variables[5].variable = f;
    auto simplified = simplifyExpr(val, {variables.begin(), variables.end()});

    // simplify it, expecting to get
    // (a * (((8 + a) + b) + c)) * (((6 + (c * d)) + e) + f)
    auto expect =
        mul(mul(a, add(add(add(eight, a), b), c)),
            add(add(add(six, mul(c, d)), e), f));
    TORCH_CHECK(
        // Use isEquivalent to check equivalence because distributeMul will
        // expand the expression.
        isEquivalent(simplified, expect),
        "Expect the simplified expression",
        simplified->toInlineString(),
        " to be the same as ",
        expect->toInlineString());
  }
}

TEST_F(ExprSimplifierTest, EliminateTrivialComputation_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto i = IrBuilder::create<NamedScalar>("i", DataType::Int);
  auto d = IrBuilder::create<NamedScalar>("d", DataType::Double);
  auto b = IrBuilder::create<NamedScalar>("b", DataType::Bool);
  auto i0 = IrBuilder::create<Int>(0);
  auto i1 = IrBuilder::create<Int>(1);
  auto i2 = IrBuilder::create<Int>(2);
  auto d0 = IrBuilder::create<Double>(0);
  auto d1 = IrBuilder::create<Double>(1);
  auto d2 = IrBuilder::create<Double>(2);
  auto t = IrBuilder::create<Bool>(true);
  auto f = IrBuilder::create<Bool>(false);

  // 1 * a -> a
  TORCH_CHECK(simplifyExpr(mul(i1, i))->sameAs(i));
  TORCH_CHECK(simplifyExpr(mul(d1, d))->sameAs(d));
  // a * 1 -> a
  TORCH_CHECK(simplifyExpr(mul(i, i1))->sameAs(i));
  TORCH_CHECK(simplifyExpr(mul(d, d1))->sameAs(d));
  // 0 * a -> 0
  TORCH_CHECK(simplifyExpr(mul(i0, i))->sameAs(i0));
  // a * 0 -> 0
  TORCH_CHECK(simplifyExpr(mul(i, i0))->sameAs(i0));

  // 0 + a -> a
  TORCH_CHECK(simplifyExpr(add(i0, i))->sameAs(i));
  TORCH_CHECK(simplifyExpr(add(d0, d))->sameAs(d));
  // a + 0 -> a
  TORCH_CHECK(simplifyExpr(add(i, i0))->sameAs(i));
  TORCH_CHECK(simplifyExpr(add(d, d0))->sameAs(d));

  // true && a -> a
  TORCH_CHECK(simplifyExpr(IrBuilder::andExpr(t, b))->sameAs(b));
  // a && true -> a
  TORCH_CHECK(simplifyExpr(IrBuilder::andExpr(b, t))->sameAs(b));
  // false && a -> false
  TORCH_CHECK(simplifyExpr(IrBuilder::andExpr(f, b))->sameAs(f));
  // a && false -> false
  TORCH_CHECK(simplifyExpr(IrBuilder::andExpr(b, f))->sameAs(f));

  // a && a -> a
  TORCH_CHECK(simplifyExpr(IrBuilder::andExpr(b, b))->sameAs(b));
  // a || a -> a
  TORCH_CHECK(simplifyExpr(IrBuilder::orExpr(b, b))->sameAs(b));

  // true || a -> true
  TORCH_CHECK(simplifyExpr(IrBuilder::orExpr(t, b))->sameAs(t));
  // a || true -> true
  TORCH_CHECK(simplifyExpr(IrBuilder::orExpr(b, t))->sameAs(t));
  // false || a -> a
  TORCH_CHECK(simplifyExpr(IrBuilder::orExpr(f, b))->sameAs(b));
  // a || false -> a
  TORCH_CHECK(simplifyExpr(IrBuilder::orExpr(b, f))->sameAs(b));

  // a / 1 -> a
  TORCH_CHECK(simplifyExpr(cpp_div(i, i1))->sameAs(i));
  TORCH_CHECK(simplifyExpr(cpp_div(d, d1))->sameAs(d));
  // 0 / a -> 0
  auto tdimx = NamedScalar::getParallelDim(ParallelType::TIDx);
  TORCH_CHECK(simplifyExpr(cpp_div(i0, tdimx))->sameAs(i0));
  // a % 1 -> 0
  TORCH_CHECK(simplifyExpr(mod(i, i1))->sameAs(i0));

  // -(-a) -> a
  TORCH_CHECK(simplifyExpr(neg(neg(i)))->sameAs(i));
  TORCH_CHECK(simplifyExpr(notOp(notOp(i)))->sameAs(i));
  TORCH_CHECK(simplifyExpr(notOp(notOp(b)))->sameAs(b));

  // Test constant folding
  TORCH_CHECK(simplifyExpr(add(add(i1, i), i1))->sameAs(add(i, i2)));
  TORCH_CHECK(simplifyExpr(add(add(d1, d), d1))->sameAs(add(d, d2)));

  // Test that FlattenedAssocCommOp::sameAs ignores order
  auto x = IrBuilder::create<NamedScalar>("x", DataType::Int);
  auto y = IrBuilder::create<NamedScalar>("y", DataType::Int);
  TORCH_CHECK(simplifyExpr(sub(mul(x, y), mul(y, x)))->isZeroInt());
}

TEST_F(ExprSimplifierTest, SimplifyDivisibleDivMod_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto one = IrBuilder::create<Int>(1);
  auto two = IrBuilder::create<Int>(2);
  auto three = IrBuilder::create<Int>(3);
  auto six = IrBuilder::create<Int>(6);
  auto a = NamedScalar::getParallelDim(ParallelType::TIDx);
  auto b = NamedScalar::getParallelDim(ParallelType::TIDy);
  auto c = NamedScalar::getParallelDim(ParallelType::TIDz);
  auto d = add(NamedScalar::getParallelIndex(ParallelType::TIDx), one);

  // assert that our system can correctly find that x is a multiple of y and z,
  // and simplify:
  // x % y -> 0
  // x % z -> 0
  // x / y -> z
  // and if x_div_z is true, also test
  // x / z -> y
  auto assertSimplifiedDivMod = [&fusion](Val* x, Val* y, Val* z) {
    assertSimplifiedMod(x, y, fusion.zeroVal());
    assertSimplifiedMod(x, z, fusion.zeroVal());
    assertSimplifiedDiv(x, y, z);
    assertSimplifiedDiv(x, z, y);
  };

  assertSimplifiedDivMod(six, three, two);
  assertSimplifiedDivMod(mul(a, b), a, b);
  assertSimplifiedDivMod(mul(a, b), mul(a, b), one);
  assertSimplifiedDivMod(mul(mul(a, b), c), a, mul(b, c));
  assertSimplifiedDivMod(mul(mul(a, b), c), b, mul(a, c));
  assertSimplifiedDivMod(mul(mul(a, b), c), c, mul(a, b));
  assertSimplifiedDivMod(mul(mul(a, b), c), mul(a, mul(b, c)), one);
  assertSimplifiedDivMod(
      add(mul(mul(a, b), c), mul(mul(a, b), d)), a, add(mul(b, c), mul(b, d)));
  assertSimplifiedDivMod(
      add(mul(mul(a, b), c), mul(mul(a, b), d)), b, add(mul(a, c), mul(a, d)));
  assertSimplifiedDivMod(
      add(mul(mul(a, b), c), mul(mul(a, b), d)), mul(a, b), add(c, d));
  assertSimplifiedDivMod(
      add(mul(add(a, b), c), mul(add(a, b), d)), add(a, b), add(c, d));
  assertSimplifiedDivMod(
      mul(mul(a, b), mul(c, six)), mul(mul(a, b), mul(c, six)), one);
  assertSimplifiedDivMod(mul(mul(a, b), mul(c, six)), mul(a, mul(b, c)), six);
  assertSimplifiedDivMod(
      mul(mul(a, b), mul(c, six)), three, mul(mul(a, b), mul(c, two)));
  assertSimplifiedDivMod(
      mul(mul(a, b), mul(c, six)), mul(mul(a, b), mul(c, three)), two);
  assertSimplifiedDivMod(
      mul(mul(a, b), mul(c, six)), mul(a, mul(c, three)), mul(b, two));
  assertSimplifiedDivMod(mul(add(a, mul(a, c)), b), mul(a, b), add(one, c));
  assertSimplifiedDivMod(
      mul(add(mul(a, b), mul(a, c)), add(mul(b, a), mul(b, d))),
      mul(a, b),
      mul(add(b, c), add(a, d)));
  assertSimplifiedDivMod(
      mul(add(mul(three, b), mul(six, c)), add(mul(b, a), mul(b, d))),
      mul(three, b),
      mul(add(b, mul(two, c)), add(a, d)));
  assertSimplifiedDivMod(
      mul(add(mul(three, b), six), add(mul(b, a), mul(b, d))),
      mul(three, b),
      mul(add(b, two), add(a, d)));
  assertSimplifiedDivMod(
      mul(add(mul(six, b), three), add(mul(b, a), mul(b, d))),
      mul(three, b),
      mul(add(mul(two, b), one), add(a, d)));
  assertSimplifiedDivMod(
      add(mul(mul(a, b), three), mul(mul(b, a), six)),
      mul(mul(three, b), a),
      three);
}

TEST_F(ExprSimplifierTest, SignProve_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  auto assertProvedPositive =
      [&fusion](Val* x, const std::list<VarInfo>& variables = {}) {
        auto proved =
            (simplifyExpr(IrBuilder::gtExpr(x, fusion.zeroVal()), variables)
                 ->getBool() == true) &&
            (simplifyExpr(IrBuilder::geExpr(x, fusion.zeroVal()), variables)
                 ->getBool() == true) &&
            (simplifyExpr(IrBuilder::ltExpr(fusion.zeroVal(), x), variables)
                 ->getBool() == true) &&
            (simplifyExpr(IrBuilder::leExpr(fusion.zeroVal(), x), variables)
                 ->getBool() == true) &&
            (simplifyExpr(IrBuilder::leExpr(x, fusion.zeroVal()), variables)
                 ->getBool() == false) &&
            (simplifyExpr(IrBuilder::ltExpr(x, fusion.zeroVal()), variables)
                 ->getBool() == false) &&
            (simplifyExpr(IrBuilder::geExpr(fusion.zeroVal(), x), variables)
                 ->getBool() == false) &&
            (simplifyExpr(IrBuilder::gtExpr(fusion.zeroVal(), x), variables)
                 ->getBool() == false);
        TORCH_CHECK(proved, "Unable to prove ", x->toInlineString(), " > 0");
      };
  auto assertProvedNonNegative =
      [&fusion](Val* x, const std::list<VarInfo>& variables = {}) {
        auto proved =
            (simplifyExpr(IrBuilder::geExpr(x, fusion.zeroVal()), variables)
                 ->getBool() == true) &&
            (simplifyExpr(IrBuilder::leExpr(fusion.zeroVal(), x), variables)
                 ->getBool() == true) &&
            (simplifyExpr(IrBuilder::ltExpr(x, fusion.zeroVal()), variables)
                 ->getBool() == false) &&
            (simplifyExpr(IrBuilder::gtExpr(fusion.zeroVal(), x), variables)
                 ->getBool() == false);
        TORCH_CHECK(proved, "Unable to prove ", x->toInlineString(), " >= 0");
      };
  auto assertProvedNonZero =
      [&fusion](Val* x, const std::list<VarInfo>& variables = {}) {
        auto proved =
            (simplifyExpr(IrBuilder::neExpr(x, fusion.zeroVal()), variables)
                 ->getBool() == true) &&
            (simplifyExpr(IrBuilder::neExpr(fusion.zeroVal(), x), variables)
                 ->getBool() == true) &&
            (simplifyExpr(IrBuilder::eqExpr(x, fusion.zeroVal()), variables)
                 ->getBool() == false) &&
            (simplifyExpr(IrBuilder::eqExpr(fusion.zeroVal(), x), variables)
                 ->getBool() == false);
        TORCH_CHECK(proved, "Unable to prove ", x->toInlineString(), " != 0");
      };

  assertProvedPositive(NamedScalar::getParallelDim(ParallelType::TIDx));
  assertProvedPositive(NamedScalar::getParallelDim(ParallelType::TIDy));
  assertProvedPositive(NamedScalar::getParallelDim(ParallelType::TIDz));
  assertProvedNonNegative(NamedScalar::getParallelDim(ParallelType::TIDx));
  assertProvedNonNegative(NamedScalar::getParallelDim(ParallelType::TIDy));
  assertProvedNonNegative(NamedScalar::getParallelDim(ParallelType::TIDz));
  assertProvedNonZero(NamedScalar::getParallelDim(ParallelType::TIDx));
  assertProvedNonZero(NamedScalar::getParallelDim(ParallelType::TIDy));
  assertProvedNonZero(NamedScalar::getParallelDim(ParallelType::TIDz));

  assertProvedNonNegative(NamedScalar::getParallelIndex(ParallelType::TIDx));
  assertProvedNonNegative(NamedScalar::getParallelIndex(ParallelType::TIDy));
  assertProvedNonNegative(NamedScalar::getParallelIndex(ParallelType::TIDz));

  auto zero = fusion.zeroVal();
  auto one = fusion.oneVal();
  auto two = IrBuilder::newConstant(2, DataType::Int);

  assertProvedPositive(one);
  assertProvedPositive(two);
  assertProvedNonNegative(one);
  assertProvedNonNegative(two);
  assertProvedNonZero(one);
  assertProvedNonZero(two);

  assertProvedNonNegative(
      IrBuilder::create<NamedScalar>("T123.size[3]", DataType::Int));
  assertProvedNonNegative(
      IrBuilder::create<NamedScalar>("T123.stride[3]", DataType::Int));

  auto a = IrBuilder::newScalar(DataType::Int);
  VarInfo ainfo{a, zero, two, one};
  auto b = IrBuilder::newScalar(DataType::Int);
  VarInfo binfo{b, zero, two, one};
  auto c = IrBuilder::newScalar(DataType::Int);
  VarInfo cinfo{c, zero, two, one};
  auto d = IrBuilder::newScalar(DataType::Int);
  VarInfo dinfo{d, zero, two, one};
  auto variables = {ainfo, binfo, cinfo, dinfo};

  assertProvedNonNegative(a, variables);
  assertProvedNonNegative(b, variables);
  assertProvedNonNegative(c, variables);
  assertProvedNonNegative(d, variables);
  assertProvedNonNegative(add(a, b), variables);
  assertProvedNonNegative(add(a, add(b, c)), variables);
  assertProvedNonNegative(mul(add(a, d), add(b, c)), variables);
  assertProvedNonNegative(mul(add(a, two), add(b, c)), variables);

  assertProvedPositive(add(add(a, two), add(b, c)), variables);
  assertProvedNonZero(add(add(a, two), add(b, c)), variables);

  assertProvedNonNegative(
      cpp_div(add(d, one), add(add(a, two), add(b, c))), variables);
  assertProvedNonNegative(
      mod(add(d, one), add(add(a, two), add(b, c))), variables);
}

TEST_F(ExprSimplifierTest, EquivalenceSimplification_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto assertProvedEquiv = [](Val* x, Val* y) {
    auto proved = (simplifyExpr(IrBuilder::eqExpr(x, y))->getBool() == true) &&
        (simplifyExpr(IrBuilder::neExpr(x, y))->getBool() == false);
    TORCH_CHECK(
        proved,
        "Unable to prove ",
        x->toInlineString(),
        " == ",
        y->toInlineString());
  };

  auto a = IrBuilder::create<NamedScalar>("a", DataType::Int);
  auto b = IrBuilder::create<NamedScalar>("b", DataType::Int);
  auto c = IrBuilder::create<NamedScalar>("c", DataType::Int);

  assertProvedEquiv(a, a);
  assertProvedEquiv(mul(a, b), mul(b, a));
  assertProvedEquiv(mod(mul(a, c), b), mod(mul(c, a), b));
}

TEST_F(ExprSimplifierTest, CancelDivMod_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto one = IrBuilder::create<Int>(1);
  auto two = IrBuilder::create<Int>(2);
  auto three = IrBuilder::create<Int>(3);
  auto five = IrBuilder::create<Int>(5);
  auto six = IrBuilder::create<Int>(6);
  auto fifteen = IrBuilder::create<Int>(15);
  auto a = NamedScalar::getParallelDim(ParallelType::TIDx);
  auto b = NamedScalar::getParallelDim(ParallelType::TIDy);
  auto c = NamedScalar::getParallelDim(ParallelType::TIDz);
  assertSimplifiedDiv(
      mul(six, mul(a, c)),
      mul(fifteen, mul(a, b)),
      cpp_div(mul(two, c), mul(five, b)));
  assertSimplifiedMod(
      mul(six, mul(a, c)),
      mul(fifteen, mul(a, b)),
      mul(mod(mul(two, c), mul(five, b)), mul(three, a)));
  assertSimplifiedDiv(
      mul(three, a), mul(fifteen, mul(a, b)), cpp_div(one, mul(five, b)));
  assertSimplifiedMod(
      mul(three, a),
      mul(fifteen, mul(a, b)),
      mul(mod(one, mul(five, b)), mul(three, a)));
}

TEST_F(ExprSimplifierTest, DistributeDivisibleDivMod_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto a = NamedScalar::getParallelDim(ParallelType::TIDx);
  auto b = NamedScalar::getParallelDim(ParallelType::TIDy);
  auto c = NamedScalar::getParallelDim(ParallelType::TIDz);

  assertSimplifiedDiv(add(mul(a, b), c), a, add(b, cpp_div(c, a)));
  assertSimplifiedMod(add(mul(a, b), c), a, mod(c, a));
}

TEST_F(ExprSimplifierTest, DistributeMul_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto a = IrBuilder::create<NamedScalar>("a", DataType::Int);
  auto b = IrBuilder::create<NamedScalar>("b", DataType::Int);
  auto c = IrBuilder::create<NamedScalar>("c", DataType::Int);
  auto d = IrBuilder::create<NamedScalar>("d", DataType::Int);

  TORCH_CHECK(isEquivalent(mul(a, add(b, c)), add(mul(a, b), mul(a, c))));
  TORCH_CHECK(isEquivalent(
      mul(a, add(add(b, c), d)), add(add(mul(a, b), mul(a, c)), mul(a, d))));
}

TEST_F(ExprSimplifierTest, FundamentalDivisionWithRemainderProperty_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto a = IrBuilder::create<NamedScalar>("a", DataType::Int);
  auto b = IrBuilder::create<NamedScalar>("b", DataType::Int);
  auto c = IrBuilder::create<NamedScalar>("T1.size[0]", DataType::Int);
  auto d = IrBuilder::create<NamedScalar>("T1.size[1]", DataType::Int);

  TORCH_CHECK(isEquivalent(add(mul(cpp_div(a, c), c), mod(a, c)), a));
  TORCH_CHECK(
      isEquivalent(add(add(b, mul(cpp_div(a, c), c)), mod(a, c)), add(a, b)));
  TORCH_CHECK(isEquivalent(
      add(mul(cpp_div(a, c), mul(c, d)), mul(d, mod(a, c))), mul(a, d)));
  TORCH_CHECK(isEquivalent(
      add(add(b, mul(cpp_div(a, c), mul(c, d))), mul(d, mod(a, c))),
      add(mul(a, d), b)));
}

TEST_F(ExprSimplifierTest, ReducePredicateRegisterUsage_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto a = IrBuilder::create<NamedScalar>("a", DataType::Int);
  auto b = IrBuilder::create<NamedScalar>("b", DataType::Int);
  auto u1 = IrBuilder::create<NamedScalar>("u1", DataType::Int);
  auto u2 = IrBuilder::create<NamedScalar>("u2", DataType::Int);
  auto tidx = NamedScalar::getParallelIndex(ParallelType::TIDx);
  auto zero = fusion.zeroVal();
  auto one = fusion.oneVal();
  auto five = IrBuilder::create<Int>(5);
  auto neg_five = IrBuilder::create<Int>(-5);

  auto unroll_gp1 = mul(tidx, u1);
  auto unroll_uniform1 = mul(a, u1);
  auto unroll_imm1 = mul(five, u1);
  auto unroll_gp2 = mul(tidx, u2);
  auto unroll_uniform2 = mul(a, u2);
  auto unroll_imm2 = mul(five, u2);

  std::list<VarInfo> variables{
      VarInfo{u1, zero, five, one, true}, VarInfo{u2, zero, five, one, true}};

  // unroll + other == unroll + other
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), add(five, unroll_gp2));
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), add(five, unroll_uniform2));
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), add(unroll_uniform2, five));
    auto v3_simplified =
        eq(add(tidx, neg_five), add(neg(unroll_uniform1), unroll_uniform2));
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), add(unroll_imm2, five));
    auto v4_simplified =
        eq(add(tidx, neg_five), add(neg(unroll_imm1), unroll_imm2));
    TORCH_CHECK(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), add(unroll_imm2, five));
    auto v5_simplify = eq(add(a, neg_five), add(neg(unroll_imm1), unroll_imm2));
    TORCH_CHECK(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll + other == unroll
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), unroll_gp2);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), unroll_uniform2);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), unroll_uniform2);
    auto v3_simplified = eq(tidx, add(neg(unroll_uniform1), unroll_uniform2));
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), unroll_imm2);
    auto v4_simplified = eq(tidx, add(neg(unroll_imm1), unroll_imm2));
    TORCH_CHECK(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), unroll_imm2);
    auto v5_simplify = eq(a, add(neg(unroll_imm1), unroll_imm2));
    TORCH_CHECK(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll + other == other
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), five);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), five);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), five);
    auto v3_simplified = eq(add(tidx, neg_five), neg(unroll_uniform1));
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), five);
    auto v4_simplified = eq(add(tidx, neg_five), neg(unroll_imm1));
    TORCH_CHECK(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), five);
    auto v5_simplify = eq(add(a, neg_five), neg(unroll_imm1));
    TORCH_CHECK(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll + other == 0
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), zero);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), zero);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), zero);
    auto v3_simplified = eq(tidx, neg(unroll_uniform1));
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), zero);
    auto v4_simplified = eq(tidx, neg(unroll_imm1));
    TORCH_CHECK(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), zero);
    auto v5_simplify = eq(a, neg(unroll_imm1));
    TORCH_CHECK(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll == other
  {
    auto v1 = eq(unroll_gp1, five);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(unroll_uniform1, five);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));
    auto v3 = eq(unroll_imm1, five);
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3));
  }

  // other == unroll
  {
    auto v1 = eq(five, unroll_gp1);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(five, unroll_uniform1);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));
    auto v3 = eq(five, unroll_imm1);
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3));
  }

  // other == other
  {
    auto v1 = eq(five, tidx);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(five, a);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));
  }

  // unroll == unroll
  {
    auto v1 = eq(unroll_gp2, unroll_gp1);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(unroll_gp2, unroll_uniform1);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));
    auto v3 = eq(unroll_gp2, unroll_imm1);
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3));
    auto v4 = eq(unroll_uniform2, unroll_gp1);
    TORCH_CHECK(simplifyExpr(v4, variables)->sameAs(v4));
    auto v5 = eq(unroll_uniform2, unroll_uniform1);
    TORCH_CHECK(simplifyExpr(v5, variables)->sameAs(v5));
    auto v6 = eq(unroll_uniform2, unroll_imm1);
    TORCH_CHECK(simplifyExpr(v6, variables)->sameAs(v6));
    auto v7 = eq(unroll_imm2, unroll_gp1);
    TORCH_CHECK(simplifyExpr(v7, variables)->sameAs(v7));
    auto v8 = eq(unroll_imm2, unroll_uniform1);
    TORCH_CHECK(simplifyExpr(v8, variables)->sameAs(v8));
    auto v9 = eq(unroll_imm2, unroll_imm1);
    TORCH_CHECK(simplifyExpr(v9, variables)->sameAs(v9));
  }
}

} // namespace nvfuser
