#include <arith.h>
#include <ir_builder.h>
#include <ir_utils.h>
#include <ops/alias.h>
#include <transform_view.h>
#include <type_promotion.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! Transform TensorView according to keep, merge, and split transformations.
//! Squeeze and broadcast transformations are handled separately.
//! It is recommend to use the composite ops view function, which will call
//! the analyzeView function to generate the appropriate transformations.
//!
//! For example:
//! original sizes = [2, 10, 40]
//! new_size = [2, 10, 2, 20]
//! auto analysis = analyzeView(TV0, original_sizes, new_sizes)
//! auto TV1 = TV0->view(analysis.transforms);
//!
//! Transforms = [(Keep I0), (Keep I1), (Split I2 by 2)]
//! Before: TV0[I0, I1, I2]
//! After: TV0[I0, I1, 2, ceilDiv(I2, 2)]
//!
//! orig_tv is the tensor view originally coming in from user for the view
//! operation. This is the tensor view all of the view analysis is relative to.
//! View might be doing squeezes before sending into the view operation, so we
//! want the actual input to the view operation to be potentially after the
//! original view operation.
TensorView* applyViewTransforms(
    TensorView* orig_tv,
    TensorView* post_reduce_tv,
    const AnalyzeViewResult& view_analysis) {
  TORCH_INTERNAL_ASSERT(orig_tv != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(post_reduce_tv != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(
      !post_reduce_tv->hasComputeAt(),
      "Cannot modify rfactor domain after compute at has been set.");

  TORCH_INTERNAL_ASSERT(
      post_reduce_tv->nDims() > 0, "Tried to view a 0-dim TensorView");

  TORCH_INTERNAL_ASSERT(!view_analysis.transforms.empty());

  TensorView* consumer = IrBuilder::create<TensorView>(
      orig_tv->container(),
      orig_tv->domain()->view(view_analysis),
      orig_tv->getDataType().value());

  IrBuilder::create<ViewOp>(orig_tv->container(), consumer, post_reduce_tv);

  return consumer;
}

} // namespace

TensorView* view(TensorView* x, DataType dtype) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  if (x->getDataType() == dtype) {
    return x;
  }

  auto input_type = x->getDataType().value();
  auto input_size = dataTypeSize(input_type);
  auto newsize = dataTypeSize(dtype);

  if (input_size == newsize) {
    return bitCastOp(dtype, x);
  }
  // TODO: support view(dtype) for dtypes where input_size != newsize
  TORCH_INTERNAL_ASSERT(false, "Unsupported reinterpret casting view");
}

TensorView* reshape(
    TensorView* x,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size() ==
      original_sizes.size());

  auto view_analysis = analyzeView(x, original_sizes, new_sizes);

  auto squeezed = std::any_of(
                      view_analysis.squeeze_axes.begin(),
                      view_analysis.squeeze_axes.end(),
                      [](bool s) { return s; })
      ? squeeze(x, view_analysis.squeeze_axes)
      : x;

  auto view = view_analysis.transforms.empty()
      ? squeezed
      : applyViewTransforms(x, squeezed, view_analysis);

  auto bcasted = std::any_of(
                     view_analysis.broadcast_axes.begin(),
                     view_analysis.broadcast_axes.end(),
                     [](bool b) { return b; })
      ? broadcast(view, view_analysis.broadcast_axes)
      : view;

  return bcasted;
}

TensorView* flatten(TensorView* x, int64_t start_dim, int64_t end_dim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  auto inp_domain = TensorDomain::noReductions(x->getMaybeRFactorDomain());
  if (start_dim < 0) {
    start_dim += inp_domain.size();
  }
  if (end_dim < 0) {
    end_dim += inp_domain.size();
  }
  TORCH_CHECK(
      start_dim >= 0 && start_dim < int64_t(inp_domain.size()),
      "Invalid start_dim ",
      start_dim);
  TORCH_CHECK(
      end_dim >= 0 && end_dim < int64_t(inp_domain.size()),
      "Invalid end_dim ",
      end_dim);
  TORCH_CHECK(start_dim <= end_dim, "start_dim must be <= end_dim");

  if (start_dim == end_dim) {
    return x;
  }

  auto out = IrBuilder::create<TensorView>(
      x->container(),
      x->domain()->flatten(start_dim, end_dim),
      x->getDataType().value());

  IrBuilder::create<ViewOp>(out, x);
  return out;
}

TensorView* squeeze(TensorView* x, const std::vector<bool>& to_squeeze) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  auto x_dom = x->domain()->noReductions();
  const auto ndims = static_cast<int>(x_dom.size());

  TORCH_INTERNAL_ASSERT(
      ndims == (int)to_squeeze.size(),
      "Invalid to_squeeze for squeeze: ",
      to_squeeze,
      ". Input tensor: ",
      x->toString());

  std::vector<IterDomain*> out_domain;
  for (const auto idx : c10::irange(ndims)) {
    auto id = x_dom[idx];
    if (to_squeeze[idx]) {
      TORCH_CHECK(
          id->isBroadcast(), "Can not squeeze non-broadcasting dimension(s).");
      TORCH_CHECK(
          !id->hasExpandedExtent(), "Can not squeeze expanded dimension(s).");
      TORCH_CHECK(
          id->extent()->isOneInt(),
          "Can not squeeze dimension(s) with size != 1.");
    } else {
      out_domain.push_back(id->cloneWithoutRFactor());
    }
  }

  auto out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguousContiguity(out_domain)),
      *x->getDataType());

  IrBuilder::create<SqueezeOp>(x->container(), out, x, to_squeeze);

  return out;
}

TensorView* squeeze(TensorView* x, const std::vector<int64_t>& sizes) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  TORCH_INTERNAL_ASSERT(
      ndims == int(sizes.size()),
      "Invalid sizes for squeeze: ",
      sizes,
      ". Input tensor: ",
      x->toString());

  std::vector<bool> to_squeeze(ndims);
  for (const auto idx : c10::irange(sizes.size())) {
    to_squeeze[idx] = (sizes[idx] == 1);
  }
  return squeeze(x, to_squeeze);
}

TensorView* squeeze(TensorView* x, const std::vector<int64_t>& sizes, int dim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  TORCH_INTERNAL_ASSERT(
      ndims == int(sizes.size()),
      "Invalid sizes for squeeze: ",
      sizes,
      ". Input tensor: ",
      x->toString());

  if (dim < 0) {
    dim = ndims + dim;
  }

  TORCH_INTERNAL_ASSERT(
      dim >= 0 && dim < ndims,
      "Invalid position to squeeze: ",
      dim,
      ". Input tensor: ",
      x->toString());

  if (sizes[dim] == 1) {
    std::vector<bool> to_squeeze(ndims, false);
    to_squeeze[dim] = true;
    return squeeze(x, to_squeeze);
  } else {
    return set(x);
  }
}

TensorView* squeeze(
    TensorView* x,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& dims) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  TORCH_INTERNAL_ASSERT(
      ndims == int(sizes.size()),
      "Invalid sizes for squeeze: ",
      sizes,
      ". Input tensor: ",
      x->toString());

  bool is_all_singleton_dimensions = true;

  std::vector<bool> to_squeeze(ndims);
  for (auto dim : dims) {
    if (dim < 0) {
      dim = ndims + dim;
    }

    TORCH_INTERNAL_ASSERT(
        dim >= 0 && dim < ndims,
        "Invalid position to squeeze: ",
        dim,
        ". Input tensor: ",
        x->toString());

    bool is_singleton_dim = (sizes[dim] == 1);
    to_squeeze.at(dim) = is_singleton_dim;
    is_all_singleton_dimensions &= is_singleton_dim;
  }

  if (is_all_singleton_dimensions) {
    return squeeze(x, to_squeeze);
  } else {
    return set(x);
  }
}

TensorView* unsqueeze(TensorView* x, int dim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  if (dim < 0) {
    dim = ndims + dim + 1;
  }

  TORCH_INTERNAL_ASSERT(
      dim >= 0 && dim <= ndims,
      "Invalid position to unsqueeze: ",
      dim,
      ". Input tensor: ",
      x->toString());

  std::vector<bool> broadcast_axes(ndims + 1, false);
  broadcast_axes[dim] = true;
  return broadcast(x, broadcast_axes);
}

TensorView* permute(TensorView* x, const std::vector<int64_t>& new2old) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  if (new2old.size() == 0) {
    return set(x);
  }
  auto inp_domain = TensorDomain::noReductions(x->getMaybeRFactorDomain());
  std::vector<IterDomain*> out_domain(inp_domain.size());

  TORCH_CHECK(
      inp_domain.size() == new2old.size(),
      "The number of dimensions in the tensor input does not match the length",
      " of the desired ordering of dimensions i.e. input.dim() = ",
      inp_domain.size(),
      " is not equal to len(dims) = ",
      new2old.size());

  // Return scalar tensors immediately
  if (inp_domain.size() == 0) {
    return set(x);
  }

  auto normalized_new2old =
      ir_utils::normalizeNew2Old(new2old, inp_domain.size());

  for (const auto i : c10::irange(out_domain.size())) {
    auto in_id = inp_domain[normalized_new2old[i]];
    out_domain[i] = in_id->cloneWithoutRFactor();
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguousContiguity(out_domain)),
      x->getDataType().value());
  IrBuilder::create<TransposeOp>(out_tensor, x, normalized_new2old);
  return out_tensor;
}

TensorView* transpose(TensorView* x, int64_t dim0, int64_t dim1) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  if (dim0 < 0) {
    dim0 = ndims + dim0;
  }

  if (dim1 < 0) {
    dim1 = ndims + dim1;
  }

  TORCH_CHECK(
      dim0 >= 0 && dim0 <= ndims, "Invalid transpose dimension 0: ", dim0);

  TORCH_CHECK(
      dim1 >= 0 && dim1 <= ndims, "Invalid transpose dimension 1: ", dim1);

  std::vector<int64_t> new2old(ndims);
  for (const auto i : c10::irange(ndims)) {
    if (i == dim0) {
      new2old[i] = dim1;
    } else if (i == dim1) {
      new2old[i] = dim0;
    } else {
      new2old[i] = i;
    }
  }
  return permute(x, new2old);
}

TensorView* transpose(TensorView* x) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  TORCH_CHECK(
      ndims <= 2,
      "Expected a tensor with <= 2 dimensions, but it has ",
      ndims,
      "D.");

  // short-circuit: return original tensorview if less than 2 dimensions
  if (ndims < 2) {
    return x;
  }

  return transpose(x, 0, 1);
}

namespace {
Val* simplifiedInt(Val* val) {
  TORCH_INTERNAL_ASSERT(
      val->isConstInt(), "Expecting Const Int's only in this routine.");
  if (val->as<Int>()->value().has_value()) {
    return val;
  }
  return IrBuilder::create<Int>(val->evaluateInt());
}

Val* promoteSize(Val* v1, Val* v2) {
  if (v1 == nullptr) {
    TORCH_INTERNAL_ASSERT(
        v2 == nullptr || v2->isIntegralScalar(),
        "Expecting Int's only in this routine.");
    return v2;
  }
  if (v2 == nullptr) {
    return v1;
  }
  TORCH_INTERNAL_ASSERT(
      v1->isIntegralScalar() && v2->isIntegralScalar(),
      "Expecting Int's only in this routine.");

  if (!v1->isConstInt() && !v2->isConstInt()) {
    return v1;
  } else if (v1->isConstInt() && v2->isConstInt()) {
    TORCH_INTERNAL_ASSERT(
        v1->evaluateInt() == v2->evaluateInt(),
        "Expected sizes of, ",
        v1->toString(),
        " and ",
        v2->toString(),
        " to match but found ",
        v1->evaluateInt(),
        " and ",
        v2->evaluateInt(),
        ".");
    return simplifiedInt(v1);
  } else if (v1->isConstInt()) {
    return simplifiedInt(v1);
  }
  return simplifiedInt(v2);
}

IterType promoteIterType(IterType type1, IterType type2) {
  // Iteration: Default
  // Reduction: Should not appear here
  // Broadcast: Propagated only if type1 and type2 are Broadcast
  // Gather: Converted to Iteration
  // Stride: Shold not appear here
  // VectorComponent: Converted to Iteration

  TORCH_INTERNAL_ASSERT(
      type1 != IterType::Reduction && type1 != IterType::Stride,
      "Invalid IterType: ",
      type1)
  TORCH_INTERNAL_ASSERT(
      type2 != IterType::Reduction && type2 != IterType::Stride,
      "Invalid IterType: ",
      type2);

  // Do not propagate Gather and VectorComponent
  if (type1 == IterType::Gather || type1 == IterType::VectorComponent ||
      type1 == IterType::GatherScatter) {
    type1 = IterType::Iteration;
  }
  if (type2 == IterType::Gather || type2 == IterType::VectorComponent ||
      type2 == IterType::GatherScatter) {
    type2 = IterType::Iteration;
  }

  // At this point, type1 and type2 must be either Iteration or
  // Broadcast
  TORCH_INTERNAL_ASSERT(
      type1 == IterType::Iteration || type1 == IterType::Broadcast,
      "Unexpected IterType: ",
      type1);
  TORCH_INTERNAL_ASSERT(
      type2 == IterType::Iteration || type2 == IterType::Broadcast,
      "Unexpected IterType: ",
      type2);

  if (type1 == IterType::Broadcast) {
    return type2;
  } else {
    return type1;
  }
}

TensorView* newOutputTV(const std::vector<Val*>& vals, DataType dtype) {
  std::vector<TensorView*> tvs;
  for (auto val : vals) {
    if (val->getValType() == ValType::TensorView) {
      tvs.push_back(val->as<TensorView>());
    }
  }
  TORCH_CHECK(
      !tvs.empty(),
      "Tried to create new output TensorView but received empty list.");

  std::vector<IterDomain*> out_domain(
      TensorDomain::noReductions(tvs[0]->getMaybeRFactorDomain()).size(),
      nullptr);

  // For the start and stop offsets, take the maximum of input axes.
  // For now, the offsets of both start and stop are always integer
  // constant, so we can statically compute them. It is unclear
  // whether we would need to support dynamic offsetting, e.g.,
  // shifting by a dynamic offset.
  std::vector<int64_t> start_offsets(out_domain.size(), 0);
  std::vector<int64_t> stop_offsets(out_domain.size(), 0);
  std::vector<Val*> extent_vals(out_domain.size(), nullptr);
  std::vector<Val*> expanded_extent_vals(out_domain.size(), nullptr);
  std::vector<c10::optional<IterType>> iter_types(
      out_domain.size(), c10::nullopt);

  for (auto tv : tvs) {
    auto dom = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
    TORCH_INTERNAL_ASSERT(
        dom.size() == out_domain.size(),
        "Invalid tensor view found while producing an output, it has ",
        dom.size(),
        " dimensions but expected ",
        out_domain.size());
    for (const auto i : c10::irange(dom.size())) {
      if (dom[i]->isBroadcast()) {
        if (dom[i]->hasExpandedExtent()) {
          expanded_extent_vals[i] =
              promoteSize(expanded_extent_vals[i], dom[i]->expandedExtent());
        }
        continue;
      }
      extent_vals[i] = promoteSize(extent_vals[i], dom[i]->extent());
      if (iter_types[i].has_value()) {
        iter_types[i] =
            promoteIterType(iter_types[i].value(), dom[i]->getIterType());
      } else {
        iter_types[i] = dom[i]->getIterType();
      }

      auto start_offset = dom[i]->start()->as<Int>();
      auto stop_offset = dom[i]->stopOffset()->as<Int>();
      // Currently, start is always constant
      TORCH_INTERNAL_ASSERT(
          start_offset->isConstInt(),
          "Invalid IterDomain start: ",
          start_offset);
      TORCH_INTERNAL_ASSERT(
          stop_offset->isConstInt(),
          "Invalid IterDomain stop offset: ",
          stop_offset);
      start_offsets[i] =
          std::max(start_offsets[i], start_offset->evaluateInt());
      stop_offsets[i] = std::max(stop_offsets[i], stop_offset->evaluateInt());
    }
  }
  for (const auto dim_i : c10::irange(out_domain.size())) {
    if (extent_vals[dim_i] != nullptr) {
      TORCH_INTERNAL_ASSERT(
          iter_types[dim_i].has_value(),
          "Could not deduce iter type for new tensor view.");
      out_domain[dim_i] =
          IterDomainBuilder(
              IrBuilder::create<Int>(start_offsets[dim_i]), extent_vals[dim_i])
              .stop_offset(IrBuilder::create<Int>(stop_offsets[dim_i]))
              .iter_type(iter_types[dim_i].value())
              .build();
    } else {
      out_domain[dim_i] = IterDomainBuilder(
                              FusionGuard::getCurFusion()->zeroVal(),
                              FusionGuard::getCurFusion()->oneVal())
                              .expanded_extent(expanded_extent_vals[dim_i])
                              .iter_type(IterType::Broadcast)
                              .build();
    }
  }

  return IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguousContiguity(out_domain)),
      dtype);
}

} // namespace

TensorView* cat(const std::vector<TensorView*>& inputs, int cat_dim) {
  TORCH_CHECK(!inputs.empty(), "No input given");

  const auto dtype = inputs.at(0)->getDataType().value();

  std::vector<std::vector<IterDomain*>> inp_doms;
  int ndims = -1;

  for (auto inp : inputs) {
    TORCH_CHECK(
        inp->getDataType().value() == dtype,
        "Can't concatenate tensors with different data types: ",
        dtype,
        ", ",
        inp->getDataType().value());
    inp_doms.emplace_back(
        TensorDomain::noReductions(inp->getMaybeRFactorDomain()));
    auto i_ndims = static_cast<int>(inp_doms.back().size());
    if (ndims == -1) {
      ndims = i_ndims;
    } else {
      TORCH_CHECK(
          ndims == i_ndims,
          "Unexpected number of dimensions: ",
          inp->toString(),
          ", expected: ",
          ndims);
    }
  }

  if (cat_dim < 0) {
    cat_dim += ndims;
  }

  TORCH_CHECK(
      cat_dim >= 0 && cat_dim < ndims, "Invalid dimension to cat: ", cat_dim);

  Val* concat_ext = nullptr;

  for (const auto i : c10::irange(inputs.size())) {
    // TODO: expanded extent?
    concat_ext = SimplifyingIrBuilder::addExpr(
        concat_ext, inp_doms.at(i).at(cat_dim)->extent());
  }

  // Create a new rfactor tensor by padding the concat dim.
  Val* left_pad = FusionGuard::getCurFusion()->zeroVal();
  Val* right_pad = concat_ext;
  std::vector<Val*> expanded_inputs(inputs.size());
  for (const auto input_idx : c10::irange(inputs.size())) {
    const auto& inp_dom = inp_doms.at(input_idx);
    std::vector<IterDomain*> root_ids(ndims);
    std::vector<IterDomain*> rfactor_ids(ndims);
    std::vector<Val*> pad_widths(ndims * 2);
    for (const auto dim : c10::irange(ndims)) {
      auto inp_id = inp_dom.at(dim);
      auto root_id = inp_id->cloneWithoutRFactor();
      root_ids.at(dim) = root_id;
      if (dim != cat_dim) {
        rfactor_ids.at(dim) = root_id;
        pad_widths.at(dim * 2) = FusionGuard::getCurFusion()->zeroVal();
        pad_widths.at(dim * 2 + 1) = FusionGuard::getCurFusion()->zeroVal();
      } else {
        // TODO: what to do if inp_id is not a normal iterdomain, i.e.,
        // broadcast, partial, etc?
        right_pad = sub(right_pad, inp_id->extent());
        auto expanded_id =
            IterDomain::expand(root_id, left_pad, right_pad, true);
        std::cerr << "Expanded domain: " << expanded_id->toString()
                  << std::endl;
        rfactor_ids.at(dim) = expanded_id;
        pad_widths.at(dim * 2) = left_pad;
        pad_widths.at(dim * 2 + 1) = right_pad;
        left_pad = add(left_pad, inp_id->extent());
      }
    }

    auto expanded_inp = IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(
            root_ids,
            rfactor_ids,
            rfactor_ids,
            TensorDomain::getContiguousContiguity(rfactor_ids)),
        dtype);

    IrBuilder::create<PadOp>(expanded_inp, inputs.at(input_idx), pad_widths);

    expanded_inputs.at(input_idx) = expanded_inp;
  }

  // At this point, left_pad should be equal to concat_ext, and
  // right_pad should be zero

  auto out = newOutputTV(expanded_inputs, dtype);

  IrBuilder::create<CatOp>(out, expanded_inputs, cat_dim);

  return out;
}

TensorView* pad(TensorView* inp, const std::vector<Val*>& pad_widths) {
  const auto& inp_dom = inp->domain()->noReductions();

  const auto ndims = inp->domain()->noReductions().size();

  TORCH_CHECK(
      pad_widths.size() % 2 == 0 && pad_widths.size() / 2 <= ndims,
      "Invalid number of padding widths: ",
      pad_widths.size());

  const auto num_padded_dims = pad_widths.size() / 2;
  const auto num_non_padded_dims = ndims - num_padded_dims;

  std::cerr << "num_padded_dims: " << num_padded_dims << std::endl;

  std::vector<IterDomain*> root_ids(ndims);
  std::vector<IterDomain*> rfactor_ids(ndims);

  int pad_idx = 0;
  for (const auto idx : c10::irange(ndims)) {
    auto consumer_root = inp_dom[idx]->cloneWithoutRFactor();
    root_ids.at(idx) = consumer_root;
    if (idx < num_non_padded_dims) {
      rfactor_ids.at(idx) = consumer_root;
    } else {
      auto left_pad = pad_widths.at(pad_idx++);
      auto right_pad = pad_widths.at(pad_idx++);
      auto padded_id =
          IterDomain::expand(consumer_root, left_pad, right_pad, true);
      std::cerr << "Padded domain: " << padded_id->toString() << std::endl;
      rfactor_ids.at(idx) = padded_id;
    }
  }

  auto out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          root_ids,
          rfactor_ids,
          rfactor_ids,
          TensorDomain::getContiguousContiguity(rfactor_ids)),
      *inp->getDataType());

  IrBuilder::create<PadOp>(out, inp, pad_widths);

  return out;
}

TensorView* slice(TensorView* inp, const std::vector<Slice>& ranges) {
  const auto inp_dom = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  const int ndims = static_cast<int>(inp_dom.size());

  TORCH_CHECK(ndims == static_cast<int>(ranges.size()));

  auto normalize_slice_range = [](const Slice& range, Val* extent) -> Slice {
    auto normalized_range = range;
    if (range.start == nullptr) {
      normalized_range.start = FusionGuard::getCurFusion()->zeroVal();
    }
    if (range.stop == nullptr) {
      normalized_range.stop = extent;
    }
    if (range.step == nullptr) {
      normalized_range.step = FusionGuard::getCurFusion()->oneVal();
    }
    return normalized_range;
  };

  for (auto& range : ranges) {
    // Step not supported yet
    TORCH_CHECK(
        range.step == nullptr || range.step->isOneInt(),
        "Unsupported step: ",
        range.step->toString());
  }

  std::vector<IterDomain*> root_ids(ndims);
  std::vector<IterDomain*> rfactor_ids(ndims);
  std::vector<Slice> normalized_ranges(ndims);

  bool needs_real_slicing = false;
  for (const auto idx : c10::irange(ndims)) {
    auto inp_root_id = inp_dom[idx]->cloneWithoutRFactor();
    auto out_root_id = inp_root_id->cloneWithoutRFactor();
    root_ids.at(idx) = out_root_id;
    auto range = normalize_slice_range(ranges.at(idx), inp_root_id->extent());
    normalized_ranges.push_back(range);
    if (range.start->isZeroInt() && range.stop->sameAs(inp_root_id->extent()) &&
        range.step->isOneInt()) {
      // This dim doesn't need slicing
      rfactor_ids.at(idx) = out_root_id;
    } else {
      auto sliced_id = IterDomain::expand(
          out_root_id,
          IrBuilder::negExpr(range.start),
          sub(range.stop, inp_root_id->extent()),
          true);
      std::cerr << "Sliced domain: " << sliced_id->toString() << std::endl;
      rfactor_ids.at(idx) = sliced_id;
      needs_real_slicing = true;
    }
  }

  // If slicing isn't actually needed, just return a copy
  if (!needs_real_slicing) {
    return set(inp);
  }

  auto out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          root_ids,
          rfactor_ids,
          rfactor_ids,
          TensorDomain::getContiguousContiguity(rfactor_ids)),
      *inp->getDataType());

  IrBuilder::create<SliceOp>(out, inp, normalized_ranges);
  return out;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
