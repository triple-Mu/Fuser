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

TensorView* cat(TensorView* x, TensorView* y, int dim) {
  const auto& x_dom = x->domain()->noReductions();
  const auto& y_dom = y->domain()->noReductions();
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  std::vector<IterDomain*> out_domain;
  for (const auto idx : c10::irange(ndims)) {
    if (idx != dim) {
      out_domain.push_back(x_dom[idx]->cloneWithoutRFactor());
    } else {
      auto x_id = x_dom[idx];
      auto y_id = y_dom[idx];
      for (auto id_to_concat : {x_id, y_id}) {
        TORCH_CHECK(
            !id_to_concat->isBroadcast(),
            "Invalid domain to concat: ",
            id_to_concat->toString());
        // TODO: what about expanded domains?
        // Ignore partial domains for now
        TORCH_CHECK(
            !id_to_concat->maybePartial(),
            "Invalid domain to concat: ",
            id_to_concat->toString());
        auto concatenated_extent = add(x_id->extent(), y_id->extent());
        auto concatenated_id =
            IterDomainBuilder(
                FusionGuard::getCurFusion()->zeroVal(), concatenated_extent)
                .iter_type(IterType::Iteration)
                .build();
        std::cerr << "Concatenated domain: " << concatenated_id->toString()
                  << std::endl;
        out_domain.push_back(concatenated_id);
      }
    }
  }

  auto out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguousContiguity(out_domain)),
      *x->getDataType());

  IrBuilder::create<CatOp>(out, x, y, dim);

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
#if 0
      auto padded_extent =
          add(add(inp_dom[idx]->extent(), left_pad), right_pad);
      auto padded_id =
          IterDomainBuilder(
              FusionGuard::getCurFusion()->zeroVal(), padded_extent)
              .iter_type(IterType::Iteration)
              .build();
#else
      auto padded_id = IterDomain::expand(consumer_root, left_pad, right_pad);
#endif
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

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
