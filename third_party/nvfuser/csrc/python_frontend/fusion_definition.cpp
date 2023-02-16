#include <instrumentation.h>
#include <python_frontend/fusion_cache.h>
#include <python_frontend/fusion_definition.h>
#include <utils.h>

// Require namespace for perf scope instrumentation
using namespace nvfuser::inst;

namespace nvfuser::python_frontend {

const char* dtypeToPyString(nvfuser::DataType t) {
  switch (t) {
    case nvfuser::DataType::Bool:
      return "DataType.Bool";
    case nvfuser::DataType::Double:
      return "DataType.Double";
    case nvfuser::DataType::Float:
      return "DataType.Float";
    case nvfuser::DataType::Half:
      return "DataType.Half";
    case nvfuser::DataType::BFloat16:
      return "DataType.Bfloat16";
    case nvfuser::DataType::Int:
      return "DataType.Int";
    case nvfuser::DataType::Int32:
      return "DataType.Int32";
    case nvfuser::DataType::ComplexFloat:
      return "DataType.ComplexFloat";
    case nvfuser::DataType::ComplexDouble:
      return "DataType.ComplexDouble";
    case nvfuser::DataType::Null:
      return "DataType.Null";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for data type.");
  return nullptr;
}

bool State::operator==(const State& other) const {
  return (index == other.index) && (stype == other.stype);
}

bool State::operator!=(const State& other) const {
  return (index != other.index) || (stype != other.stype);
}

// Generalized printing of State
std::ostream& operator<<(std::ostream& os, const State& state) {
  if (state.stype == StateType::Scalar) {
    os << "S";
  } else if (state.stype == StateType::Tensor) {
    os << "T";
  } else if (state.stype == StateType::None) {
    os << "None";
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported StateType");
  }
  os << state.index;
  return os;
}

FusionDefinition::FusionDefinition(c10::optional<size_t> id, size_t max_length)
    : max_length_(max_length),
      fusion_id_(id),
      fusion_cache_(FusionCache::get()),
      end_record_(new EndRecord()),
      recording_(),
      recording_state_(),
      fusion_state_(),
      ops(this),
      sched(this) {}

void FusionDefinition::buildFusionIr() {
  FUSER_PERF_SCOPE("FusionDefinition::buildFusionIr");
  auto fusion_guard = nvfuser::FusionGuard(preschedFusion());
  fusion_state_.resize(recording_state_.size(), nullptr);
  for (auto& record : recording_) {
    auto functor = record.get();
    (*functor)(*this);
  }
}

FusionCache* FusionDefinition::fusionCache() const {
  TORCH_INTERNAL_ASSERT(
      fusion_cache_ != nullptr, "FusionCache pointer is null!");
  return fusion_cache_;
}

FusionDefinition* FusionDefinition::setupDefinition() {
  TORCH_CHECK(max_length_ > 0, "Can't make a FusionDefinition with 0 records!");
  TORCH_CHECK(!fusion_id_.has_value(), "Fusion Schedule is already found!");
  fusionCache()->resetTriePtr();
  return this;
}

void FusionDefinition::finalizeDefinition() {
  FUSER_PERF_SCOPE("FusionDefinition::finalizeDefinition");
  auto cache_entry = fusionCache()->queryChildren(end_record_.get());
  if (!cache_entry.has_value()) {
    if (nvfuser::isDebugDumpEnabled(
            nvfuser::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Terminal Node not found.\n";
    }
    fusion_id_ = fusionCache()->createChild(end_record_.get());
    TORCH_CHECK(fusion_id_.has_value(), "Invalid fusion id!");
    fusionCache()->traverseTrie(end_record_.get());

    if (nvfuser::isDebugDumpEnabled(
            nvfuser::DebugDumpOption::PythonDefinition)) {
      print(std::cout);
    }

    buildFusionIr();

    if (nvfuser::isDebugDumpEnabled(
            nvfuser::DebugDumpOption::FusionIrPresched)) {
      printIr();
    }
  } else {
    if (nvfuser::isDebugDumpEnabled(
            nvfuser::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Terminal Node found!\n";
    }
    fusion_id_ = c10::optional<size_t>(cache_entry.value()->fusion_id);
    fusionCache()->traverseTrie(end_record_.get());
  }
}

void FusionDefinition::print(std::ostream& os) const {
  if (fusion_id_.has_value()) {
    os << "\ndef nvfuser_fusion_id" << fusion_id_.value();
  } else {
    os << "\ndef nvfuser_incomplete_fusion";
  }
  os << "(fd : FusionDefinition) -> None :\n";
  os << std::dec;
  for (auto& rec : recording_) {
    os << "    ";
    rec->print(os);
    os << "\n";
  }
  os << std::endl;
}

void FusionDefinition::printIr() {
  preschedFusion()->printMath();
}

std::vector<at::Tensor> FusionDefinition::execute(
    const at::ArrayRef<c10::IValue>& inputs) const {
  TORCH_CHECK(
      fusion_id_.has_value(), "Valid fusion schedule is not available!");
  return fusionCache()
      ->querySchedule(fusion_id_.value())
      .auto_gen_schedules->runFusionWithInputs(inputs);
}

c10::optional<size_t> FusionDefinition::id() const {
  return fusion_id_;
}

Scalar FusionDefinition::defineScalar() {
  FUSER_PERF_SCOPE("FusionDefinition::defineScalar");
  Scalar out(recording_state_.size());
  recording_state_.emplace_back(out(), StateType::Scalar);
  return out;
}

Tensor FusionDefinition::defineTensor(size_t dims) {
  FUSER_PERF_SCOPE("FusionDefinition::defineTensor");
  Tensor out(recording_state_.size(), dims);
  recording_state_.emplace_back(out(), StateType::Tensor);
  return out;
}

void FusionDefinition::defineRecord(RecordFunctor* record) {
  FUSER_PERF_SCOPE("FusionDefinition::defineRecord");
  TORCH_CHECK(
      (recording_.size() + 1) <= max_length_,
      "The fusion definition has exceeded ",
      max_length_,
      "operations.  The max_length for FusionDefintion's might need to be ",
      "increased if the definition is created as expected.");
  recording_.emplace_back(record);
  auto cache_entry = fusionCache()->queryChildren(recording_.back().get());
  // If the Record is found in the cache, the FusionDefinition and the Cache
  // will not share Record given the Record had to be created in order to
  // match it but it also already existed in the cache.
  if (cache_entry.has_value()) {
    if (nvfuser::isDebugDumpEnabled(
            nvfuser::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Record (hash: 0x" << std::hex
                << record->hash() << ") hit in Fusion Cache.\n";
    }
    // The FusionDefinition and the Cache will share the Record
  } else {
    if (nvfuser::isDebugDumpEnabled(
            nvfuser::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Record (hash: 0x" << std::hex
                << record->hash() << ") missed in Fusion Cache.\n";
    }
    fusionCache()->createChild(recording_.back().get());
  }
  fusionCache()->traverseTrie(recording_.back().get());
}

nvfuser::Fusion* FusionDefinition::preschedFusion() {
  TORCH_CHECK(fusion_id_.has_value(), "Invalid fusion id!");
  return fusionCache()->querySchedule(fusion_id_.value()).preschedFusion();
}

void FusionDefinition::addInput(nvfuser::Val* input) {
  preschedFusion()->addInput(input);
}
void FusionDefinition::addOutput(nvfuser::Val* output) {
  preschedFusion()->addOutput(output);
}
void FusionDefinition::aliasOutputToInput(
    nvfuser::Val* output,
    nvfuser::Val* input) {
  preschedFusion()->aliasOutputToInput(output, input);
}

nvfuser::Val* FusionDefinition::getFusionState(size_t index) const {
  return fusion_state_.at(index);
}
void FusionDefinition::setFusionState(size_t index, nvfuser::Val* val) {
  fusion_state_.at(index) = val;
}

State FusionDefinition::recordingState(size_t index) const {
  return recording_state_.at(index);
}

} // namespace nvfuser::python_frontend
