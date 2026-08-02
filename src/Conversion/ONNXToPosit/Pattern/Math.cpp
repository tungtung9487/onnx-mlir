#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/Posit/PositOps.h"
#include "src/Conversion/ONNXToPosit/TypeConverters.hpp"
#include <universal/utility/compiler.hpp>
#include <universal/utility/architecture.hpp>
#include <universal/utility/bit_cast.hpp>
#include <universal/utility/long_double.hpp>
#include <universal/traits/number_traits.hpp>
#include <universal/traits/arithmetic_traits.hpp>
#include <universal/common/number_traits_reports.hpp>
#include <universal/number/posit/posit.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>


using namespace mlir;
namespace onnx_mlir {
namespace { // anonymous namespace for internal implementation

static unsigned getPositStorageBitWidth(unsigned nbits) {
  if (nbits <= 8)
    return 8;
  if (nbits <= 16)
    return 16;
  if (nbits <= 32)
    return 32;
  if (nbits <= 64)
    return 64;
  return nbits;
}

static int64_t computeQAlignKey(Operation *op, int64_t axis, bool inputSigned,
                                bool hasZeroPoint, int64_t scaleLen,
                                int64_t zpLen, double scale,
                                int64_t zeroPoint) {
  std::string locStr;
  llvm::raw_string_ostream os(locStr);
  op->getLoc().print(os);
  os.flush();
  int64_t scaleKey = static_cast<int64_t>(std::llround(scale * 1.0e9));
  llvm::hash_code h = llvm::hash_combine(
      locStr, axis, inputSigned, hasZeroPoint, scaleLen, zpLen, scaleKey,
      zeroPoint);
  uint64_t u = static_cast<uint64_t>(h) & 0x7fffffffffffffffULL;
  if (u == 0)
    u = 1;
  return static_cast<int64_t>(u);
}

static int64_t computeTensorQAlignKey(Operation *op) {
  return computeQAlignKey(
      op, /*axis=*/-1, /*inputSigned=*/true, /*hasZeroPoint=*/false,
      /*scaleLen=*/1, /*zpLen=*/0, /*scale=*/1.0, /*zeroPoint=*/0);
}


static Operation *createGenericOpForQLinearConv(OpBuilder &rewriter,
    Location loc, StringRef name, TypeRange resultTypes, ValueRange operands,
    ArrayRef<NamedAttribute> attrs = {}) {
  OperationState st(loc, name);
  st.addOperands(operands);
  st.addTypes(resultTypes);
  st.addAttributes(attrs);
  return rewriter.create(st);
}

static Value createONNXTensorConstant(OpBuilder &rewriter, Location loc,
                                      Type resultType, Attribute valueAttr) {
  OperationState st(loc, "onnx.Constant");
  st.addTypes({resultType});
  st.addAttribute("value", valueAttr);
  st.addAttribute("onnx-mlir.posit-helper-const", rewriter.getUnitAttr());
  return rewriter.create(st)->getResult(0);
}

static Value stripUnrealizedCast(Value v) {
  Value cur = v;
  while (auto cast = cur.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() != 1)
      break;
    cur = cast.getOperand(0);
  }
  return cur;
}

static ElementsAttr getElementsAttrFromValue(Value v) {
  Value base = stripUnrealizedCast(v);
  if (auto cst = base.getDefiningOp<mlir::ONNXConstantOp>())
    return llvm::dyn_cast_or_null<ElementsAttr>(cst->getAttr("value"));
  if (auto cst = base.getDefiningOp<mlir::arith::ConstantOp>())
    return llvm::dyn_cast<ElementsAttr>(cst.getValue());
  if (auto cst = base.getDefiningOp<mlir::posit::ConstantOp>())
    return llvm::dyn_cast_or_null<ElementsAttr>(cst->getAttr("value"));
  return nullptr;
}

static LogicalResult collectFPValuesAsDouble(ElementsAttr elements,
                                             SmallVectorImpl<double> &out) {
  if (auto denseFP = llvm::dyn_cast<DenseFPElementsAttr>(elements)) {
    out.reserve(denseFP.getNumElements());
    for (APFloat apf : denseFP.getValues<APFloat>())
      out.push_back(apf.convertToDouble());
    return success();
  }

  if (auto resF32 = llvm::dyn_cast<mlir::detail::DenseResourceElementsAttrBase<float>>(elements)) {
    auto arr = resF32.tryGetAsArrayRef();
    if (!arr)
      return failure();
    out.reserve(arr->size());
    for (float v : *arr)
      out.push_back(static_cast<double>(v));
    return success();
  }

  if (auto resF64 = llvm::dyn_cast<mlir::detail::DenseResourceElementsAttrBase<double>>(elements)) {
    auto arr = resF64.tryGetAsArrayRef();
    if (!arr)
      return failure();
    out.reserve(arr->size());
    for (double v : *arr)
      out.push_back(v);
    return success();
  }

  return failure();
}


static bool parseEnvBoolLocal(const char *name, bool fallback = false) {
  const char *e = std::getenv(name);
  if (!e || !*e)
    return fallback;
  std::string v(e);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return v == "1" || v == "true" || v == "on" || v == "yes" || v == "y";
}

static double parseEnvDoubleLocal(const char *name, double fallback) {
  const char *e = std::getenv(name);
  if (!e || !*e)
    return fallback;
  char *end = nullptr;
  double v = std::strtod(e, &end);
  if (!end || end == e)
    return fallback;
  return v;
}

static int parseEnvIntLocal(const char *name, int fallback) {
  const char *e = std::getenv(name);
  if (!e || !*e)
    return fallback;
  char *end = nullptr;
  long v = std::strtol(e, &end, 10);
  if (!end || end == e)
    return fallback;
  return static_cast<int>(v);
}

static bool isPositConstDebugEnabled() {
  return parseEnvBoolLocal("POSIT_CONST_DEBUG", false) ||
         parseEnvBoolLocal("ONNX_MLIR_POSIT_CONST_DEBUG", false);
}

static unsigned positConstAlpsJobs() {
  int jobs = parseEnvIntLocal(
      "ONNX_MLIR_POSIT_CONST_ALPS_JOBS",
      parseEnvIntLocal("POSIT_CONST_ALPS_JOBS", 1));
  if (jobs <= 0)
    jobs = 1;
  return static_cast<unsigned>(jobs);
}

// Decode a standard posit raw bit pattern. This is intentionally local to the
// compiler pass so nqdq/f32 constants can be emitted directly as compact posit
// raw-bit tensors instead of as f32 constants followed by posit.from_f32.
template <unsigned NBits, unsigned ES>
static uint64_t universalEncodeRaw(double x) {
  if (!std::isfinite(x))
    return 1ULL << (NBits - 1); // NaR for NaN/Inf constants.
  sw::universal::posit<NBits, ES> p;
  p = x;
  return static_cast<uint64_t>(p.bits().to_ull());
}

static bool universalEncodeRawDispatch(double x, unsigned nbits, unsigned es,
                                       uint64_t &rawOut) {
  switch (nbits) {
  case 8:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<8, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<8, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<8, 2>(x);
      return true;
    }
    break;
  case 4:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<4, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<4, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<4, 2>(x);
      return true;
    }
    break;
  case 5:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<5, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<5, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<5, 2>(x);
      return true;
    }
    break;
  case 6:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<6, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<6, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<6, 2>(x);
      return true;
    }
    break;
  case 7:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<7, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<7, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<7, 2>(x);
      return true;
    }
    break;
  case 9:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<9, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<9, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<9, 2>(x);
      return true;
    }
    break;
  case 10:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<10, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<10, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<10, 2>(x);
      return true;
    }
    break;
  case 11:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<11, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<11, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<11, 2>(x);
      return true;
    }
    break;
  case 12:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<12, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<12, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<12, 2>(x);
      return true;
    }
    break;
  case 13:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<13, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<13, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<13, 2>(x);
      return true;
    }
    break;
  case 14:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<14, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<14, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<14, 2>(x);
      return true;
    }
    break;
  case 15:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<15, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<15, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<15, 2>(x);
      return true;
    }
    break;
  case 16:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<16, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<16, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<16, 2>(x);
      return true;
    }
    break;
  case 32:
    switch (es) {
    case 0:
      rawOut = universalEncodeRaw<32, 0>(x);
      return true;
    case 1:
      rawOut = universalEncodeRaw<32, 1>(x);
      return true;
    case 2:
      rawOut = universalEncodeRaw<32, 2>(x);
      return true;
    }
    break;
  }
  return false;
}

static bool universalRoundToDoubleDispatch(double x, unsigned nbits, unsigned es,
                                           double &roundedOut,
                                           uint64_t *rawOut = nullptr) {
  auto roundOne = [&](auto nbitsTag, auto esTag) -> bool {
    constexpr unsigned N = decltype(nbitsTag)::value;
    constexpr unsigned E = decltype(esTag)::value;
    sw::universal::posit<N, E> p;
    p = x;
    roundedOut = static_cast<double>(p);
    if (rawOut)
      *rawOut = static_cast<uint64_t>(p.bits().to_ull());
    return true;
  };

  switch (nbits) {
  case 8:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 8>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 8>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 8>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 4:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 4>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 4>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 4>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 5:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 5>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 5>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 5>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 6:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 6>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 6>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 6>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 7:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 7>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 7>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 7>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 9:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 9>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 9>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 9>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 10:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 10>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 10>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 10>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 11:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 11>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 11>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 11>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 12:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 12>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 12>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 12>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 13:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 13>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 13>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 13>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 14:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 14>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 14>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 14>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 15:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 15>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 15>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 15>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 16:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 16>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 16>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 16>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  case 32:
    switch (es) {
    case 0:
      return roundOne(std::integral_constant<unsigned, 32>{},
          std::integral_constant<unsigned, 0>{});
    case 1:
      return roundOne(std::integral_constant<unsigned, 32>{},
          std::integral_constant<unsigned, 1>{});
    case 2:
      return roundOne(std::integral_constant<unsigned, 32>{},
          std::integral_constant<unsigned, 2>{});
    }
    break;
  }
  return false;
}

struct BuildTimeConstCompandDecision {
  bool useAlps = false;
  int compandMode = 0;
  double theta = 1.0;
  double gamma = 0.0;
  bool useGp = false;
  int gpRs = 7;
  int gpSc = 0;
  double directScore = std::numeric_limits<double>::infinity();
  double chosenScore = std::numeric_limits<double>::infinity();
  SmallVector<uint64_t, 8> rawBits;
};

struct BuildTimePerAxisConstDecision {
  int64_t axis = 0;
  SmallVector<int64_t, 8> compandModes;
  SmallVector<double, 8> thetas;
  SmallVector<double, 8> gammas;
  SmallVector<int64_t, 8> gpEnabled;
  SmallVector<int64_t, 8> gpRs;
  SmallVector<int64_t, 8> gpSc;
  SmallVector<uint64_t, 8> rawBits;
};

struct BuildTimeGPConfig {
  bool enabled = false;
  int rs = 7;
  int sc = 0;
};

static std::vector<int> parseCsvIntsLocal(const char *envName);

static void appendUniqueClampedInts(std::vector<int> &dst,
                                    const std::vector<int> &src, int lo,
                                    int hi) {
  for (int v : src) {
    v = std::min(hi, std::max(lo, v));
    if (std::find(dst.begin(), dst.end(), v) == dst.end())
      dst.push_back(v);
  }
}

static std::vector<int> resolveSweepInts(
    const std::string &fmtUpper, const char *fmtPrefix1, const char *fmtPrefix2,
    const char *fmtPrefix3, const char *p8Prefix1, const char *p8Prefix2,
    const char *p8Prefix3, const char *genericPrefix1, const char *genericPrefix2,
    const char *genericPrefix3, int fallbackValue, int lo, int hi) {
  std::vector<int> vals;
  const std::string valuesFmt1 = std::string(fmtPrefix1) + "_" + fmtUpper;
  const std::string valuesFmt2 = std::string(fmtPrefix2) + "_" + fmtUpper;
  const std::string valuesFmt3 = std::string(fmtPrefix3) + "_" + fmtUpper;
  appendUniqueClampedInts(vals, parseCsvIntsLocal(valuesFmt1.c_str()), lo, hi);
  appendUniqueClampedInts(vals, parseCsvIntsLocal(valuesFmt2.c_str()), lo, hi);
  appendUniqueClampedInts(vals, parseCsvIntsLocal(valuesFmt3.c_str()), lo, hi);
  if (!vals.empty())
    return vals;

  const std::string valuesP8_1 = std::string(p8Prefix1);
  const std::string valuesP8_2 = std::string(p8Prefix2);
  const std::string valuesP8_3 = std::string(p8Prefix3);
  appendUniqueClampedInts(vals, parseCsvIntsLocal(valuesP8_1.c_str()), lo, hi);
  appendUniqueClampedInts(vals, parseCsvIntsLocal(valuesP8_2.c_str()), lo, hi);
  appendUniqueClampedInts(vals, parseCsvIntsLocal(valuesP8_3.c_str()), lo, hi);
  if (!vals.empty())
    return vals;

  const std::string valuesGen1 = std::string(genericPrefix1);
  const std::string valuesGen2 = std::string(genericPrefix2);
  const std::string valuesGen3 = std::string(genericPrefix3);
  appendUniqueClampedInts(vals, parseCsvIntsLocal(valuesGen1.c_str()), lo, hi);
  appendUniqueClampedInts(vals, parseCsvIntsLocal(valuesGen2.c_str()), lo, hi);
  appendUniqueClampedInts(vals, parseCsvIntsLocal(valuesGen3.c_str()), lo, hi);
  if (!vals.empty())
    return vals;

  const std::string minFmt1 = std::string(fmtPrefix1) + "_MIN_" + fmtUpper;
  const std::string maxFmt1 = std::string(fmtPrefix1) + "_MAX_" + fmtUpper;
  const std::string minFmt2 = std::string(fmtPrefix2) + "_MIN_" + fmtUpper;
  const std::string maxFmt2 = std::string(fmtPrefix2) + "_MAX_" + fmtUpper;
  const std::string minFmt3 = std::string(fmtPrefix3) + "_MIN_" + fmtUpper;
  const std::string maxFmt3 = std::string(fmtPrefix3) + "_MAX_" + fmtUpper;
  const std::string minP8_1 = std::string(p8Prefix1) + "_MIN_P8";
  const std::string maxP8_1 = std::string(p8Prefix1) + "_MAX_P8";
  const std::string minP8_2 = std::string(p8Prefix2) + "_MIN_P8";
  const std::string maxP8_2 = std::string(p8Prefix2) + "_MAX_P8";
  const std::string minP8_3 = std::string(p8Prefix3) + "_MIN_P8";
  const std::string maxP8_3 = std::string(p8Prefix3) + "_MAX_P8";
  const std::string minGen1 = std::string(genericPrefix1) + "_MIN";
  const std::string maxGen1 = std::string(genericPrefix1) + "_MAX";
  const std::string minGen2 = std::string(genericPrefix2) + "_MIN";
  const std::string maxGen2 = std::string(genericPrefix2) + "_MAX";
  const std::string minGen3 = std::string(genericPrefix3) + "_MIN";
  const std::string maxGen3 = std::string(genericPrefix3) + "_MAX";

  int minV = parseEnvIntLocal(
      minFmt1.c_str(),
      parseEnvIntLocal(
          minFmt2.c_str(),
          parseEnvIntLocal(
              minFmt3.c_str(),
              parseEnvIntLocal(
                  minP8_1.c_str(),
                  parseEnvIntLocal(
                      minP8_2.c_str(),
                      parseEnvIntLocal(
                          minP8_3.c_str(),
                          parseEnvIntLocal(
                              minGen1.c_str(),
                              parseEnvIntLocal(
                                  minGen2.c_str(),
                                  parseEnvIntLocal(minGen3.c_str(),
                                                   fallbackValue)))))))));

  int maxV = parseEnvIntLocal(
      maxFmt1.c_str(),
      parseEnvIntLocal(
          maxFmt2.c_str(),
          parseEnvIntLocal(
              maxFmt3.c_str(),
              parseEnvIntLocal(
                  maxP8_1.c_str(),
                  parseEnvIntLocal(
                      maxP8_2.c_str(),
                      parseEnvIntLocal(
                          maxP8_3.c_str(),
                          parseEnvIntLocal(
                              maxGen1.c_str(),
                              parseEnvIntLocal(
                                  maxGen2.c_str(),
                                  parseEnvIntLocal(maxGen3.c_str(),
                                                   fallbackValue)))))))));

  minV = std::min(hi, std::max(lo, minV));
  maxV = std::min(hi, std::max(lo, maxV));
  if (minV > maxV)
    std::swap(minV, maxV);
  for (int v = minV; v <= maxV; ++v)
    vals.push_back(v);
  if (vals.empty())
    vals.push_back(std::min(hi, std::max(lo, fallbackValue)));
  return vals;
}

static std::vector<std::string> parseCsvLowerTokensLocal(const char *envName) {
  std::vector<std::string> out;
  const char *e = std::getenv(envName);
  if (!e || !*e)
    return out;
  std::string s(e);
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(',', start);
    std::string tok = s.substr(start, end == std::string::npos ? std::string::npos
                                                                : end - start);
    size_t b = 0;
    while (b < tok.size() && std::isspace(static_cast<unsigned char>(tok[b])))
      ++b;
    size_t epos = tok.size();
    while (epos > b && std::isspace(static_cast<unsigned char>(tok[epos - 1])))
      --epos;
    tok = tok.substr(b, epos - b);
    std::transform(tok.begin(), tok.end(), tok.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (!tok.empty())
      out.push_back(tok);
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return out;
}

static std::vector<int> parseCsvIntsLocal(const char *envName) {
  std::vector<int> out;
  for (const std::string &tok : parseCsvLowerTokensLocal(envName)) {
    char *end = nullptr;
    long v = std::strtol(tok.c_str(), &end, 10);
    if (end && end != tok.c_str())
      out.push_back(static_cast<int>(v));
  }
  return out;
}

static std::string positFormatNameForConfig(unsigned nbits, unsigned es) {
  return ("p" + std::to_string(nbits) + "e" + std::to_string(es));
}

static std::string positFormatUpperNameForConfig(unsigned nbits, unsigned es) {
  std::string s = positFormatNameForConfig(nbits, es);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return s;
}

static BuildTimeGPConfig resolveBuildTimeGPConfig(unsigned nbits, unsigned es) {
  BuildTimeGPConfig cfg;
  if (nbits != 8 || es > 2)
    return cfg;

  const std::string fmt = positFormatNameForConfig(nbits, es);
  auto addTokens = [&](std::vector<std::string> &dst, const char *name) {
    auto toks = parseCsvLowerTokensLocal(name);
    dst.insert(dst.end(), toks.begin(), toks.end());
  };

  bool enabled = parseEnvBoolLocal("ONNX_MLIR_POSIT_CONST_GP", false) ||
                 parseEnvBoolLocal("POSIT_CONST_GP", false);
  std::vector<std::string> tokens;
  addTokens(tokens, "ONNX_MLIR_POSIT_CONST_GP_FORMATS");
  addTokens(tokens, "POSIT_CONST_GP_FORMATS");
  addTokens(tokens, "POSIT_GP_EXPERIMENTAL_FORMATS");
  addTokens(tokens, "POSIT_GP_FORMATS");
  if (std::find(tokens.begin(), tokens.end(), fmt) != tokens.end())
    enabled = true;
  if (!enabled)
    return cfg;

  const std::string upper = positFormatUpperNameForConfig(nbits, es);
  const std::string rsFmt1 = "ONNX_MLIR_POSIT_CONST_GP_RS_" + upper;
  const std::string scFmt1 = "ONNX_MLIR_POSIT_CONST_GP_SC_" + upper;
  const std::string rsFmt2 = "POSIT_CONST_GP_RS_" + upper;
  const std::string scFmt2 = "POSIT_CONST_GP_SC_" + upper;
  const std::string rsFmt3 = "POSIT_GP_RS_" + upper;
  const std::string scFmt3 = "POSIT_GP_SC_" + upper;

  int rs = parseEnvIntLocal(
      rsFmt1.c_str(),
      parseEnvIntLocal(
          rsFmt2.c_str(),
          parseEnvIntLocal(
              rsFmt3.c_str(),
              parseEnvIntLocal(
                  "ONNX_MLIR_POSIT_CONST_GP_RS_P8",
                  parseEnvIntLocal(
                      "POSIT_CONST_GP_RS_P8",
                      parseEnvIntLocal(
                          "POSIT_GP_RS_P8",
                          parseEnvIntLocal(
                              "ONNX_MLIR_POSIT_CONST_GP_RS",
                              parseEnvIntLocal(
                                  "POSIT_CONST_GP_RS",
                                  parseEnvIntLocal("POSIT_GP_RS", 7)))))))));

  int sc = parseEnvIntLocal(
      scFmt1.c_str(),
      parseEnvIntLocal(
          scFmt2.c_str(),
          parseEnvIntLocal(
              scFmt3.c_str(),
              parseEnvIntLocal(
                  "ONNX_MLIR_POSIT_CONST_GP_SC_P8",
                  parseEnvIntLocal(
                      "POSIT_CONST_GP_SC_P8",
                      parseEnvIntLocal(
                          "POSIT_GP_SC_P8",
                          parseEnvIntLocal(
                              "ONNX_MLIR_POSIT_CONST_GP_SC",
                              parseEnvIntLocal(
                                  "POSIT_CONST_GP_SC",
                                  parseEnvIntLocal("POSIT_GP_SC", 0)))))))));

  cfg.enabled = true;
  cfg.rs = std::min(7, std::max(1, rs));
  cfg.sc = std::min(16, std::max(-16, sc));
  return cfg;
}

static std::vector<BuildTimeGPConfig> resolveBuildTimeGPCandidates(
    unsigned nbits, unsigned es) {
  std::vector<BuildTimeGPConfig> out;
  BuildTimeGPConfig base = resolveBuildTimeGPConfig(nbits, es);
  if (!base.enabled)
    return out;

  const std::string upper = positFormatUpperNameForConfig(nbits, es);
  std::vector<int> rsVals = resolveSweepInts(
      upper,
      "ONNX_MLIR_POSIT_CONST_GP_RS_VALUES",
      "POSIT_CONST_GP_RS_VALUES",
      "POSIT_GP_RS_VALUES",
      "ONNX_MLIR_POSIT_CONST_GP_RS_VALUES_P8",
      "POSIT_CONST_GP_RS_VALUES_P8",
      "POSIT_GP_RS_VALUES_P8",
      "ONNX_MLIR_POSIT_CONST_GP_RS",
      "POSIT_CONST_GP_RS",
      "POSIT_GP_RS",
      base.rs, 1, 7);
  std::vector<int> scVals = resolveSweepInts(
      upper,
      "ONNX_MLIR_POSIT_CONST_GP_SC_VALUES",
      "POSIT_CONST_GP_SC_VALUES",
      "POSIT_GP_SC_VALUES",
      "ONNX_MLIR_POSIT_CONST_GP_SC_VALUES_P8",
      "POSIT_CONST_GP_SC_VALUES_P8",
      "POSIT_GP_SC_VALUES_P8",
      "ONNX_MLIR_POSIT_CONST_GP_SC",
      "POSIT_CONST_GP_SC",
      "POSIT_GP_SC",
      base.sc, -16, 16);

  for (int rs : rsVals) {
    for (int sc : scVals) {
      BuildTimeGPConfig cfg;
      cfg.enabled = true;
      cfg.rs = std::min(7, std::max(1, rs));
      cfg.sc = std::min(16, std::max(-16, sc));
      bool seen = false;
      for (const auto &cur : out) {
        if (cur.enabled == cfg.enabled && cur.rs == cfg.rs && cur.sc == cfg.sc) {
          seen = true;
          break;
        }
      }
      if (!seen)
        out.push_back(cfg);
    }
  }
  if (out.empty())
    out.push_back(base);
  return out;
}

static std::optional<llvm::StringRef> getConvOrGemmPerAxisConstantUseRole(
    OpOperand &use, RankedTensorType type) {
  Operation *owner = use.getOwner();
  unsigned operand = use.getOperandNumber();
  if (llvm::isa<mlir::ONNXConvOp>(owner)) {
    if (operand == 1)
      return llvm::StringRef("conv_weight");
    if (operand == 2 && type.getRank() == 1)
      return llvm::StringRef("conv_bias");
    return std::nullopt;
  }
  if (llvm::isa<mlir::ONNXGemmOp>(owner)) {
    if (operand == 1)
      return llvm::StringRef("gemm_weight");
    if (operand == 2 && type.getRank() == 1)
      return llvm::StringRef("gemm_bias");
    return std::nullopt;
  }
  return std::nullopt;
}

// True if every use of this constant is as a Conv/Gemm BIAS (and there is at
// least one such use). Bias constants must NOT be ALPS-companded: their
// per-tensor dynamic range is extreme (observed up to ~5000x within one bias
// vector on MobileNetV2), so a single ALPS theta cannot cover it — the grid
// search picks theta≈theta_min, which pushes every bias value into posit's
// tiny-magnitude region and crushes it (~95% bias loss → activation mean stays
// negative → ReLU6 zeros everything → accuracy collapse). Plain DIRECT posit
// (float-like, relative precision everywhere) represents each bias value within
// a few % — enough to recenter activations. This mirrors INT8 keeping bias in
// int32 (high precision), not int8.
static bool isBiasOnlyConstant(mlir::ONNXConstantOp op, RankedTensorType type) {
  bool sawBias = false;
  for (OpOperand &use : op->getResult(0).getUses()) {
    std::optional<llvm::StringRef> role =
        getConvOrGemmPerAxisConstantUseRole(use, type);
    if (!role)
      return false;
    if (*role == "conv_bias" || *role == "gemm_bias")
      sawBias = true;
    else
      return false;  // a non-bias weight-like use
  }
  return sawBias;
}

// True if the constant has at least one Conv/Gemm WEIGHT use. Only weights
// benefit from ALPS companding (their per-channel/per-tensor distributions are
// tight and posit precision can be focused via theta). Everything else —
// biases, and especially BIAS CONSTANTS THAT WERE PRE-BROADCAST to the full
// activation shape (e.g. bias [16] folded into a [1,16,12544] constant feeding
// posit.add) — must stay DIRECT. The broadcast bias is not a Conv/Gemm operand
// anymore (it feeds Add), so it is not caught by isBiasOnlyConstant; without
// this gate it would fall into the scalar-ALPS path and get crushed (extreme
// dynamic range → theta≈theta_min → ~95% bias loss → activation collapse).
static bool hasConvGemmWeightUse(mlir::ONNXConstantOp op, RankedTensorType type) {
  for (OpOperand &use : op->getResult(0).getUses()) {
    std::optional<llvm::StringRef> role =
        getConvOrGemmPerAxisConstantUseRole(use, type);
    if (role && (*role == "conv_weight" || *role == "gemm_weight"))
      return true;
  }
  return false;
}

static bool shouldUsePerAxisConstMetadata(mlir::ONNXConstantOp op,
                                          RankedTensorType type,
                                          int64_t &axisOut,
                                          std::string *roleSummaryOut = nullptr) {
  axisOut = 0;
  if (!type || !type.hasStaticShape() || type.getRank() < 1)
    return false;
  if (type.getDimSize(0) <= 1)
    return false;
  bool sawWeightLikeUse = false;
  bool sawNonBiasWeightUse = false;
  SmallVector<llvm::StringRef, 4> roles;
  for (OpOperand &use : op->getResult(0).getUses()) {
    std::optional<llvm::StringRef> role =
        getConvOrGemmPerAxisConstantUseRole(use, type);
    if (!role)
      return false;
    sawWeightLikeUse = true;
    if (*role != "conv_bias" && *role != "gemm_bias")
      sawNonBiasWeightUse = true;
    if (llvm::find(roles, *role) == roles.end())
      roles.push_back(*role);
  }
  // Bias constants must NOT use per-axis (per-channel) metadata: the elemwise
  // Add kernel that consumes the bias only does a per-TENSOR metadata lookup
  // (lookupTensorGPMetadata), not per-channel. If the bias were encoded with
  // per-channel ALPS thetas, the Add would decode every channel with a single
  // (wrong) per-tensor metadata → encode/decode mismatch → bias collapses
  // (~95% loss, observed as ReLU6 activation collapse on MobileNetV2). Force
  // bias to the scalar (per-tensor) ALPS decision path, which the Add decodes
  // correctly and which still falls back to direct posit when ALPS gives no gain.
  if (sawWeightLikeUse && !sawNonBiasWeightUse)
    return false;
  if (roleSummaryOut && sawWeightLikeUse) {
    std::string joined;
    llvm::raw_string_ostream os(joined);
    for (size_t i = 0; i < roles.size(); ++i) {
      if (i)
        os << ",";
      os << roles[i];
    }
    *roleSummaryOut = os.str();
  }
  return sawWeightLikeUse;
}

static double decodeApproxGeneralizedPosit8Build(
    uint8_t raw, int es, int rs, int sc) {
  constexpr int nbits = 8;
  if (raw == 0)
    return 0.0;
  if (raw == 0x80u)
    return std::numeric_limits<double>::quiet_NaN();
  bool neg = (raw & 0x80u) != 0;
  uint8_t ui = raw;
  if (neg)
    ui = static_cast<uint8_t>((~ui) + 1u);
  int pos = nbits - 2;
  bool regBit = ((ui >> pos) & 0x1u) != 0;
  int run = 0;
  int maxRun = std::max(1, rs);
  while (pos >= 0 && (((ui >> pos) & 0x1u) != 0) == regBit && run < maxRun) {
    ++run;
    --pos;
  }
  bool stoppedByCap =
      (run >= maxRun && pos >= 0 && ((((ui >> pos) & 0x1u) != 0) == regBit));
  int k = regBit ? (run - 1) : (-run);
  if (!stoppedByCap && pos >= 0)
    --pos;
  int exp = 0;
  for (int i = 0; i < es && pos >= 0; ++i, --pos)
    exp = (exp << 1) | static_cast<int>((ui >> pos) & 0x1u);
  double frac = 0.0;
  double bit = 0.5;
  while (pos >= 0) {
    if ((ui >> pos) & 0x1u)
      frac += bit;
    bit *= 0.5;
    --pos;
  }
  double scale = static_cast<double>((1 << es) * k + exp + sc);
  double mag = std::ldexp(1.0 + frac, static_cast<int>(scale));
  return neg ? -mag : mag;
}

struct GP8BuildTimeTable {
  int es = 0;
  int rs = 7;
  int sc = 0;
  std::array<double, 256> decode{};
  std::vector<std::pair<double, uint8_t>> sorted;
};

template <int ES>
static const GP8BuildTimeTable &getGP8BuildTimeTableCustom(int rs, int sc) {
  static std::map<std::pair<int, int>, GP8BuildTimeTable> cache;
  static std::mutex cacheMutex;
  rs = std::min(7, std::max(1, rs));
  sc = std::min(16, std::max(-16, sc));
  auto key = std::make_pair(rs, sc);
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second;
  }
  GP8BuildTimeTable t;
  t.es = ES;
  t.rs = rs;
  t.sc = sc;
  for (int i = 0; i < 256; ++i) {
    uint8_t raw = static_cast<uint8_t>(i);
    t.decode[i] = decodeApproxGeneralizedPosit8Build(raw, ES, rs, sc);
    if (raw == 0x80u || !std::isfinite(t.decode[i]))
      continue;
    t.sorted.push_back({t.decode[i], raw});
  }
  std::sort(t.sorted.begin(), t.sorted.end(),
            [](const auto &a, const auto &b) {
              if (a.first != b.first)
                return a.first < b.first;
              return a.second < b.second;
            });
  std::lock_guard<std::mutex> lock(cacheMutex);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  auto [inserted, _] = cache.emplace(key, std::move(t));
  return inserted->second;
}

// Round x to the nearest value in a PREBUILT GP8 table (binary search only, no
// table fetch/lock). The const-ALPS weight search fetches the table once per
// (rs,sc) candidate and calls this per weight; the old code re-entered the
// mutex-locked table cache on EVERY element, so with POSIT_CONST_ALPS_JOBS
// threads doing ~1e11 rounds they all contended a single lock — the real reason
// big models (ResNet18) took hours to build.
static inline bool roundInGP8Table(double x, const GP8BuildTimeTable &table,
                                   double &roundedOut,
                                   uint64_t *rawOut = nullptr) {
  if (!std::isfinite(x)) {
    roundedOut = std::numeric_limits<double>::quiet_NaN();
    if (rawOut)
      *rawOut = 0x80u;
    return true;
  }
  if (table.sorted.empty()) {
    roundedOut = 0.0;
    if (rawOut)
      *rawOut = 0u;
    return true;
  }
  auto it = std::lower_bound(
      table.sorted.begin(), table.sorted.end(), x,
      [](const std::pair<double, uint8_t> &a, double v) { return a.first < v; });
  uint8_t raw = 0;
  if (it == table.sorted.begin()) {
    raw = it->second;
  } else if (it == table.sorted.end()) {
    raw = table.sorted.back().second;
  } else {
    auto prev = it - 1;
    raw = (std::fabs(prev->first - x) <= std::fabs(it->first - x)) ? prev->second
                                                                   : it->second;
  }
  roundedOut = table.decode[raw];
  if (rawOut)
    *rawOut = raw;
  return true;
}

// Fetch the (cached) GP8 build-time table for a runtime es; returns nullptr for
// unsupported es. Still locks the cache once — call it ONCE per (es,rs,sc), not
// per element.
static const GP8BuildTimeTable *getGP8BuildTimeTablePtr(unsigned es, int rs,
                                                        int sc) {
  switch (es) {
  case 0:
    return &getGP8BuildTimeTableCustom<0>(rs, sc);
  case 1:
    return &getGP8BuildTimeTableCustom<1>(rs, sc);
  case 2:
    return &getGP8BuildTimeTableCustom<2>(rs, sc);
  default:
    return nullptr;
  }
}

static bool generalizedP8RoundToDoubleDispatch(double x, unsigned es, int rs,
                                               int sc, double &roundedOut,
                                               uint64_t *rawOut = nullptr) {
  const GP8BuildTimeTable *table = getGP8BuildTimeTablePtr(es, rs, sc);
  if (!table)
    return false;
  return roundInGP8Table(x, *table, roundedOut, rawOut);
}

static double percentileAbsInPlace(SmallVectorImpl<double> &vals, double q) {
  if (vals.empty())
    return 0.0;
  q = std::clamp(q, 0.0, 1.0);
  for (double &v : vals)
    v = std::fabs(v);
  size_t idx = static_cast<size_t>(
      std::llround(q * static_cast<double>(vals.size() - 1)));
  std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
  return vals[idx];
}

// Weight ALPS scoring metric. NSR (noise-to-signal ratio) = sum((a-b)^2)/sum(a^2)
// = 1/SQNR. Lower is better (keeps the existing minimize + minGain logic; maximize
// SQNR == minimize NSR). Replaces the previous MAE = mean(|a-b|): NSR weights by
// signal power (a^2) so the few large discriminative weights dominate the theta
// choice instead of being averaged away by many small weights. Matches the runtime
// metric (posit_runtime.cpp::runtimeOutputAlpsScoreForMeta).
static double noiseToSignalRatio(ArrayRef<double> a, ArrayRef<double> b) {
  if (a.size() != b.size() || a.empty())
    return std::numeric_limits<double>::infinity();
  long double errPow = 0.0L, sigPow = 0.0L;
  for (size_t i = 0; i < a.size(); ++i) {
    long double d = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
    errPow += d * d;
    sigPow += static_cast<long double>(a[i]) * static_cast<long double>(a[i]);
  }
  if (sigPow <= 0.0L)
    return static_cast<double>(errPow);
  return static_cast<double>(errPow / sigPow);
}

static std::optional<BuildTimeConstCompandDecision>
buildTimeWeightAlpsDecision(ArrayRef<double> vals, unsigned nbits,
                            unsigned es, unsigned searchJobs = 0) {
  if (vals.empty()) {
    if (isPositConstDebugEnabled())
      llvm::errs() << "[posit-const] build-time ALPS skipped: empty tensor\n";
    return std::nullopt;
  }
  const bool envAlps =
      parseEnvBoolLocal("ONNX_MLIR_POSIT_CONST_ALPS", false) ||
      parseEnvBoolLocal("POSIT_CONST_ALPS", false);
  if (!envAlps) {
    if (isPositConstDebugEnabled())
      llvm::errs() << "[posit-const] build-time ALPS skipped: env disabled\n";
    return std::nullopt;
  }

  const double thetaMin = parseEnvDoubleLocal(
      "ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN",
      parseEnvDoubleLocal("POSIT_CONST_ALPS_THETA_MIN", 0.25));
  const double thetaMax = parseEnvDoubleLocal(
      "ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX",
      parseEnvDoubleLocal("POSIT_CONST_ALPS_THETA_MAX", 4.0));
  const int thetaSteps = std::max(2, parseEnvIntLocal(
      "ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS",
      parseEnvIntLocal("POSIT_CONST_ALPS_THETA_STEPS", 9)));
  const double gammaTarget = parseEnvDoubleLocal(
      "ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET",
      parseEnvDoubleLocal("POSIT_CONST_ALPS_GAMMA_TARGET", 1.0));
  const double gammaPct = parseEnvDoubleLocal(
      "ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE",
      parseEnvDoubleLocal("POSIT_CONST_ALPS_GAMMA_PERCENTILE", 0.95));
  const double minGain = parseEnvDoubleLocal(
      "ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN",
      parseEnvDoubleLocal("POSIT_CONST_ALPS_MIN_GAIN", 0.0));
  const int maxSamples = parseEnvIntLocal(
      "ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES",
      parseEnvIntLocal("POSIT_CONST_ALPS_MAX_SAMPLES", 4096));
  const BuildTimeGPConfig gpCfg = resolveBuildTimeGPConfig(nbits, es);
  const std::vector<BuildTimeGPConfig> gpCandidates =
      resolveBuildTimeGPCandidates(nbits, es);

  SmallVector<double, 8> sampledVals;
  ArrayRef<double> scoreVals = vals;
  if (maxSamples > 0 && static_cast<int64_t>(vals.size()) > maxSamples) {
    sampledVals.reserve(maxSamples);
    const size_t n = vals.size();
    const size_t m = static_cast<size_t>(maxSamples);
    for (size_t i = 0; i < m; ++i) {
      size_t idx = (m <= 1) ? 0 : ((n - 1) * i) / (m - 1);
      sampledVals.push_back(vals[idx]);
    }
    scoreVals = sampledVals;
  }

  if (isPositConstDebugEnabled())
    llvm::errs() << "[posit-const] build-time ALPS start nbits=" << nbits
                 << " es=" << es << " elems=" << vals.size()
                 << " score_elems=" << scoreVals.size()
                 << " gp_enabled=" << (gpCfg.enabled ? 1 : 0)
                 << " gp_rs=" << gpCfg.rs
                 << " gp_sc=" << gpCfg.sc
                 << " gp_candidate_count=" << gpCandidates.size()
                 << " theta_min=" << thetaMin
                 << " theta_max=" << thetaMax
                 << " theta_steps=" << thetaSteps
                 << " search_jobs="
                 << (searchJobs ? searchJobs : positConstAlpsJobs())
                 << " gamma_target=" << gammaTarget
                 << " gamma_pct=" << gammaPct
                 << " min_gain=" << minGain << "\n";

  if (!(thetaMin > 0.0) || !(thetaMax >= thetaMin) || !(gammaTarget > 0.0)) {
    if (isPositConstDebugEnabled())
      llvm::errs() << "[posit-const] build-time ALPS skipped: invalid params\n";
    return std::nullopt;
  }

  BuildTimeConstCompandDecision best;
  best.rawBits.reserve(vals.size());
  auto maybeUpdateBestDirect = [&](const BuildTimeGPConfig &cfg,
                                   double directScore) {
    if (directScore < best.chosenScore) {
      best.useAlps = false;
      best.compandMode = 0;
      best.theta = 1.0;
      best.gamma = 0.0;
      best.useGp = cfg.enabled;
      best.gpRs = cfg.rs;
      best.gpSc = cfg.sc;
      best.chosenScore = directScore;
    }
  };

  SmallVector<BuildTimeGPConfig, 8> configs;
  configs.push_back(BuildTimeGPConfig{false, 7, 0});
  for (const auto &cfg : gpCandidates)
    configs.push_back(cfg);

  SmallVector<double, 8> absZ;
  SmallVector<double, 8> decoded;
  absZ.reserve(scoreVals.size());
  decoded.reserve(scoreVals.size());

  const double logThetaMin = std::log2(thetaMin);
  const double logThetaMax = std::log2(thetaMax);
  constexpr double gammaScales[] = {0.5, 1.0, 2.0};

  best.directScore = std::numeric_limits<double>::infinity();
  best.chosenScore = std::numeric_limits<double>::infinity();

  struct DirectCandidateResult {
    BuildTimeGPConfig cfg;
    double directScore = std::numeric_limits<double>::infinity();
  };
  std::vector<DirectCandidateResult> directCandidates;
  directCandidates.reserve(configs.size());

  for (const auto &cfg : configs) {
    SmallVector<double, 8> directDecoded;
    directDecoded.reserve(scoreVals.size());
    bool directBad = false;
    // Table fetched once per config (not per weight) — same lock-free round.
    const GP8BuildTimeTable *gpTabD =
        cfg.enabled ? getGP8BuildTimeTablePtr(es, cfg.rs, cfg.sc) : nullptr;
    for (double v : scoreVals) {
      double rounded = v;
      uint64_t raw = 0;
      bool ok = cfg.enabled
                    ? (gpTabD ? roundInGP8Table(v, *gpTabD, rounded, &raw)
                              : false)
                    : universalRoundToDoubleDispatch(
                          v, nbits, es, rounded, &raw);
      if (!ok) {
        directBad = true;
        break;
      }
      directDecoded.push_back(rounded);
    }
    if (directBad) {
      if (isPositConstDebugEnabled())
        llvm::errs() << "[posit-const] build-time ALPS skipped config: direct "
                        "round unsupported nbits="
                     << nbits << " es=" << es
                     << " gp_enabled=" << (cfg.enabled ? 1 : 0)
                     << " gp_rs=" << cfg.rs << " gp_sc=" << cfg.sc << "\n";
      continue;
    }
    double directScore = noiseToSignalRatio(scoreVals, directDecoded);
    if (!cfg.enabled)
      best.directScore = directScore;
    maybeUpdateBestDirect(cfg, directScore);
    directCandidates.push_back({cfg, directScore});
  }

  struct LocalBestCandidate {
    bool valid = false;
    double score = std::numeric_limits<double>::infinity();
    double theta = 1.0;
    double gamma = 0.0;
    bool useGp = false;
    int rs = 7;
    int sc = 0;
  };

  auto evaluateTask = [&](const DirectCandidateResult &cfgResult, int t,
                          LocalBestCandidate &localBest) {
    const auto &cfg = cfgResult.cfg;
    const double directScore = cfgResult.directScore;
    double frac = (thetaSteps <= 1) ? 0.0
                                    : static_cast<double>(t) /
                                          static_cast<double>(thetaSteps - 1);
    double theta =
        std::pow(2.0, logThetaMin + (logThetaMax - logThetaMin) * frac);
    SmallVector<double, 8> localAbsZ;
    localAbsZ.reserve(scoreVals.size());
    for (double v : scoreVals) {
      double z = std::asinh(theta * v);
      localAbsZ.push_back(std::isfinite(z) ? std::fabs(z) : 0.0);
    }
    double gammaBase = percentileAbsInPlace(localAbsZ, gammaPct) / gammaTarget;
    if (!(gammaBase > 0.0) || !std::isfinite(gammaBase))
      return;

    // Fetch the GP8 round table ONCE per candidate (rs,sc are fixed for this
    // task) so the per-weight round below is a lock-free binary search instead
    // of re-locking the table cache on every element.
    const GP8BuildTimeTable *gpTab =
        cfg.enabled ? getGP8BuildTimeTablePtr(es, cfg.rs, cfg.sc) : nullptr;

    SmallVector<double, 8> localDecoded;
    for (double gammaScale : gammaScales) {
      double gamma = gammaBase * gammaScale;
      if (!(gamma > 0.0) || !std::isfinite(gamma))
        continue;
      localDecoded.clear();
      localDecoded.reserve(scoreVals.size());
      bool bad = false;
      for (double v : scoreVals) {
        double y = std::asinh(theta * v) / gamma;
        double yq = y;
        bool ok = cfg.enabled
                      ? (gpTab ? roundInGP8Table(y, *gpTab, yq, nullptr) : false)
                      : universalRoundToDoubleDispatch(
                            y, nbits, es, yq, nullptr);
        if (!ok) {
          bad = true;
          break;
        }
        double xdq = std::sinh(gamma * yq) / theta;
        if (!std::isfinite(xdq)) {
          bad = true;
          break;
        }
        localDecoded.push_back(xdq);
      }
      if (bad)
        continue;
      double score = noiseToSignalRatio(scoreVals, localDecoded);
      if (score + minGain < directScore && score < localBest.score) {
        localBest.valid = true;
        localBest.score = score;
        localBest.theta = theta;
        localBest.gamma = gamma;
        localBest.useGp = cfg.enabled;
        localBest.rs = cfg.rs;
        localBest.sc = cfg.sc;
      }
    }
  };

  const unsigned effectiveSearchJobs =
      std::max(1u, searchJobs ? searchJobs : positConstAlpsJobs());
  const size_t taskCount =
      directCandidates.size() * static_cast<size_t>(thetaSteps);
  if (effectiveSearchJobs <= 1 || taskCount <= 1) {
    LocalBestCandidate localBest;
    for (const auto &cfgResult : directCandidates)
      for (int t = 0; t < thetaSteps; ++t)
        evaluateTask(cfgResult, t, localBest);
    if (localBest.valid && localBest.score < best.chosenScore) {
      best.useAlps = true;
      best.compandMode = 1;
      best.theta = localBest.theta;
      best.gamma = localBest.gamma;
      best.useGp = localBest.useGp;
      best.gpRs = localBest.rs;
      best.gpSc = localBest.sc;
      best.chosenScore = localBest.score;
    }
  } else {
    const unsigned workerCount = std::min<unsigned>(
        effectiveSearchJobs, static_cast<unsigned>(std::max<size_t>(1, taskCount)));
    std::atomic<size_t> nextTask{0};
    std::vector<LocalBestCandidate> workerBest(workerCount);
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (unsigned worker = 0; worker < workerCount; ++worker) {
      workers.emplace_back([&, worker]() {
        while (true) {
          size_t task = nextTask.fetch_add(1, std::memory_order_relaxed);
          if (task >= taskCount)
            break;
          size_t cfgIdx = task / static_cast<size_t>(thetaSteps);
          int thetaIdx = static_cast<int>(task % static_cast<size_t>(thetaSteps));
          evaluateTask(directCandidates[cfgIdx], thetaIdx, workerBest[worker]);
        }
      });
    }
    for (auto &worker : workers)
      worker.join();
    for (const auto &localBest : workerBest) {
      if (!localBest.valid || !(localBest.score < best.chosenScore))
        continue;
      best.useAlps = true;
      best.compandMode = 1;
      best.theta = localBest.theta;
      best.gamma = localBest.gamma;
      best.useGp = localBest.useGp;
      best.gpRs = localBest.rs;
      best.gpSc = localBest.sc;
      best.chosenScore = localBest.score;
    }
  }

  if (!std::isfinite(best.chosenScore)) {
    if (isPositConstDebugEnabled())
      llvm::errs() << "[posit-const] build-time ALPS skipped: no viable direct/ALPS candidate\n";
    return std::nullopt;
  }

  best.rawBits.clear();
  best.rawBits.reserve(vals.size());
  if (best.useAlps) {
    for (double v : vals) {
      double y = std::asinh(best.theta * v) / best.gamma;
      double yq = y;
      uint64_t raw = 0;
      bool ok = best.useGp
                    ? generalizedP8RoundToDoubleDispatch(
                          y, es, best.gpRs, best.gpSc, yq, &raw)
                    : universalRoundToDoubleDispatch(y, nbits, es, yq, &raw);
      if (!ok)
        return std::nullopt;
      best.rawBits.push_back(raw);
    }
  } else {
    for (double v : vals) {
      double rounded = v;
      uint64_t raw = 0;
      bool ok = best.useGp
                    ? generalizedP8RoundToDoubleDispatch(
                          v, es, best.gpRs, best.gpSc, rounded, &raw)
                    : universalRoundToDoubleDispatch(v, nbits, es, rounded, &raw);
      if (!ok)
        return std::nullopt;
      best.rawBits.push_back(raw);
    }
  }
  return best;
}

static std::optional<BuildTimePerAxisConstDecision>
buildTimeWeightAlpsDecisionPerAxis(ArrayRef<double> vals,
                                   RankedTensorType origOutType,
                                   unsigned nbits, unsigned es, int64_t axis) {
  if (!origOutType || !origOutType.hasStaticShape() || axis != 0)
    return std::nullopt;
  if (origOutType.getRank() < 1 || origOutType.getDimSize(0) <= 1)
    return std::nullopt;
  int64_t channels = origOutType.getDimSize(0);
  int64_t sliceElems = origOutType.getNumElements() / channels;
  if (channels <= 0 || sliceElems <= 0)
    return std::nullopt;
  if (static_cast<int64_t>(vals.size()) != origOutType.getNumElements())
    return std::nullopt;

  std::vector<std::optional<BuildTimeConstCompandDecision>> decisions(
      static_cast<size_t>(channels));
  unsigned jobs = std::min<unsigned>(
      positConstAlpsJobs(), static_cast<unsigned>(std::max<int64_t>(1, channels)));
  if (isPositConstDebugEnabled())
    llvm::errs() << "[posit-const] per-axis search channels=" << channels
                 << " slice_elems=" << sliceElems
                 << " jobs=" << jobs
                 << " nbits=" << nbits
                 << " es=" << es << "\n";

  auto runOneChannel = [&](int64_t ch) {
    auto begin = vals.begin() + ch * sliceElems;
    decisions[static_cast<size_t>(ch)] = buildTimeWeightAlpsDecision(
        ArrayRef<double>(begin, static_cast<size_t>(sliceElems)), nbits, es,
        /*searchJobs=*/1);
  };

  if (jobs <= 1 || channels <= 1) {
    for (int64_t ch = 0; ch < channels; ++ch)
      runOneChannel(ch);
  } else {
    std::atomic<int64_t> nextChannel{0};
    std::vector<std::thread> workers;
    workers.reserve(jobs);
    for (unsigned worker = 0; worker < jobs; ++worker) {
      workers.emplace_back([&]() {
        while (true) {
          int64_t ch = nextChannel.fetch_add(1, std::memory_order_relaxed);
          if (ch >= channels)
            break;
          runOneChannel(ch);
        }
      });
    }
    for (auto &worker : workers)
      worker.join();
  }

  BuildTimePerAxisConstDecision out;
  out.axis = axis;
  out.compandModes.reserve(static_cast<size_t>(channels));
  out.thetas.reserve(static_cast<size_t>(channels));
  out.gammas.reserve(static_cast<size_t>(channels));
  out.gpEnabled.reserve(static_cast<size_t>(channels));
  out.gpRs.reserve(static_cast<size_t>(channels));
  out.gpSc.reserve(static_cast<size_t>(channels));
  out.rawBits.reserve(vals.size());

  for (int64_t ch = 0; ch < channels; ++ch) {
    const auto &decision = decisions[static_cast<size_t>(ch)];
    if (!decision)
      return std::nullopt;
    out.compandModes.push_back(decision->compandMode);
    out.thetas.push_back(decision->theta);
    out.gammas.push_back(decision->gamma);
    out.gpEnabled.push_back(decision->useGp ? 1 : 0);
    out.gpRs.push_back(decision->gpRs);
    out.gpSc.push_back(decision->gpSc);
    out.rawBits.append(decision->rawBits.begin(), decision->rawBits.end());
  }
  return out;
}

static Value createCompactPositConstantIfEnabled(ConversionPatternRewriter &rewriter,
                                                 Location loc,
                                                 mlir::ONNXConstantOp srcOp,
                                                 RankedTensorType origOutType,
                                                 Type positTensorType,
                                                 ElementsAttr elements,
                                                 unsigned nbits,
                                                 unsigned es,
                                                 unsigned storageBits) {
  if (!parseEnvBoolLocal("ONNX_MLIR_POSIT_COMPACT_CONSTANTS", false) &&
      !parseEnvBoolLocal("POSIT_COMPACT_CONSTANTS", false))
    return Value();
  if (!parseEnvBoolLocal("ONNX_MLIR_POSIT_FORCE_NQDQ", false) &&
      !parseEnvBoolLocal("POSIT_FORCE_NQDQ_POSIT", false))
    return Value();
  // Only compact formats whose storage is currently modeled as i8/i16/i32 raw
  // bits in posit.constant. Keep qdq-posit untouched; this path is for
  // compile-time nqdq/f32 -> posit constant materialization.
  if (nbits == 0 || nbits > 32 || storageBits > 32)
    return Value();

  SmallVector<double, 8> vals;
  if (failed(collectFPValuesAsDouble(elements, vals))) {
    if (isPositConstDebugEnabled())
      llvm::errs() << "[posit-const] collectFPValuesAsDouble failed nbits="
                   << nbits << " es=" << es << " type=" << origOutType << "\n";
    return Value();
  }

  auto intElemTy = rewriter.getIntegerType(storageBits);
  auto rawTensorTy = RankedTensorType::get(origOutType.getShape(), intElemTy);
  SmallVector<uint64_t, 8> rawBits;
  int compandMode = 0;
  double compandTheta = 1.0;
  double compandGamma = 0.0;
  bool gpEnabled = false;
  int gpRs = 7;
  int gpSc = 0;
  std::optional<BuildTimePerAxisConstDecision> perAxisDecision;

  auto makeDenseI64VectorAttr = [&](ArrayRef<int64_t> vals) -> DenseElementsAttr {
    auto ty = RankedTensorType::get(
        {static_cast<int64_t>(vals.size())}, rewriter.getI64Type());
    SmallVector<APInt, 8> apVals;
    apVals.reserve(vals.size());
    for (int64_t v : vals)
      apVals.push_back(APInt(64, static_cast<uint64_t>(v), true));
    return DenseElementsAttr::get(ty, apVals);
  };
  auto makeDenseF64VectorAttr = [&](ArrayRef<double> vals) -> DenseElementsAttr {
    auto ty = RankedTensorType::get(
        {static_cast<int64_t>(vals.size())}, rewriter.getF64Type());
    SmallVector<Attribute, 8> attrs;
    attrs.reserve(vals.size());
    for (double v : vals)
      attrs.push_back(FloatAttr::get(rewriter.getF64Type(), v));
    return DenseElementsAttr::get(ty, attrs);
  };

  int64_t perAxis = -1;
  std::string perAxisRoleSummary;
  if (shouldUsePerAxisConstMetadata(srcOp, origOutType, perAxis,
                                    &perAxisRoleSummary)) {
    perAxisDecision =
        buildTimeWeightAlpsDecisionPerAxis(vals, origOutType, nbits, es, perAxis);
    if (perAxisDecision) {
      rawBits.assign(perAxisDecision->rawBits.begin(), perAxisDecision->rawBits.end());
      if (isPositConstDebugEnabled())
        llvm::errs() << "[posit-const] per-axis metadata selected role="
                     << (perAxisRoleSummary.empty() ? "unknown"
                                                   : perAxisRoleSummary)
                     << " axis=" << perAxis
                     << " channels=" << origOutType.getDimSize(0)
                     << " nbits=" << nbits << " es=" << es << "\n";
    }
  }

  // ALPS companding is for Conv/Gemm WEIGHTS only. Everything else (biases, and
  // pre-broadcast bias constants feeding posit.add) stays DIRECT — a single ALPS
  // theta cannot cover a bias's extreme dynamic range and crushes it (see
  // hasConvGemmWeightUse / isBiasOnlyConstant). Direct posit gives float-like
  // relative precision, preserving the bias well enough to recenter activations.
  const bool isWeight = hasConvGemmWeightUse(srcOp, origOutType);
  std::optional<BuildTimeConstCompandDecision> scalarDecision;
  if (!perAxisDecision && isWeight)
    scalarDecision = buildTimeWeightAlpsDecision(vals, nbits, es);

  if (scalarDecision) {
    const auto &decision = *scalarDecision;
    rawBits.assign(decision.rawBits.begin(), decision.rawBits.end());
    compandMode = decision.compandMode;
    compandTheta = decision.theta;
    compandGamma = decision.gamma;
    gpEnabled = decision.useGp;
    gpRs = decision.gpRs;
    gpSc = decision.gpSc;
    if (isPositConstDebugEnabled())
      llvm::errs() << "[posit-const] build-time ALPS "
                   << (decision.useAlps ? "selected" : "rejected")
                   << " nbits=" << nbits << " es=" << es
                   << " direct_score=" << decision.directScore
                   << " chosen_score=" << decision.chosenScore
                    << " theta=" << compandTheta
                   << " gamma=" << compandGamma
                   << " gp_enabled=" << (gpEnabled ? 1 : 0)
                   << " gp_rs=" << gpRs
                   << " gp_sc=" << gpSc << "\n";
  } else if (!perAxisDecision) {
    rawBits.reserve(vals.size());
    BuildTimeGPConfig gpCfg = resolveBuildTimeGPConfig(nbits, es);
    gpEnabled = gpCfg.enabled;
    gpRs = gpCfg.rs;
    gpSc = gpCfg.sc;
    for (double v : vals) {
      uint64_t raw = 0;
      double rounded = v;
      bool ok = gpEnabled
                    ? generalizedP8RoundToDoubleDispatch(
                          v, es, gpRs, gpSc, rounded, &raw)
                    : universalEncodeRawDispatch(v, nbits, es, raw);
      if (!ok) {
        if (isPositConstDebugEnabled())
          llvm::errs() << "[posit-const] universalEncodeRawDispatch failed nbits="
                       << nbits << " es=" << es << " value=" << v << "\n";
        return Value();
      }
      rawBits.push_back(raw);
    }
  }

  SmallVector<Attribute, 8> rawAttrs;
  rawAttrs.reserve(rawBits.size());
  for (uint64_t raw : rawBits)
    rawAttrs.push_back(IntegerAttr::get(intElemTy, APInt(storageBits, raw)));
  DenseElementsAttr rawElements = DenseElementsAttr::get(rawTensorTy, rawAttrs);
  if (isPositConstDebugEnabled())
    llvm::errs() << "[posit-const] compact constant ok nbits=" << nbits
                 << " es=" << es << " storageBits=" << storageBits
                 << " elems=" << vals.size() << " tensor=" << origOutType
                 << "\n";
  auto cst = rewriter.create<mlir::posit::ConstantOp>(loc, positTensorType, rawElements);
  if (perAxisDecision) {
    cst->setAttr("constmeta_axis", rewriter.getI64IntegerAttr(perAxisDecision->axis));
    cst->setAttr("compand_mode_axis0",
                 makeDenseI64VectorAttr(perAxisDecision->compandModes));
    cst->setAttr("compand_theta_axis0",
                 makeDenseF64VectorAttr(perAxisDecision->thetas));
    cst->setAttr("compand_gamma_axis0",
                 makeDenseF64VectorAttr(perAxisDecision->gammas));
    cst->setAttr("gp_enabled_axis0",
                 makeDenseI64VectorAttr(perAxisDecision->gpEnabled));
    cst->setAttr("gp_rs_axis0",
                 makeDenseI64VectorAttr(perAxisDecision->gpRs));
    cst->setAttr("gp_sc_axis0",
                 makeDenseI64VectorAttr(perAxisDecision->gpSc));
  } else if (compandMode != 0) {
    cst->setAttr("compand_mode", rewriter.getI64IntegerAttr(compandMode));
    cst->setAttr("compand_theta", rewriter.getF64FloatAttr(compandTheta));
    cst->setAttr("compand_gamma", rewriter.getF64FloatAttr(compandGamma));
  }
  if (!perAxisDecision && gpEnabled) {
    cst->setAttr("gp_enabled", rewriter.getI64IntegerAttr(1));
    cst->setAttr("gp_rs", rewriter.getI64IntegerAttr(gpRs));
    cst->setAttr("gp_sc", rewriter.getI64IntegerAttr(gpSc));
  }
  return cst.getResult();
}

static bool getDenseIntegerTensorFromValue(Value v, RankedTensorType &rtt,
                                           SmallVectorImpl<int64_t> &vals);

static bool getDenseFloatTensorFromValue(Value v, RankedTensorType &rtt,
                                         SmallVectorImpl<double> &vals) {
  ElementsAttr ea = getElementsAttrFromValue(v);
  auto ty = ea ? llvm::dyn_cast<RankedTensorType>(ea.getType()) : RankedTensorType();
  if (!ty)
    return false;
  if (!llvm::isa<FloatType>(ty.getElementType()))
    return false;

  vals.clear();
  if (failed(collectFPValuesAsDouble(ea, vals)))
    return false;
  if (static_cast<int64_t>(vals.size()) != ty.getNumElements())
    return false;
  rtt = ty;
  return true;
}


static bool getDenseFloatTensorFromValueRecursive(Value v, RankedTensorType &rtt,
                                                  SmallVectorImpl<double> &vals,
                                                  int depth = 0) {
  if (getDenseFloatTensorFromValue(v, rtt, vals))
    return true;
  if (depth > 8)
    return false;

  Value base = stripUnrealizedCast(v);

  // QDQ bias scales are often materialized as ONNXMul of two constant scales
  // (activation scale * weight scale). The plain dense extractor cannot see
  // through that op, which previously left int32 bias DequantizeLinear illegal
  // in --strict-qdq-mode. Fold simple constant Mul here so DequantizeLinear
  // can still be legalized to f32.
  if (auto mul = base.getDefiningOp<mlir::ONNXMulOp>()) {
    RankedTensorType lhsTy, rhsTy;
    SmallVector<double, 16> lhsVals, rhsVals;
    if (!getDenseFloatTensorFromValueRecursive(mul.getA(), lhsTy, lhsVals, depth + 1) ||
        !getDenseFloatTensorFromValueRecursive(mul.getB(), rhsTy, rhsVals, depth + 1) ||
        lhsVals.empty() || rhsVals.empty())
      return false;

    int64_t outCount = std::max<int64_t>(lhsVals.size(), rhsVals.size());
    if ((lhsVals.size() != 1 && static_cast<int64_t>(lhsVals.size()) != outCount) ||
        (rhsVals.size() != 1 && static_cast<int64_t>(rhsVals.size()) != outCount))
      return false;

    vals.clear();
    vals.reserve(static_cast<size_t>(outCount));
    for (int64_t i = 0; i < outCount; ++i) {
      double a = lhsVals[(lhsVals.size() == 1) ? 0 : static_cast<size_t>(i)];
      double b = rhsVals[(rhsVals.size() == 1) ? 0 : static_cast<size_t>(i)];
      vals.push_back(a * b);
    }

    if (auto outRtt = llvm::dyn_cast<RankedTensorType>(mul.getResult().getType())) {
      rtt = outRtt;
    } else if (static_cast<int64_t>(vals.size()) == lhsTy.getNumElements()) {
      rtt = lhsTy;
    } else if (static_cast<int64_t>(vals.size()) == rhsTy.getNumElements()) {
      rtt = rhsTy;
    } else {
      rtt = RankedTensorType::get({static_cast<int64_t>(vals.size())}, lhsTy.getElementType());
    }
    return true;
  }

  return false;
}

static bool getScalarFloatFromValue(Value v, double &out) {
  ElementsAttr ea = getElementsAttrFromValue(v);
  if (!ea || ea.getNumElements() != 1)
    return false;

  SmallVector<double, 1> fpVals;
  if (succeeded(collectFPValuesAsDouble(ea, fpVals)) && fpVals.size() == 1) {
    out = fpVals[0];
    return true;
  }

  if (auto fp = llvm::dyn_cast<DenseFPElementsAttr>(ea)) {
    out = (*fp.getValues<APFloat>().begin()).convertToDouble();
    return true;
  }
  if (auto ints = llvm::dyn_cast<DenseIntElementsAttr>(ea)) {
    APInt ap = *ints.getValues<APInt>().begin();
    auto intTy = llvm::dyn_cast<IntegerType>(ints.getElementType());
    out = (intTy && intTy.isUnsigned()) ? static_cast<double>(ap.getZExtValue())
                                        : static_cast<double>(ap.getSExtValue());
    return true;
  }
  return false;
}

static bool getScalarIntFromValue(Value v, int64_t &out) {
  ElementsAttr ea = getElementsAttrFromValue(v);
  if (!ea || ea.getNumElements() != 1)
    return false;
  if (auto ints = llvm::dyn_cast<DenseIntElementsAttr>(ea)) {
    APInt ap = *ints.getValues<APInt>().begin();
    auto intTy = llvm::dyn_cast<IntegerType>(ints.getElementType());
    out = (intTy && intTy.isUnsigned()) ? static_cast<int64_t>(ap.getZExtValue())
                                        : static_cast<int64_t>(ap.getSExtValue());
    return true;
  }
  RankedTensorType rtt;
  SmallVector<int64_t, 1> vals;
  if (getDenseIntegerTensorFromValue(v, rtt, vals) && vals.size() == 1) {
    out = vals.front();
    return true;
  }
  return false;
}

static bool getDenseIntegerTensorFromValue(Value v, RankedTensorType &rtt,
                                           SmallVectorImpl<int64_t> &vals) {
  ElementsAttr ea = getElementsAttrFromValue(v);
  auto ty = ea ? llvm::dyn_cast<RankedTensorType>(ea.getType()) : RankedTensorType();
  if (!ty)
    return false;

  auto intTy = llvm::dyn_cast<IntegerType>(ty.getElementType());
  if (!intTy)
    return false;
  rtt = ty;
  vals.clear();

  if (auto ints = llvm::dyn_cast_or_null<DenseIntElementsAttr>(ea)) {
    const bool isUnsigned = intTy.isUnsigned();
    vals.reserve(static_cast<size_t>(ints.getNumElements()));
    for (APInt ap : ints.getValues<APInt>())
      vals.push_back(isUnsigned ? static_cast<int64_t>(ap.getZExtValue())
                                : static_cast<int64_t>(ap.getSExtValue()));
    return true;
  }

  const bool isUnsigned = intTy.isUnsigned();
  auto pushValsFromResource = [&](auto dummy) -> bool {
    using ElemTy = decltype(dummy);
    auto res = llvm::dyn_cast_or_null<mlir::detail::DenseResourceElementsAttrBase<ElemTy>>(ea);
    if (!res)
      return false;
    auto arr = res.tryGetAsArrayRef();
    if (!arr)
      return false;
    vals.reserve(arr->size());
    for (ElemTy x : *arr)
      vals.push_back(static_cast<int64_t>(x));
    return true;
  };

  switch (intTy.getWidth()) {
  case 8:
    return isUnsigned ? pushValsFromResource(uint8_t{}) : pushValsFromResource(int8_t{});
  case 16:
    return isUnsigned ? pushValsFromResource(uint16_t{}) : pushValsFromResource(int16_t{});
  case 32:
    return isUnsigned ? pushValsFromResource(uint32_t{}) : pushValsFromResource(int32_t{});
  case 64:
    return isUnsigned ? pushValsFromResource(uint64_t{}) : pushValsFromResource(int64_t{});
  default:
    return false;
  }
}

static bool isTensorOfF32(Type t) {
  auto st = llvm::dyn_cast<ShapedType>(t);
  return st && st.getElementType().isF32();
}

static bool isTensorOfPosit(Type t) {
  auto st = llvm::dyn_cast<ShapedType>(t);
  return st && llvm::isa<mlir::posit::PositType>(st.getElementType());
}

// Cast a binary-op operand to a posit tensor that keeps the operand's OWN shape,
// changing only the element type to posit. Broadcasting posit.add/sub/mul/div no
// longer require identical operand/result types, so a scalar / lower-rank operand
// (e.g. GPT-2 attention mask) must NOT be force-cast up to the result rank — that
// created rank-changing unrealized casts (rank-0 -> rank-N) that broke posit->krnl.
// If the operand is already a posit tensor, return it unchanged (preserve shape).
static Value castOperandToPositKeepShape(ConversionPatternRewriter &rewriter,
                                         Location loc, Value v, Type positElemTy) {
  Type vt = v.getType();
  if (isTensorOfPosit(vt))
    return v;
  Type tgt;
  if (auto rtt = llvm::dyn_cast<RankedTensorType>(vt))
    tgt = RankedTensorType::get(rtt.getShape(), positElemTy);
  else if (llvm::isa<UnrankedTensorType>(vt))
    tgt = UnrankedTensorType::get(positElemTy);
  else
    return v;
  if (vt == tgt)
    return v;
  return rewriter.create<UnrealizedConversionCastOp>(loc, tgt, v).getResult(0);
}

// ------------- ONNXAddOp -> posit.add -------------

// [FIX] Forward declaration: ONNXAddOpLowering uses this helper before its definition.
static Value broadcastPositConstantIfNeeded(ConversionPatternRewriter &rewriter,
                                           Location loc, Value rhs,
                                           RankedTensorType outRtt,
                                           unsigned storageBits);
static Value buildZeroPositTensorConst(ConversionPatternRewriter &rewriter,
                                       Location loc, Type positTensorTy,
                                       unsigned storageBits);


struct ONNXAddOpLowering : public OpConversionPattern<mlir::ONNXAddOp> {
  using OpConversionPattern<mlir::ONNXAddOp>::OpConversionPattern;
  ONNXAddOpLowering(TypeConverter &tc, MLIRContext *ctx, unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXAddOp>(tc, ctx), nbits(nbits), es(es),
        storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXAddOp op,
                                typename OpConversionPattern::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 把 result type 用 TypeConverter 轉成 posit 型別 (tensor<...x!posit.type<8,0>>)
    Type origOutType = op.getResult().getType();
    Type convertedType = getTypeConverter()->convertType(origOutType);
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // adaptor 的 operands 已經是「轉換後」的型別
    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2 operands");

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    // posit.add requires identical operand/result types. When ONNX output type is
    // unranked, pick a ranked operand type (if available) for stable lowering.
    Type addTy = convertedType;
    if (llvm::isa<UnrankedTensorType>(addTy)) {
      if (auto lhsRtt = llvm::dyn_cast<RankedTensorType>(lhs.getType()))
        addTy = lhsRtt;
      else if (auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType()))
        addTy = rhsRtt;
    }

    // [FIX] Handle MNIST bias broadcasting for RHS posit constants.
    auto addRtt = llvm::dyn_cast<RankedTensorType>(addTy);
    auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType());
    // If ranks differ, first expand RHS rank by prefixing ones (e.g., 1000 -> 1x1000).
    // This avoids unresolved rank-changing tensor casts later in Krnl/LLVM lowering.
    if (addRtt && rhsRtt && rhsRtt.hasRank() &&
        rhsRtt.getRank() < addRtt.getRank() && rhsRtt.hasStaticShape()) {
      SmallVector<int64_t, 4> expanded(addRtt.getRank(), 1);
      for (int64_t i = 0, e = rhsRtt.getRank(); i < e; ++i)
        expanded[addRtt.getRank() - rhsRtt.getRank() + i] = rhsRtt.getDimSize(i);
      auto expandedTy =
          RankedTensorType::get(expanded, rhsRtt.getElementType());
      auto shapeTy = RankedTensorType::get(
          {static_cast<int64_t>(expanded.size())}, rewriter.getI64Type());
      SmallVector<Attribute, 4> shapeAttrs;
      shapeAttrs.reserve(expanded.size());
      for (int64_t d : expanded)
        shapeAttrs.push_back(rewriter.getI64IntegerAttr(d));
      Value shapeCst = rewriter.create<arith::ConstantOp>(
          loc, DenseIntElementsAttr::get(shapeTy, shapeAttrs));
      rhs = rewriter
                .create<mlir::posit::ReshapeOp>(loc, expandedTy, rhs, shapeCst)
                .getResult();
      rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType());
    }

    if (addRtt && rhs && rhs.getType() != addTy) {
      if (Value bc =
              broadcastPositConstantIfNeeded(rewriter, loc, rhs, addRtt, storageBits))
        rhs = bc;
    }

    Type addElemTy = llvm::cast<ShapedType>(addTy).getElementType();
    lhs = castOperandToPositKeepShape(rewriter, loc, lhs, addElemTy);
    rhs = castOperandToPositKeepShape(rewriter, loc, rhs, addElemTy);

    auto addOp = rewriter.create<mlir::posit::AddOp>(loc, addTy, lhs, rhs);
    addOp->setAttr("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));
    Value addRes = addOp.getResult();
    if (addRes.getType() != convertedType) {
      addRes = rewriter
                   .create<UnrealizedConversionCastOp>(loc, convertedType, addRes)
                   .getResult(0);
    }
    rewriter.replaceOp(op, addRes);
    return success();
  }
  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

// sub mul div 新增
struct ONNXSubOpLowering : public OpConversionPattern<mlir::ONNXSubOp> {
  using OpConversionPattern<mlir::ONNXSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXSubOp op,
                                typename OpConversionPattern::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type origOutType = op.getResult().getType();
    Type convertedType = getTypeConverter()->convertType(origOutType);
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2 operands");

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Type binTy = convertedType;
    if (llvm::isa<UnrankedTensorType>(binTy)) {
      if (auto lhsRtt = llvm::dyn_cast<RankedTensorType>(lhs.getType()))
        binTy = lhsRtt;
      else if (auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType()))
        binTy = rhsRtt;
    }
    Type binElemTy = llvm::cast<ShapedType>(binTy).getElementType();
    lhs = castOperandToPositKeepShape(rewriter, loc, lhs, binElemTy);
    rhs = castOperandToPositKeepShape(rewriter, loc, rhs, binElemTy);

    auto subOp = rewriter.create<mlir::posit::SubOp>(loc, binTy, lhs, rhs);
    subOp->setAttr("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));
    Value out = subOp.getResult();
    if (out.getType() != convertedType)
      out = rewriter
                .create<UnrealizedConversionCastOp>(loc, convertedType, out)
                .getResult(0);
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ONNXMulOpLowering : public OpConversionPattern<mlir::ONNXMulOp> {
  using OpConversionPattern<mlir::ONNXMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXMulOp op,
                                typename OpConversionPattern::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type origOutType = op.getResult().getType();
    Type convertedType = getTypeConverter()->convertType(origOutType);
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2 operands");

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Type binTy = convertedType;
    if (llvm::isa<UnrankedTensorType>(binTy)) {
      if (auto lhsRtt = llvm::dyn_cast<RankedTensorType>(lhs.getType()))
        binTy = lhsRtt;
      else if (auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType()))
        binTy = rhsRtt;
    }
    Type binElemTy = llvm::cast<ShapedType>(binTy).getElementType();
    lhs = castOperandToPositKeepShape(rewriter, loc, lhs, binElemTy);
    rhs = castOperandToPositKeepShape(rewriter, loc, rhs, binElemTy);

    auto mulOp = rewriter.create<mlir::posit::MulOp>(loc, binTy, lhs, rhs);
    mulOp->setAttr("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));
    Value out = mulOp.getResult();
    if (out.getType() != convertedType)
      out = rewriter
                .create<UnrealizedConversionCastOp>(loc, convertedType, out)
                .getResult(0);
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ONNXDivOpLowering : public OpConversionPattern<mlir::ONNXDivOp> {
  using OpConversionPattern<mlir::ONNXDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXDivOp op,
                                typename OpConversionPattern::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type origOutType = op.getResult().getType();
    Type convertedType = getTypeConverter()->convertType(origOutType);
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2 operands");

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Type binTy = convertedType;
    if (llvm::isa<UnrankedTensorType>(binTy)) {
      if (auto lhsRtt = llvm::dyn_cast<RankedTensorType>(lhs.getType()))
        binTy = lhsRtt;
      else if (auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType()))
        binTy = rhsRtt;
    }
    Type binElemTy = llvm::cast<ShapedType>(binTy).getElementType();
    lhs = castOperandToPositKeepShape(rewriter, loc, lhs, binElemTy);
    rhs = castOperandToPositKeepShape(rewriter, loc, rhs, binElemTy);

    auto divOp = rewriter.create<mlir::posit::DivOp>(loc, binTy, lhs, rhs);
    divOp->setAttr("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));
    Value out = divOp.getResult();
    if (out.getType() != convertedType)
      out = rewriter
                .create<UnrealizedConversionCastOp>(loc, convertedType, out)
                .getResult(0);
    rewriter.replaceOp(op, out);
    return success();
  }
};

// 需要新增i64 i32
 
// ONNXConstantOp -> posit.constant

//===----------------------------------------------------------------------===//
// [FIX] MNIST 需要的非 elementwise op：Conv / Relu / MaxPool / Reshape / MatMul
//===----------------------------------------------------------------------===//

// 產生一個全 0 的 posit tensor constant（用於 MatMul -> Gemm 的 C，beta=0 時可忽略）。
static Value buildZeroPositTensorConst(ConversionPatternRewriter &rewriter, Location loc,
                                      Type positTensorTy, unsigned storageBits) {
  auto rtt = llvm::dyn_cast<RankedTensorType>(positTensorTy);
  if (!rtt)
    return Value();
  // 目前 pass 固定 nbits=8, es=0。
  auto intElemTy = rewriter.getIntegerType(storageBits);
  auto intTensorTy = RankedTensorType::get(rtt.getShape(), intElemTy);

  // [FIX] Use the Attribute-based DenseElementsAttr::get for compatibility across MLIR versions.
  SmallVector<Attribute, 1> splat{IntegerAttr::get(intElemTy, 0)};
  DenseElementsAttr zeros = DenseElementsAttr::get(intTensorTy, splat);

  auto cst = rewriter.create<mlir::posit::ConstantOp>(loc, positTensorTy, zeros);
  return cst.getResult();
}


// [FIX][0127] Broadcast RHS posit.constant to match output tensor type when posit.add
// requires identical operand/result types. MNIST uses channel-bias constants shaped like
// (C,1,1) added to (N,C,H,W). We support general numpy-style broadcast for RHS constants.
static Value broadcastPositConstantIfNeeded(ConversionPatternRewriter &rewriter,
                                           Location loc, Value rhs,
                                           RankedTensorType outRtt,
                                           unsigned storageBits) {
  Value rhsBase = stripUnrealizedCast(rhs);
  if (!rhsBase || !outRtt)
    return Value();

  auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhsBase.getType());
  if (!rhsRtt || !rhsRtt.hasStaticShape() || !outRtt.hasStaticShape())
    return Value();

  if (rhsRtt == outRtt)
    return rhsBase;

  SmallVector<int64_t, 4> outShape(outRtt.getShape().begin(), outRtt.getShape().end());
  SmallVector<int64_t, 4> rhsShape(rhsRtt.getShape().begin(), rhsRtt.getShape().end());
  const int outRank = (int)outShape.size();
  const int rhsRank = (int)rhsShape.size();

  // Align ranks by prefixing ones.
  SmallVector<int64_t, 4> rhsAligned(outRank, 1);
  for (int i = 0; i < rhsRank; ++i)
    rhsAligned[outRank - rhsRank + i] = rhsShape[i];

  // Validate broadcast.
  for (int i = 0; i < outRank; ++i) {
    if (rhsAligned[i] != 1 && rhsAligned[i] != outShape[i])
      return Value();
  }

  auto computeStrides = [](ArrayRef<int64_t> shape) {
    SmallVector<int64_t, 4> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
  };

  SmallVector<int64_t, 4> outStrides = computeStrides(outShape);
  SmallVector<int64_t, 4> rhsStrides = computeStrides(rhsAligned);

  // Case 1: rhs is posit.constant(i<bits>) -> broadcast in bit-domain.
  if (auto rhsCst = rhsBase.getDefiningOp<mlir::posit::ConstantOp>()) {
    Attribute a = rhsCst->getAttr("value");
    if (!a)
      a = rhsCst->getAttr("valueAttr");
    auto bitsEA = llvm::dyn_cast_or_null<ElementsAttr>(a);
    auto dense = llvm::dyn_cast_or_null<DenseElementsAttr>(bitsEA);
    auto intElemTy = rewriter.getIntegerType(storageBits);
    if (!dense || dense.getElementType() != intElemTy)
      return Value();

    SmallVector<APInt, 64> rhsBits;
    rhsBits.reserve((size_t)rhsRtt.getNumElements());
    if (dense.isSplat()) {
      APInt ap = dense.getSplatValue<APInt>();
      rhsBits.assign((size_t)rhsRtt.getNumElements(),
          APInt(storageBits, ap.getZExtValue()));
    } else {
      for (APInt ap : dense.getValues<APInt>())
        rhsBits.push_back(APInt(storageBits, ap.getZExtValue()));
    }

    const int64_t outNumElts = outRtt.getNumElements();
    SmallVector<APInt, 64> outBits(outNumElts);
    for (int64_t lin = 0; lin < outNumElts; ++lin) {
      int64_t rem = lin;
      int64_t rhsLin = 0;
      for (int d = 0; d < outRank; ++d) {
        int64_t idx = rem / outStrides[d];
        rem = rem % outStrides[d];
        int64_t rIdx = (rhsAligned[d] == 1) ? 0 : idx;
        rhsLin += rIdx * rhsStrides[d];
      }
      outBits[(size_t)lin] = rhsBits[(size_t)rhsLin];
    }

    auto outBitsTy = RankedTensorType::get(outShape, intElemTy);
    SmallVector<Attribute, 8> outAttrs;
    outAttrs.reserve((size_t)outNumElts);
    for (const APInt &v : outBits)
      outAttrs.push_back(IntegerAttr::get(intElemTy, v));

    DenseElementsAttr outBitsAttr = DenseElementsAttr::get(outBitsTy, outAttrs);
    auto newCst = rewriter.create<mlir::posit::ConstantOp>(loc, outRtt, outBitsAttr);
    return newCst.getResult();
  }

  // Case 2: rhs is posit.from_f32(arith.constant) -> broadcast in f32 domain,
  // then convert back with posit.from_f32 so add operands keep identical shape.
  if (auto fromF32 = rhsBase.getDefiningOp<mlir::posit::FromF32Op>()) {
    Value srcBase = stripUnrealizedCast(fromF32.getInput());
    auto srcCst = srcBase.getDefiningOp<mlir::arith::ConstantOp>();
    if (!srcCst)
      return Value();
    auto srcDense = llvm::dyn_cast<DenseElementsAttr>(srcCst.getValue());
    auto srcRtt = srcDense ? llvm::dyn_cast<RankedTensorType>(srcDense.getType())
                           : RankedTensorType();
    auto srcFp = srcDense ? llvm::dyn_cast<DenseFPElementsAttr>(srcDense)
                          : DenseFPElementsAttr();
    if (!srcDense || !srcRtt || !srcRtt.hasStaticShape() || !srcFp ||
        srcRtt.getShape() != rhsRtt.getShape())
      return Value();

    SmallVector<float, 64> rhsVals;
    rhsVals.reserve((size_t)rhsRtt.getNumElements());
    if (srcFp.isSplat()) {
      float v = srcFp.getSplatValue<APFloat>().convertToFloat();
      rhsVals.assign((size_t)rhsRtt.getNumElements(), v);
    } else {
      for (APFloat ap : srcFp.getValues<APFloat>())
        rhsVals.push_back(ap.convertToFloat());
    }

    const int64_t outNumElts = outRtt.getNumElements();
    SmallVector<float, 64> outVals(outNumElts);
    for (int64_t lin = 0; lin < outNumElts; ++lin) {
      int64_t rem = lin;
      int64_t rhsLin = 0;
      for (int d = 0; d < outRank; ++d) {
        int64_t idx = rem / outStrides[d];
        rem = rem % outStrides[d];
        int64_t rIdx = (rhsAligned[d] == 1) ? 0 : idx;
        rhsLin += rIdx * rhsStrides[d];
      }
      outVals[(size_t)lin] = rhsVals[(size_t)rhsLin];
    }

    auto f32Ty = rewriter.getF32Type();
    auto outF32Ty = RankedTensorType::get(outShape, f32Ty);
    SmallVector<Attribute, 8> outAttrs;
    outAttrs.reserve((size_t)outNumElts);
    for (float v : outVals)
      outAttrs.push_back(FloatAttr::get(f32Ty, v));
    auto outF32 = DenseElementsAttr::get(outF32Ty, outAttrs);
    Value outF32Cst = createONNXTensorConstant(rewriter, loc, outF32Ty, outF32);
    return rewriter.create<mlir::posit::FromF32Op>(loc, outRtt, outF32Cst)
        .getResult();
  }

  return Value();
}

// QDQ lowering strategy:
// 1) Keep QuantizeLinear as the integer-code boundary, like the original INT8 QDQ graph.
// 2) Lower DequantizeLinear to f32 when the original ONNX result is f32:
//      int8/uint8 -> qalign/ALPS/posit quant-dequant -> f32 runtime_x_dq.
// 3) Only legacy pure-posit islands use tensor<...x!posit.type> storage.
//
// This keeps following ONNX Conv/Gemm/Add/Relu in the f32 domain whenever the
// original QDQ graph had DQ -> f32 -> op.
struct ONNXDequantizeLinearOpLowering
    : public OpConversionPattern<mlir::ONNXDequantizeLinearOp> {
  using OpConversionPattern<mlir::ONNXDequantizeLinearOp>::OpConversionPattern;
  ONNXDequantizeLinearOpLowering(TypeConverter &tc, MLIRContext *ctx,
                                 unsigned nbits, unsigned es,
                                 bool alignToInt8QDomain,
                                 bool preferDirectF32FromQDQ,
                                 bool preferDirectPositFromQDQ = false,
                                 bool preferStoreAsPosit = false)
      : OpConversionPattern<mlir::ONNXDequantizeLinearOp>(tc, ctx), nbits(nbits),
        es(es), storageBits(getPositStorageBitWidth(nbits)),
        alignToInt8QDomain(alignToInt8QDomain),
        preferDirectF32FromQDQ(preferDirectF32FromQDQ),
        preferDirectPositFromQDQ(preferDirectPositFromQDQ),
        preferStoreAsPosit_(preferStoreAsPosit) {}

  LogicalResult matchAndRewrite(mlir::ONNXDequantizeLinearOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type origOutType = op.getResult().getType();
    // QDQ-format DequantizeLinear should materialize f32, matching ONNX QDQ
    // semantics: Q stores the low-precision/posit boundary, while DQ returns
    // runtime_x_dq as f32 for following ordinary ONNX ops.  Do not gate this
    // on preferDirectF32FromQDQ; --strict-qdq-mode still needs DQ -> f32.
    const bool boundaryF32Flow = isTensorOfF32(origOutType);
    auto positElemTy = mlir::posit::PositType::get(rewriter.getContext(), nbits, es);

    auto toPositShaped = [&](Type srcTy) -> Type {
      if (auto rtt = llvm::dyn_cast<RankedTensorType>(srcTy))
        return RankedTensorType::get(rtt.getShape(), positElemTy);
      if (auto utt = llvm::dyn_cast<UnrankedTensorType>(srcTy))
        return UnrankedTensorType::get(positElemTy);
      return Type();
    };
    auto computeQAlignKeyForQuantizeOp = [&](mlir::ONNXQuantizeLinearOp qop)
        -> int64_t {
      int64_t axis = qop.getAxis();
      bool inputSigned = true;
      if (auto stTy = llvm::dyn_cast<ShapedType>(qop.getResult().getType())) {
        if (auto it = llvm::dyn_cast<IntegerType>(stTy.getElementType()))
          inputSigned = !it.isUnsignedInteger();
      }
      bool hasZeroPoint = !llvm::isa<NoneType>(qop.getYZeroPoint().getType());
      int64_t scaleLen = 1;
      int64_t zpLen = hasZeroPoint ? 1 : 0;
      double scaleForKey = 1.0;
      int64_t zeroPointForKey = 0;

      RankedTensorType qScaleRtt;
      SmallVector<double, 8> qScaleVals;
      if (getDenseFloatTensorFromValue(qop.getYScale(), qScaleRtt, qScaleVals) &&
          !qScaleVals.empty()) {
        scaleLen = static_cast<int64_t>(qScaleVals.size());
        scaleForKey = qScaleVals.front();
      } else {
        (void)getScalarFloatFromValue(qop.getYScale(), scaleForKey);
      }

      if (hasZeroPoint) {
        RankedTensorType qZpRtt;
        SmallVector<int64_t, 8> qZpVals;
        if (getDenseIntegerTensorFromValue(
                qop.getYZeroPoint(), qZpRtt, qZpVals) &&
            !qZpVals.empty()) {
          zpLen = static_cast<int64_t>(qZpVals.size());
          zeroPointForKey = qZpVals.front();
        } else {
          (void)getScalarIntFromValue(qop.getYZeroPoint(), zeroPointForKey);
        }
      }
      return computeQAlignKey(op, axis, inputSigned, hasZeroPoint, scaleLen,
          zpLen, scaleForKey, zeroPointForKey);
    };

    auto createFromF32WithQAlignKey = [&](Type posTy, Value src,
                                          int64_t qalignKey) -> Value {
      auto from = rewriter.create<mlir::posit::FromF32Op>(loc, posTy, src);
      from->setAttr("qalign_key", rewriter.getI64IntegerAttr(qalignKey));
      return from.getResult();
    };

    // Materialize qalign-carrying boundary for f32 source.
    // If source is already posit, keep passthrough behavior (no extra
    // to_f32/from_f32 round-trip) and only cast when type needs alignment.
    auto materializeQAlignBoundary = [&](Type outPosTy, Value src,
                                         int64_t qalignKey) -> Value {
      src = stripUnrealizedCast(src);
      if (isTensorOfF32(src.getType()))
        return createFromF32WithQAlignKey(outPosTy, src, qalignKey);
      if (isTensorOfPosit(src.getType())) {
        if (src.getType() == outPosTy)
          return src;
        return rewriter
            .create<UnrealizedConversionCastOp>(loc, TypeRange{outPosTy}, src)
            .getResult(0);
      }
      return Value();
    };

    Type positOutType = toPositShaped(origOutType);
    if (!positOutType)
      return rewriter.notifyMatchFailure(
          op, "dequantize result must be a ranked/unranked tensor");
    Type finalOutType = boundaryF32Flow ? origOutType : positOutType;

    auto toFinalOutputType = [&](Value v) -> Value {
      if (!v)
        return Value();
      if (v.getType() == finalOutType)
        return v;
      if (boundaryF32Flow && isTensorOfPosit(v.getType()))
        return rewriter.create<mlir::posit::ToF32Op>(loc, finalOutType, v)
            .getResult();
      if (!boundaryF32Flow && isTensorOfF32(v.getType()))
        return rewriter.create<mlir::posit::FromF32Op>(loc, finalOutType, v)
            .getResult();
      if (v.getType() != finalOutType)
        return rewriter
            .create<UnrealizedConversionCastOp>(loc, TypeRange{finalOutType}, v)
            .getResult(0);
      return v;
    };

    auto buildRuntimeDQFromQParams = [&](Value qTensor, Value yScale,
                                        Value yZeroPoint, int64_t axisAttr,
                                        Type dqOutTy,
                                        Value origRef = Value()) -> Value {
      RankedTensorType scaleRtt;
      SmallVector<double, 16> scaleVals;
      if (!getDenseFloatTensorFromValueRecursive(yScale, scaleRtt, scaleVals) ||
          scaleVals.empty())
        return Value();
      double scale = scaleVals.front();
      bool perAxis = scaleVals.size() > 1;

      int64_t zeroPoint = 0;
      bool hasZeroPoint = false;
      RankedTensorType zpRtt;
      SmallVector<int64_t, 16> zpVals;
      if (!llvm::isa<NoneType>(yZeroPoint.getType())) {
        hasZeroPoint = true;
        if (getDenseIntegerTensorFromValue(yZeroPoint, zpRtt, zpVals)) {
          if (zpVals.empty())
            return Value();
          zeroPoint = zpVals.front();
          if (zpVals.size() > 1)
            perAxis = true;
        } else if (getScalarIntFromValue(yZeroPoint, zeroPoint)) {
          zpVals.push_back(zeroPoint);
        } else {
          return Value();
        }
      }

      int64_t axis = axisAttr;
      if (auto qTy = llvm::dyn_cast<ShapedType>(qTensor.getType())) {
        if (qTy.hasRank()) {
          int64_t rank = qTy.getRank();
          if (axis < 0)
            axis += rank;
          if (axis < 0 || axis >= rank)
            return Value();
          if (perAxis && !qTy.isDynamicDim(axis)) {
            int64_t axisDim = qTy.getDimSize(axis);
            if (axisDim > 0 && static_cast<int64_t>(scaleVals.size()) != axisDim &&
                static_cast<int64_t>(scaleVals.size()) != 1)
              return Value();
            if (hasZeroPoint && !zpVals.empty() &&
                static_cast<int64_t>(zpVals.size()) != axisDim &&
                static_cast<int64_t>(zpVals.size()) != 1)
              return Value();
          }
        } else if (alignToInt8QDomain && perAxis) {
          // In strict INT8-domain mode, per-axis qparams require ranked input so
          // axis normalization/broadcast legality can be validated.
          return Value();
        }
      } else if (alignToInt8QDomain && perAxis) {
        return Value();
      }

      bool inputSigned = true;
      if (auto stTy = llvm::dyn_cast<ShapedType>(qTensor.getType())) {
        if (auto it = llvm::dyn_cast<IntegerType>(stTy.getElementType()))
          inputSigned = !it.isUnsignedInteger();
      }

      int64_t qalignKey = computeQAlignKey(op, axis, inputSigned, hasZeroPoint,
          static_cast<int64_t>(scaleVals.size()),
          hasZeroPoint ? static_cast<int64_t>(zpVals.size()) : 0, scale,
          zeroPoint);
      OperationState st(loc, mlir::posit::DequantizeLinearOp::getOperationName());
      SmallVector<Value, 2> dqOperands;
      dqOperands.push_back(qTensor);
      if (origRef)
        dqOperands.push_back(origRef);
      st.addOperands(dqOperands);
      st.addTypes({dqOutTy});
      st.addAttribute(
          "scale", FloatAttr::get(rewriter.getF32Type(), static_cast<float>(scale)));
      st.addAttribute("zero_point", rewriter.getI64IntegerAttr(zeroPoint));
      st.addAttribute("has_zero_point", rewriter.getBoolAttr(hasZeroPoint));
      st.addAttribute("axis", rewriter.getI64IntegerAttr(axis));
      st.addAttribute("input_signed", rewriter.getBoolAttr(inputSigned));
      st.addAttribute("qalign_key", rewriter.getI64IntegerAttr(qalignKey));

      if (perAxis) {
        auto f32Ty = rewriter.getF32Type();
        auto scaleTy =
            RankedTensorType::get({static_cast<int64_t>(scaleVals.size())}, f32Ty);
        SmallVector<Attribute, 16> sAttrs;
        sAttrs.reserve(scaleVals.size());
        for (double v : scaleVals)
          sAttrs.push_back(FloatAttr::get(f32Ty, static_cast<float>(v)));
        st.addAttribute("scale_values", DenseElementsAttr::get(scaleTy, sAttrs));

        if (hasZeroPoint) {
          if (zpVals.empty())
            zpVals.push_back(zeroPoint);
          auto i64Ty = rewriter.getI64Type();
          auto zpTy =
              RankedTensorType::get({static_cast<int64_t>(zpVals.size())}, i64Ty);
          SmallVector<APInt, 16> zpAp;
          zpAp.reserve(zpVals.size());
          for (int64_t v : zpVals)
            zpAp.push_back(APInt(64, static_cast<uint64_t>(v), true));
          st.addAttribute(
              "zero_point_values", DenseIntElementsAttr::get(zpTy, zpAp));
        }
      }

      return rewriter.create(st)->getResult(0);
    };

    Value xBase = stripUnrealizedCast(op.getX());

    // Case A-1: DQ(MaxPool(Q(...))).
    // Collapse quantized maxpool islands into posit maxpool to avoid leaving
    // ONNX QuantizeLinear ops that later conflict with mixed posit lowering.
    if (auto mp = xBase.getDefiningOp<mlir::ONNXMaxPoolSingleOutOp>()) {
      Value mpInBase = stripUnrealizedCast(mp.getX());
      if (auto qop = mpInBase.getDefiningOp<mlir::ONNXQuantizeLinearOp>()) {
        Value src = stripUnrealizedCast(qop.getX());
        if (Value remapped = rewriter.getRemappedValue(src))
          src = stripUnrealizedCast(remapped);

        Value srcPos;
        if (isTensorOfPosit(src.getType()) || isTensorOfF32(src.getType())) {
          Type srcPosTy = toPositShaped(src.getType());
          if (!srcPosTy)
            return rewriter.notifyMatchFailure(
                op, "failed to build posit type for maxpool-q path source");
          int64_t qalignKey = computeQAlignKeyForQuantizeOp(qop);
          srcPos = materializeQAlignBoundary(srcPosTy, src, qalignKey);
          if (!srcPos)
            return rewriter.notifyMatchFailure(
                op, "failed to materialize qalign boundary for maxpool-q path");
        } else {
          return rewriter.notifyMatchFailure(
              op, "maxpool-q path source must be f32/posit tensor");
        }

        OperationState st(loc, mlir::posit::MaxPool2DOp::getOperationName());
        st.addOperands({srcPos});
        st.addTypes({positOutType});
        if (auto a = mp->getAttr("kernel_shape"))
          st.addAttribute("kernel_shape", a);
        if (auto a = mp->getAttr("strides"))
          st.addAttribute("strides", a);
        if (auto a = mp->getAttr("pads"))
          st.addAttribute("pads", a);
        if (auto a = mp->getAttr("ceil_mode"))
          st.addAttribute("ceil_mode", a);
        if (auto a = mp->getAttr("auto_pad"))
          st.addAttribute("auto_pad", a);

        Value pooledPos = rewriter.create(st)->getResult(0);
        Value out = toFinalOutputType(pooledPos);
        if (!out)
          return rewriter.notifyMatchFailure(
              op, "failed to convert maxpool-q dequantize output to final type");
        rewriter.replaceOp(op, out);
        if (mp.getResult().use_empty())
          rewriter.eraseOp(mp);
        if (qop.getResult().use_empty())
          rewriter.eraseOp(qop);
        return success();
      }
    }


    // Case A0-conv: DQ(QLinearConv(...)).
    // QLinearConv is a QOperator: its main compute is quantized. For target-B,
    // keep that compute in the posit domain instead of pre-decomposing it into
    // DQ -> f32 Conv -> Q. The surrounding DequantizeLinear still returns the
    // original f32 result type when boundaryF32Flow is enabled, so normal QDQ
    // patterns (QuantizeLinear + DequantizeLinear + Conv) remain f32 after DQ.
    if (auto qconv = xBase.getDefiningOp<mlir::ONNXQLinearConvOp>()) {
      Operation *qconvOp = qconv.getOperation();
      if (qconvOp->getNumOperands() < 8 || qconvOp->getNumResults() != 1)
        return rewriter.notifyMatchFailure(op, "QLinearConv expects at least 8 operands");

      Value qX = qconvOp->getOperand(0);
      Value xScale = qconvOp->getOperand(1);
      Value xZeroPoint = qconvOp->getOperand(2);
      Value qW = qconvOp->getOperand(3);
      Value wScale = qconvOp->getOperand(4);
      Value wZeroPoint = qconvOp->getOperand(5);
      // QLinearConv operands 6/7 are output scale/zp. We intentionally do not
      // reconstruct the integer output and then DQ it here; the posit Conv
      // output represents the low-precision QOperator compute result directly,
      // consistent with the existing QLinearMatMul special path.
      (void)qconvOp->getOperand(6);
      (void)qconvOp->getOperand(7);
      Value qBias = qconvOp->getNumOperands() >= 9 ? qconvOp->getOperand(8) : Value();

      Type xPosTy = toPositShaped(qX.getType());
      Type wPosTy = toPositShaped(qW.getType());
      if (!xPosTy || !wPosTy)
        return rewriter.notifyMatchFailure(
            op, "QLinearConv supports only tensor X/W types");

      auto defaultAxisForConvInput = [&](Value v) -> int64_t {
        auto st = llvm::dyn_cast<ShapedType>(v.getType());
        if (st && st.hasRank() && st.getRank() > 1)
          return 1; // NCHW activation channel axis.
        return 0;
      };
      auto defaultAxisForConvWeight = [&](Value v) -> int64_t {
        (void)v;
        return 0; // OIHW output-channel axis.
      };

      auto buildPositOperandForQLinearConvInput = [&](Value qInput,
                                                      Value qScale,
                                                      Value qZeroPoint,
                                                      int64_t defaultAxis,
                                                      Type outPosTy) -> Value {
        Value inBase = stripUnrealizedCast(qInput);
        if (auto qop = inBase.getDefiningOp<mlir::ONNXQuantizeLinearOp>()) {
          Value src = stripUnrealizedCast(qop.getX());
          if (Value remapped = rewriter.getRemappedValue(src))
            src = stripUnrealizedCast(remapped);
          if (isTensorOfPosit(src.getType()) || isTensorOfF32(src.getType())) {
            Value v = materializeQAlignBoundary(
                outPosTy, src, computeQAlignKeyForQuantizeOp(qop));
            if (v)
              return v;
          }
        }
        return buildRuntimeDQFromQParams(
            qInput, qScale, qZeroPoint, defaultAxis, outPosTy);
      };

      Value xPos = buildPositOperandForQLinearConvInput(
          qX, xScale, xZeroPoint, defaultAxisForConvInput(qX), xPosTy);
      Value wPos = buildPositOperandForQLinearConvInput(
          qW, wScale, wZeroPoint, defaultAxisForConvWeight(qW), wPosTy);
      if (!xPos || !wPos)
        return rewriter.notifyMatchFailure(
            op, "failed to build posit X/W for QLinearConv");

      auto buildF32BiasForQLinearConv = [&]() -> Value {
        if (!qBias || llvm::isa<NoneType>(qBias.getType()))
          return Value();
        Type bF32Ty = Type();
        if (auto rtt = llvm::dyn_cast<RankedTensorType>(qBias.getType()))
          bF32Ty = RankedTensorType::get(rtt.getShape(), rewriter.getF32Type());
        else if (llvm::isa<UnrankedTensorType>(qBias.getType()))
          bF32Ty = UnrankedTensorType::get(rewriter.getF32Type());
        if (!bF32Ty)
          return Value();

        // Prefer compile-time folding for the common constant int32 bias case:
        // real_bias[c] = int32_bias[c] * x_scale * w_scale[c].
        RankedTensorType bRtt;
        SmallVector<int64_t, 64> bVals;
        RankedTensorType xScaleRtt;
        SmallVector<double, 8> xScaleVals;
        RankedTensorType wScaleRtt;
        SmallVector<double, 64> wScaleVals;
        if (getDenseIntegerTensorFromValue(qBias, bRtt, bVals) &&
            getDenseFloatTensorFromValue(xScale, xScaleRtt, xScaleVals) &&
            getDenseFloatTensorFromValue(wScale, wScaleRtt, wScaleVals) &&
            !xScaleVals.empty() && !wScaleVals.empty()) {
          int64_t biasElems = static_cast<int64_t>(bVals.size());
          int64_t wScaleElems = static_cast<int64_t>(wScaleVals.size());
          if (biasElems > 0 && (wScaleElems == 1 || wScaleElems == biasElems)) {
            auto f32Ty = rewriter.getF32Type();
            auto outTy = RankedTensorType::get(bRtt.getShape(), f32Ty);
            SmallVector<Attribute, 64> outAttrs;
            outAttrs.reserve(static_cast<size_t>(biasElems));
            double xS = xScaleVals.front();
            for (int64_t i = 0; i < biasElems; ++i) {
              double wS = (wScaleElems == 1) ? wScaleVals.front() : wScaleVals[i];
              double v = static_cast<double>(bVals[i]) * xS * wS;
              outAttrs.push_back(FloatAttr::get(f32Ty, static_cast<float>(v)));
            }
            DenseElementsAttr foldedAttr = DenseElementsAttr::get(outTy, outAttrs);
            OperationState cstSt(loc, "onnx.Constant");
            cstSt.addTypes({outTy});
            cstSt.addAttribute("value", foldedAttr);
            return rewriter.create(cstSt)->getResult(0);
          }
        }

        // Runtime fallback: Cast int32 bias to f32, multiply by x_scale*w_scale.
        SmallVector<NamedAttribute, 2> castAttrs;
        castAttrs.emplace_back(
            rewriter.getStringAttr("to"), TypeAttr::get(rewriter.getF32Type()));
        Operation *castBias = createGenericOpForQLinearConv(rewriter, loc, "onnx.Cast",
            TypeRange{bF32Ty}, ValueRange{qBias}, castAttrs);
        Operation *scaleMul = createGenericOpForQLinearConv(rewriter, loc, "onnx.Mul",
            TypeRange{bF32Ty}, ValueRange{xScale, wScale});
        Operation *biasMul = createGenericOpForQLinearConv(rewriter, loc, "onnx.Mul",
            TypeRange{bF32Ty},
            ValueRange{castBias->getResult(0), scaleMul->getResult(0)});
        return biasMul->getResult(0);
      };

      Value bPos;
      if (qBias && !llvm::isa<NoneType>(qBias.getType())) {
        Type bPosTy = toPositShaped(qBias.getType());
        if (!bPosTy)
          return rewriter.notifyMatchFailure(
              op, "failed to build posit bias type for QLinearConv");
        Value bF32 = buildF32BiasForQLinearConv();
        if (!bF32)
          return rewriter.notifyMatchFailure(
              op, "failed to build f32 bias for QLinearConv");
        auto from = rewriter.create<mlir::posit::FromF32Op>(loc, bPosTy, bF32);
        from->setAttr("qalign_key",
            rewriter.getI64IntegerAttr(computeTensorQAlignKey(qconvOp)));
        bPos = from.getResult();
      }

      if (!bPos) {
        int64_t c = ShapedType::kDynamic;
        Type elemTy;
        if (auto wTy = llvm::dyn_cast<RankedTensorType>(wPos.getType())) {
          if (wTy.getRank() >= 1) {
            c = wTy.getShape()[0];
            elemTy = wTy.getElementType();
          }
        }
        if ((c == ShapedType::kDynamic || !elemTy) &&
            llvm::isa<RankedTensorType>(positOutType)) {
          auto outRtt = llvm::cast<RankedTensorType>(positOutType);
          if (outRtt.getRank() >= 2) {
            c = outRtt.getShape()[1];
            elemTy = outRtt.getElementType();
          }
        }
        if (c == ShapedType::kDynamic || !elemTy)
          return rewriter.notifyMatchFailure(
              op, "cannot materialize QLinearConv default bias");
        bPos = buildZeroPositTensorConst(
            rewriter, loc, RankedTensorType::get({c}, elemTy), storageBits);
        if (!bPos)
          return rewriter.notifyMatchFailure(
              op, "failed to build default QLinearConv bias");
      }

      OperationState cst(loc, mlir::posit::Conv2DOp::getOperationName());
      cst.addOperands({xPos, wPos, bPos});
      cst.addTypes({positOutType});
      if (auto a = qconvOp->getAttr("strides"))
        cst.addAttribute("strides", a);
      if (auto a = qconvOp->getAttr("pads"))
        cst.addAttribute("pads", a);
      if (auto a = qconvOp->getAttr("dilations"))
        cst.addAttribute("dilations", a);
      if (auto a = qconvOp->getAttr("kernel_shape"))
        cst.addAttribute("kernel_shape", a);
      if (auto a = qconvOp->getAttr("group"))
        cst.addAttribute("group", a);
      if (auto a = qconvOp->getAttr("auto_pad"))
        cst.addAttribute("auto_pad", a);
      if (auto a = qconvOp->getAttr("onnx_node_name"))
        cst.addAttribute("onnx_node_name", a);
      cst.addAttribute("qalign_key",
          rewriter.getI64IntegerAttr(computeTensorQAlignKey(qconvOp)));
      Value convPos = rewriter.create(cst)->getResult(0);

      Value out = toFinalOutputType(convPos);
      if (!out)
        return rewriter.notifyMatchFailure(
            op, "failed to convert QLinearConv dequantize output to final type");
      rewriter.replaceOp(op, out);
      if (qconv.getResult().use_empty())
        rewriter.eraseOp(qconv);
      return success();
    }

    // Case A0: DQ(QLinearMatMul(...)).
    if (auto qmm = xBase.getDefiningOp<mlir::ONNXQLinearMatMulOp>()) {
      Type aPosTy = toPositShaped(qmm.getA().getType());
      Type bPosTy = toPositShaped(qmm.getB().getType());
      if (!aPosTy || !bPosTy)
        return rewriter.notifyMatchFailure(
            op, "QLinearMatMul supports only tensor A/B types");

      auto defaultAxisFor = [&](Value v) -> int64_t {
        auto st = llvm::dyn_cast<ShapedType>(v.getType());
        if (st && st.hasRank() && st.getRank() > 0)
          return st.getRank() - 1;
        return 0;
      };

      auto buildPositOperandForQmmInput = [&](Value qInput, Value qScale,
                                              Value qZeroPoint,
                                              Type outPosTy) -> Value {
        Value inBase = stripUnrealizedCast(qInput);
        if (auto qop = inBase.getDefiningOp<mlir::ONNXQuantizeLinearOp>()) {
          Value src = stripUnrealizedCast(qop.getX());
          if (Value remapped = rewriter.getRemappedValue(src))
            src = stripUnrealizedCast(remapped);
          if (isTensorOfPosit(src.getType()) || isTensorOfF32(src.getType())) {
            Value v = materializeQAlignBoundary(
                outPosTy, src, computeQAlignKeyForQuantizeOp(qop));
            if (v)
              return v;
          }
        }
        return buildRuntimeDQFromQParams(qInput, qScale, qZeroPoint,
            defaultAxisFor(qInput), outPosTy);
      };

      Value aPos = buildPositOperandForQmmInput(
          qmm.getA(), qmm.getAScale(), qmm.getAZeroPoint(), aPosTy);
      Value bPos = buildRuntimeDQFromQParams(qmm.getB(), qmm.getBScale(),
          qmm.getBZeroPoint(), defaultAxisFor(qmm.getB()), bPosTy);
      if (!aPos || !bPos)
        return rewriter.notifyMatchFailure(
            op, "failed to build scalar/per-axis qparams for QLinearMatMul");
      Value c = buildZeroPositTensorConst(
          rewriter, loc, RankedTensorType::get({1}, positElemTy), storageBits);
      if (!c)
        return rewriter.notifyMatchFailure(
            op, "failed to build GEMM C=0 constant for QLinearMatMul");

      OperationState gst(loc, mlir::posit::GemmOp::getOperationName());
      gst.addOperands({aPos, bPos, c});
      gst.addTypes({positOutType});
      gst.addAttribute("alpha", FloatAttr::get(rewriter.getF32Type(), 1.0));
      gst.addAttribute("beta", FloatAttr::get(rewriter.getF32Type(), 0.0));
      gst.addAttribute("transA", rewriter.getI64IntegerAttr(0));
      gst.addAttribute("transB", rewriter.getI64IntegerAttr(0));
      Value gemmPos = rewriter.create(gst)->getResult(0);

      Value out = toFinalOutputType(gemmPos);
      if (!out)
        return rewriter.notifyMatchFailure(
            op, "failed to convert QLinearMatMul dequantize output to final type");
      rewriter.replaceOp(op, out);
      if (auto aQ = stripUnrealizedCast(qmm.getA())
                        .getDefiningOp<mlir::ONNXQuantizeLinearOp>())
        if (aQ.getResult().use_empty())
          rewriter.eraseOp(aQ);
      if (qmm.getResult().use_empty())
        rewriter.eraseOp(qmm);
      return success();
    }

    // Case A: DQ(Q(...)).
    if (auto qop = xBase.getDefiningOp<mlir::ONNXQuantizeLinearOp>()) {
      Value qTensor = rewriter.getRemappedValue(qop.getResult());
      if (!qTensor && !adaptor.getOperands().empty())
        qTensor = adaptor.getOperands()[0];
      if (!qTensor)
        qTensor = op.getX();
      qTensor = stripUnrealizedCast(qTensor);

      Value src = stripUnrealizedCast(qop.getX());
      Value srcRemapped = rewriter.getRemappedValue(src);
      if (srcRemapped)
        src = stripUnrealizedCast(srcRemapped);
      auto rewriteDirectFromQSource = [&]() -> LogicalResult {
        if (!isTensorOfF32(src.getType()) && !isTensorOfPosit(src.getType()))
          return rewriter.notifyMatchFailure(
              op, "Q->DQ source must be f32/posit tensor");
        Type posTy = positOutType;
        if (!posTy)
          return rewriter.notifyMatchFailure(
              op, "failed to build posit type for fake QDQ tensor");
        int64_t qalignKey = computeQAlignKeyForQuantizeOp(qop);
        Value pos = materializeQAlignBoundary(posTy, src, qalignKey);
        if (!pos)
          return rewriter.notifyMatchFailure(
              op, "failed to materialize qalign boundary for Q->DQ source");
        Value out = toFinalOutputType(pos);
        if (!out)
          return rewriter.notifyMatchFailure(
              op, "failed to convert Q->DQ boundary output to final type");
        rewriter.replaceOp(op, out);
        if (qop.getResult().use_empty())
          rewriter.eraseOp(qop);
        return success();
      };

      // Non-strict mode historically preferred direct f32->posit from Q source.
      // For QDQ-compatible f32 flow, first try strict Q-domain DQ so the
      // following Conv/Gemm/Add/etc. see f32 values like qdq-f32.
      //
      // POSIT_PREFER_DIRECT_FROM_QDQ: when set, also try direct posit path for
      // f32-boundary DQ nodes (boundaryF32Flow=true). This encodes the original
      // pre-quantization f32 value through posit (f32→posit→f32), bypassing the
      // INT8 DQ step. This is "Variant A" in the qdq benchmark. The downstream
      // onnx.Conv still receives f32, but posit-rounded from orig_f32 not int8.
      if ((preferDirectPositFromQDQ || !boundaryF32Flow) && preferDirectF32FromQDQ &&
          succeeded(rewriteDirectFromQSource()))
        return success();

      // Strict ONNX QDQ path (round/clamp + scale/zp) when possible.
      //
      // POSIT_STORE_DQ_AS_POSIT: when set, even for f32-boundary DQ nodes,
      // emit posit bits output (via posit_dequantize_linear_ref) and insert
      // posit.to_f32 before the downstream onnx.Conv. This makes posit bits
      // the persistent storage format at layer boundaries (like INT8 stores int8
      // bits), decoded only when the next computation needs f32. The flow is:
      //   INT8 → posit_dequantize_linear_ref → [posit bits buffer] → posit.to_f32 → f32
      // This is "Variant B persistent" in the qdq benchmark.
      if (boundaryF32Flow && preferStoreAsPosit_) {
        if (Value dqPos = buildRuntimeDQFromQParams(
                qTensor, qop.getYScale(), qop.getYZeroPoint(), qop.getAxis(),
                positOutType, isTensorOfF32(src.getType()) ? src : Value())) {
          Value out = toFinalOutputType(dqPos);  // inserts posit.to_f32 → f32
          if (!out)
            return rewriter.notifyMatchFailure(
                op, "POSIT_STORE_DQ_AS_POSIT: failed to convert posit bits to f32");
          rewriter.replaceOp(op, out);
          if (qop.getResult().use_empty())
            rewriter.eraseOp(qop);
          return success();
        }
      }

      // If the original ONNX DQ result is f32, produce f32 directly. This stores
      // runtime_x_dq, not the final p8 storage value, and lets downstream ONNX
      // ops stay in the f32 domain.
      if (boundaryF32Flow) {
        if (Value dqF32 = buildRuntimeDQFromQParams(
                qTensor, qop.getYScale(), qop.getYZeroPoint(), qop.getAxis(),
                finalOutType, isTensorOfF32(src.getType()) ? src : Value())) {
          rewriter.replaceOp(op, dqF32);
          if (qop.getResult().use_empty())
            rewriter.eraseOp(qop);
          return success();
        }
        if (alignToInt8QDomain) {
          return rewriter.notifyMatchFailure(
              op, "align-to-int8-qdomain requires materializable scale/zp for Q->DQ f32 output");
        }
      }

      if (Value pos = buildRuntimeDQFromQParams(
              qTensor, qop.getYScale(), qop.getYZeroPoint(), qop.getAxis(),
              positOutType, isTensorOfF32(src.getType()) ? src : Value())) {
        Value out = toFinalOutputType(pos);
        if (!out)
          return rewriter.notifyMatchFailure(
              op, "failed to convert strict Q->DQ output to final type");
        rewriter.replaceOp(op, out);
        if (qop.getResult().use_empty())
          rewriter.eraseOp(qop);
        return success();
      }

      // Optional hard mode: require strict INT8 q-domain alignment and do not
      // fall back to direct f32->posit when qparams cannot be materialized.
      if (alignToInt8QDomain) {
        return rewriter.notifyMatchFailure(
            op, "align-to-int8-qdomain requires materializable scale/zp for Q->DQ");
      }

      // Strict path could not materialize qparams; fall back to direct source.
      return rewriteDirectFromQSource();
    }


    // Case B0: robust compile-time dequantization for integer constants.
    // This is especially important for int32 bias DequantizeLinear nodes such as
    //   tensor<32xi32>, tensor<1xf32>, tensor<1xi32> -> tensor<32xf32>
    // in QDQ models. In strict mode these are still ordinary QDQ/f32 semantics,
    // not posit QOperator compute. Fold them to f32 constants so no
    // onnx.DequantizeLinear remains illegal.
    {
      RankedTensorType xConstRtt;
      SmallVector<int64_t, 64> xConstVals;
      RankedTensorType sConstRtt;
      SmallVector<double, 64> sConstVals;
      RankedTensorType zConstRtt;
      SmallVector<int64_t, 64> zConstVals;
      bool hasZpConst = !llvm::isa<NoneType>(op.getXZeroPoint().getType());
      bool haveZpConst = !hasZpConst ||
          getDenseIntegerTensorFromValue(op.getXZeroPoint(), zConstRtt, zConstVals) ||
          [&]() {
            int64_t z = 0;
            if (!getScalarIntFromValue(op.getXZeroPoint(), z))
              return false;
            zConstVals.clear();
            zConstVals.push_back(z);
            return true;
          }();

      if (getDenseIntegerTensorFromValue(op.getX(), xConstRtt, xConstVals) &&
          getDenseFloatTensorFromValueRecursive(op.getXScale(), sConstRtt, sConstVals) &&
          !xConstVals.empty() && !sConstVals.empty() && haveZpConst) {
        int64_t n = static_cast<int64_t>(xConstVals.size());
        int64_t sN = static_cast<int64_t>(sConstVals.size());
        int64_t zN = hasZpConst ? static_cast<int64_t>(zConstVals.size()) : 0;
        bool scaleOk = (sN == 1 || sN == n);
        bool zpOk = (!hasZpConst || zN == 1 || zN == n);
        if (scaleOk && zpOk) {
          auto f32Ty = rewriter.getF32Type();
          RankedTensorType outF32Ty;
          if (auto outRtt = llvm::dyn_cast<RankedTensorType>(origOutType))
            outF32Ty = RankedTensorType::get(outRtt.getShape(), f32Ty);
          else
            outF32Ty = RankedTensorType::get(xConstRtt.getShape(), f32Ty);

          SmallVector<Attribute, 64> outAttrs;
          outAttrs.reserve(static_cast<size_t>(n));
          for (int64_t i = 0; i < n; ++i) {
            double s = sConstVals[(sN == 1) ? 0 : static_cast<size_t>(i)];
            int64_t z = hasZpConst ? zConstVals[(zN == 1) ? 0 : static_cast<size_t>(i)] : 0;
            double v = (static_cast<double>(xConstVals[static_cast<size_t>(i)]) -
                        static_cast<double>(z)) * s;
            outAttrs.push_back(FloatAttr::get(f32Ty, static_cast<float>(v)));
          }
          DenseElementsAttr outAttr = DenseElementsAttr::get(outF32Ty, outAttrs);
          Value f32Const = createONNXTensorConstant(rewriter, loc, outF32Ty, outAttr);
          Value out = toFinalOutputType(f32Const);
          if (!out)
            return rewriter.notifyMatchFailure(
                op, "failed to convert folded integer dequantize constant");
          rewriter.replaceOp(op, out);
          return success();
        }
      }
    }

    // Case B: compile-time dequantization for int constants.
    // Keep this path strict (scalar scale/zp) and fall back to Case C for
    // per-axis qparams so QDQ conv-weight patterns can still legalize.
    RankedTensorType xRtt;
    SmallVector<int64_t, 16> xVals;
    if (getDenseIntegerTensorFromValue(op.getX(), xRtt, xVals)) {
      double scale = 0.0;
      bool canConstFold = getScalarFloatFromValue(op.getXScale(), scale);

      int64_t zeroPoint = 0;
      if (canConstFold && !llvm::isa<NoneType>(op.getXZeroPoint().getType()) &&
          !getScalarIntFromValue(op.getXZeroPoint(), zeroPoint)) {
        canConstFold = false;
      }

      if (!canConstFold || boundaryF32Flow) {
        // Let Case C handle scalar/per-axis runtime dequantization. In f32-flow
        // mode this avoids folding to an intermediate posit tensor; the runtime
        // DQ writes runtime_x_dq directly into f32 storage.
      } else {

        SmallVector<Attribute, 16> dequantAttrs;
        dequantAttrs.reserve(xVals.size());
        auto f32Ty = rewriter.getF32Type();
        for (int64_t x : xVals) {
          const double dequantized =
              (static_cast<double>(x) - static_cast<double>(zeroPoint)) * scale;
          dequantAttrs.push_back(
              FloatAttr::get(f32Ty, static_cast<float>(dequantized)));
        }

        auto f32TensorTy = RankedTensorType::get(xRtt.getShape(), f32Ty);
        DenseElementsAttr dequantAttr =
            DenseElementsAttr::get(f32TensorTy, dequantAttrs);
        Value f32Const =
            createONNXTensorConstant(rewriter, loc, f32TensorTy, dequantAttr);

        auto rankedPositTy = RankedTensorType::get(xRtt.getShape(), positElemTy);
        bool hasZeroPoint = !llvm::isa<NoneType>(op.getXZeroPoint().getType());
        bool inputSigned = true;
        if (auto stTy = llvm::dyn_cast<ShapedType>(op.getX().getType())) {
          if (auto it = llvm::dyn_cast<IntegerType>(stTy.getElementType()))
            inputSigned = !it.isUnsignedInteger();
        }
        int64_t qalignKey = computeQAlignKey(op, op.getAxis(),
            inputSigned, hasZeroPoint, /*scaleLen=*/1,
            hasZeroPoint ? /*zpLen=*/1 : /*zpLen=*/0, scale, zeroPoint);
        Value rankedPosit = createFromF32WithQAlignKey(
            rankedPositTy, f32Const, qalignKey);

        Value out = toFinalOutputType(rankedPosit);
        if (!out)
          return rewriter.notifyMatchFailure(
              op, "failed to convert folded dequantize output to final type");
        rewriter.replaceOp(op, out);
        return success();
      }
    }

    // Case C: runtime dequantization for non-constant integer tensors with
    // scalar or per-axis constant scale / zero_point.
    double scale = 0.0;
    int64_t zeroPoint = 0;
    bool hasZeroPoint = false;
    RankedTensorType scaleRtt;
    SmallVector<double, 16> scaleVals;
    if (getDenseFloatTensorFromValueRecursive(op.getXScale(), scaleRtt, scaleVals)) {
      if (scaleVals.empty())
        return rewriter.notifyMatchFailure(op, "x_scale cannot be empty");
      scale = scaleVals.front();
      bool perAxis = scaleVals.size() > 1;

      RankedTensorType zpRtt;
      SmallVector<int64_t, 16> zpVals;
      if (!llvm::isa<NoneType>(op.getXZeroPoint().getType())) {
        hasZeroPoint = true;
        if (getDenseIntegerTensorFromValue(op.getXZeroPoint(), zpRtt, zpVals)) {
          if (zpVals.empty())
            return rewriter.notifyMatchFailure(op, "x_zero_point cannot be empty");
          zeroPoint = zpVals.front();
          if (zpVals.size() > 1)
            perAxis = true;
        } else if (getScalarIntFromValue(op.getXZeroPoint(), zeroPoint)) {
          zpVals.push_back(zeroPoint);
        } else {
          return rewriter.notifyMatchFailure(
              op, "x_zero_point must be scalar/1D integer constant or none");
        }
      }

      int64_t axis = op.getAxis();
      if (auto xTy = llvm::dyn_cast<ShapedType>(op.getX().getType())) {
        if (xTy.hasRank()) {
          int64_t rank = xTy.getRank();
          if (axis < 0)
            axis += rank;
          if (axis < 0 || axis >= rank)
            return rewriter.notifyMatchFailure(op, "invalid dequantize axis");
          if (perAxis && xTy.isDynamicDim(axis) == false) {
            int64_t axisDim = xTy.getDimSize(axis);
            if (axisDim > 0 && static_cast<int64_t>(scaleVals.size()) != axisDim &&
                static_cast<int64_t>(scaleVals.size()) != 1)
              return rewriter.notifyMatchFailure(
                  op, "x_scale length must match axis dimension or be 1");
            if (hasZeroPoint && !zpVals.empty() &&
                static_cast<int64_t>(zpVals.size()) != axisDim &&
                static_cast<int64_t>(zpVals.size()) != 1)
              return rewriter.notifyMatchFailure(
                  op, "x_zero_point length must match axis dimension or be 1");
          }
        }
      }

      OperationState st(loc, mlir::posit::DequantizeLinearOp::getOperationName());
      st.addOperands({adaptor.getX()});
      st.addTypes({boundaryF32Flow ? finalOutType : positOutType});
      st.addAttribute(
          "scale", FloatAttr::get(rewriter.getF32Type(), static_cast<float>(scale)));
      st.addAttribute("zero_point", rewriter.getI64IntegerAttr(zeroPoint));
      st.addAttribute("has_zero_point", rewriter.getBoolAttr(hasZeroPoint));
      st.addAttribute("axis", rewriter.getI64IntegerAttr(axis));
      bool inputSigned = true;
      if (auto stTy = llvm::dyn_cast<ShapedType>(op.getX().getType())) {
        if (auto it = llvm::dyn_cast<IntegerType>(stTy.getElementType()))
          inputSigned = !it.isUnsignedInteger();
      }
      int64_t qalignKey = computeQAlignKey(op, axis, inputSigned, hasZeroPoint,
          static_cast<int64_t>(scaleVals.size()),
          hasZeroPoint ? static_cast<int64_t>(zpVals.size()) : 0, scale,
          zeroPoint);
      st.addAttribute("input_signed", rewriter.getBoolAttr(inputSigned));
      st.addAttribute("qalign_key", rewriter.getI64IntegerAttr(qalignKey));

      if (perAxis) {
        auto f32Ty = rewriter.getF32Type();
        auto scaleTy =
            RankedTensorType::get({static_cast<int64_t>(scaleVals.size())}, f32Ty);
        SmallVector<Attribute, 16> sAttrs;
        sAttrs.reserve(scaleVals.size());
        for (double v : scaleVals)
          sAttrs.push_back(FloatAttr::get(f32Ty, static_cast<float>(v)));
        st.addAttribute("scale_values", DenseElementsAttr::get(scaleTy, sAttrs));

        if (hasZeroPoint) {
          if (zpVals.empty())
            zpVals.push_back(zeroPoint);
          auto i64Ty = rewriter.getI64Type();
          auto zpTy =
              RankedTensorType::get({static_cast<int64_t>(zpVals.size())}, i64Ty);
          SmallVector<APInt, 16> zpAp;
          zpAp.reserve(zpVals.size());
          for (int64_t v : zpVals)
            zpAp.push_back(APInt(64, static_cast<uint64_t>(v), true));
          st.addAttribute(
              "zero_point_values", DenseIntElementsAttr::get(zpTy, zpAp));
        }
      }

      Operation *newOp = rewriter.create(st);
      Value out = toFinalOutputType(newOp->getResult(0));
      if (!out)
        return rewriter.notifyMatchFailure(
            op, "failed to convert runtime dequantize output to final type");
      rewriter.replaceOp(op, out);
      return success();
    }

    if (alignToInt8QDomain) {
      return rewriter.notifyMatchFailure(
          op, "align-to-int8-qdomain requires constant materializable x_scale/x_zero_point for dequantize");
    }

    // Case D: passthrough when source was already converted/remapped.
    Value remappedX = rewriter.getRemappedValue(op.getX());
    if (!remappedX && !adaptor.getOperands().empty())
      remappedX = adaptor.getOperands()[0];
    if (!remappedX)
      remappedX = op.getX();

    if (isTensorOfF32(remappedX.getType())) {
      if (alignToInt8QDomain)
        return rewriter.notifyMatchFailure(
            op, "align-to-int8-qdomain forbids dequantize fallback passthrough from f32");
      Type remappedPosTy = toPositShaped(remappedX.getType());
      if (!remappedPosTy)
        return rewriter.notifyMatchFailure(
            op, "failed to build posit type for fallback dequantize source");
      int64_t axis = op.getAxis();
      bool inputSigned = true;
      if (auto stTy = llvm::dyn_cast<ShapedType>(op.getX().getType())) {
        if (auto it = llvm::dyn_cast<IntegerType>(stTy.getElementType()))
          inputSigned = !it.isUnsignedInteger();
      }
      bool hasZeroPoint = !llvm::isa<NoneType>(op.getXZeroPoint().getType());
      int64_t scaleLen = 1;
      int64_t zpLen = hasZeroPoint ? 1 : 0;
      double scaleForKey = 1.0;
      int64_t zeroPointForKey = 0;
      RankedTensorType sRtt;
      SmallVector<double, 8> sVals;
      if (getDenseFloatTensorFromValue(op.getXScale(), sRtt, sVals) &&
          !sVals.empty()) {
        scaleLen = static_cast<int64_t>(sVals.size());
        scaleForKey = sVals.front();
      } else {
        (void)getScalarFloatFromValue(op.getXScale(), scaleForKey);
      }
      if (hasZeroPoint) {
        RankedTensorType zRtt;
        SmallVector<int64_t, 8> zVals;
        if (getDenseIntegerTensorFromValue(op.getXZeroPoint(), zRtt, zVals) &&
            !zVals.empty()) {
          zpLen = static_cast<int64_t>(zVals.size());
          zeroPointForKey = zVals.front();
        } else {
          (void)getScalarIntFromValue(op.getXZeroPoint(), zeroPointForKey);
        }
      }
      int64_t qalignKey = computeQAlignKey(op, axis, inputSigned, hasZeroPoint,
          scaleLen, zpLen, scaleForKey, zeroPointForKey);
      Value remappedPos =
          createFromF32WithQAlignKey(remappedPosTy, remappedX, qalignKey);
      Value out = toFinalOutputType(remappedPos);
      if (!out)
        return rewriter.notifyMatchFailure(
            op, "failed to convert fallback f32 source to final type");
      rewriter.replaceOp(op, out);
      return success();
    }
    if (isTensorOfPosit(remappedX.getType())) {
      if (alignToInt8QDomain)
        return rewriter.notifyMatchFailure(
            op, "align-to-int8-qdomain forbids dequantize fallback passthrough from posit");
      Value out = toFinalOutputType(remappedX);
      if (!out)
        return rewriter.notifyMatchFailure(
            op, "failed to convert fallback posit source to final type");
      rewriter.replaceOp(op, out);
      return success();
    }

    return rewriter.notifyMatchFailure(op,
        "dequantize not matched (expected Q->DQ, const fold, or runtime dequant)");
  }

  unsigned nbits;
  unsigned es;
  unsigned storageBits;
  bool alignToInt8QDomain;
  bool preferDirectF32FromQDQ;
  bool preferDirectPositFromQDQ = false;  // POSIT_PREFER_DIRECT_FROM_QDQ
  bool preferStoreAsPosit_ = false;       // POSIT_STORE_DQ_AS_POSIT (Variant B persistent)
};

struct ONNXReshapeOpLowering : public OpConversionPattern<mlir::ONNXReshapeOp> {
  using OpConversionPattern<mlir::ONNXReshapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert reshape result type");

    // ONNXReshape operands: data, shape (shape tensor stays i64 by TypeConverter policy)
    Value data = adaptor.getOperands()[0];
    Value shape = adaptor.getOperands()[1];

    OperationState st(loc, mlir::posit::ReshapeOp::getOperationName());
    st.addOperands({data, shape});
    st.addTypes({outTy});
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXUnsqueezeOpLowering : public OpConversionPattern<mlir::ONNXUnsqueezeOp> {
  using OpConversionPattern<mlir::ONNXUnsqueezeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXUnsqueezeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    auto outRtt = llvm::dyn_cast_or_null<RankedTensorType>(outTy);
    if (!outRtt)
      return rewriter.notifyMatchFailure(op, "expected ranked result type");

    Value data = adaptor.getData();
    auto inRtt = llvm::dyn_cast<RankedTensorType>(data.getType());
    if (!inRtt)
      return rewriter.notifyMatchFailure(op, "expected ranked input type");

    RankedTensorType axesTy;
    SmallVector<int64_t, 8> axesVals;
    if (!getDenseIntegerTensorFromValue(op.getAxes(), axesTy, axesVals))
      return rewriter.notifyMatchFailure(op, "axes must be constant integer tensor");

    const int64_t outRank = outRtt.getRank();
    SmallVector<char, 8> isUnsqueezeAxis(static_cast<size_t>(outRank), 0);
    for (int64_t axis : axesVals) {
      int64_t a = axis < 0 ? axis + outRank : axis;
      if (a < 0 || a >= outRank)
        return rewriter.notifyMatchFailure(op, "axis out of range");
      isUnsqueezeAxis[static_cast<size_t>(a)] = 1;
    }

    SmallVector<Value, 8> shapeElems;
    shapeElems.reserve(static_cast<size_t>(outRank));
    int64_t inPos = 0;
    IntegerType i64Ty = rewriter.getI64Type();
    for (int64_t outPos = 0; outPos < outRank; ++outPos) {
      if (isUnsqueezeAxis[static_cast<size_t>(outPos)]) {
        shapeElems.push_back(rewriter.create<arith::ConstantOp>(
            loc, i64Ty, rewriter.getI64IntegerAttr(1)));
        continue;
      }

      if (inPos >= inRtt.getRank())
        return rewriter.notifyMatchFailure(op, "rank mismatch in unsqueeze lowering");

      if (!inRtt.isDynamicDim(inPos)) {
        shapeElems.push_back(rewriter.create<arith::ConstantOp>(loc, i64Ty,
            rewriter.getI64IntegerAttr(inRtt.getDimSize(inPos))));
      } else {
        Value inAxis = rewriter.create<arith::ConstantIndexOp>(loc, inPos);
        Value dimIdx = rewriter.create<tensor::DimOp>(loc, data, inAxis);
        Value dimI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, dimIdx);
        shapeElems.push_back(dimI64);
      }
      ++inPos;
    }

    auto shapeTy = RankedTensorType::get({outRank}, i64Ty);
    Value shapeTensor =
        rewriter.create<tensor::FromElementsOp>(loc, shapeTy, shapeElems);

    OperationState st(loc, mlir::posit::ReshapeOp::getOperationName());
    st.addOperands({data, shapeTensor});
    st.addTypes({outTy});
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXReluOpLowering : public OpConversionPattern<mlir::ONNXReluOp> {
  using OpConversionPattern<mlir::ONNXReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert relu result type");

    Value x = adaptor.getOperands()[0];
    OperationState st(loc, mlir::posit::ReluOp::getOperationName());
    st.addOperands({x});
    st.addTypes({outTy});
    st.addAttribute("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXClipOpLowering : public OpConversionPattern<mlir::ONNXClipOp> {
  using OpConversionPattern<mlir::ONNXClipOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXClipOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert clip result type");

    Value x = adaptor.getInput();
    OperationState st(loc, mlir::posit::ClipOp::getOperationName());
    st.addOperands({x});
    st.addTypes({outTy});

    double minVal = 0.0;
    double maxVal = 0.0;
    bool hasMin = false;
    bool hasMax = false;
    Value minInput = op.getMin();
    Value maxInput = op.getMax();
    if (minInput && !llvm::isa<NoneType>(minInput.getType())) {
      hasMin = getScalarFloatFromValue(minInput, minVal);
      if (!hasMin)
        return rewriter.notifyMatchFailure(op, "clip min must be scalar constant");
    }
    if (maxInput && !llvm::isa<NoneType>(maxInput.getType())) {
      hasMax = getScalarFloatFromValue(maxInput, maxVal);
      if (!hasMax)
        return rewriter.notifyMatchFailure(op, "clip max must be scalar constant");
    }

    st.addAttribute("has_min", rewriter.getBoolAttr(hasMin));
    st.addAttribute("has_max", rewriter.getBoolAttr(hasMax));
    st.addAttribute("min_val", FloatAttr::get(rewriter.getF32Type(),
        static_cast<float>(minVal)));
    st.addAttribute("max_val", FloatAttr::get(rewriter.getF32Type(),
        static_cast<float>(maxVal)));
    st.addAttribute("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));

    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXMaxPoolSingleOutOpLowering
    : public OpConversionPattern<mlir::ONNXMaxPoolSingleOutOp> {
  using OpConversionPattern<mlir::ONNXMaxPoolSingleOutOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert maxpool result type");

    Value x = adaptor.getOperands()[0];

    OperationState st(loc, mlir::posit::MaxPool2DOp::getOperationName());
    st.addOperands({x});
    st.addTypes({outTy});

    // attrs: kernel_shape / strides / pads / ceil_mode (照 ONNX attr)
    if (auto a = op->getAttr("kernel_shape")) st.addAttribute("kernel_shape", a);
    if (auto a = op->getAttr("strides"))      st.addAttribute("strides", a);
    if (auto a = op->getAttr("pads"))         st.addAttribute("pads", a);
    if (auto a = op->getAttr("ceil_mode"))    st.addAttribute("ceil_mode", a);
    // optional: auto_pad (不一定用得到，但保留)
    if (auto a = op->getAttr("auto_pad"))     st.addAttribute("auto_pad", a);
    st.addAttribute("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));

    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXFlattenOpLowering : public OpConversionPattern<mlir::ONNXFlattenOp> {
  using OpConversionPattern<mlir::ONNXFlattenOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXFlattenOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert flatten result type");

    OperationState st(loc, mlir::posit::FlattenOp::getOperationName());
    st.addOperands({adaptor.getInput()});
    st.addTypes({outTy});
    st.addAttribute("axis", rewriter.getI64IntegerAttr(op.getAxis()));
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXConvOpLowering : public OpConversionPattern<mlir::ONNXConvOp> {
  using OpConversionPattern<mlir::ONNXConvOp>::OpConversionPattern;
  ONNXConvOpLowering(TypeConverter &tc, MLIRContext *ctx, unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXConvOp>(tc, ctx), nbits(nbits), es(es),
        storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXConvOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert conv result type");

    auto ops = adaptor.getOperands();
    if (ops.size() < 2)
      return rewriter.notifyMatchFailure(op, "conv expects at least X and W");
    Value x = ops[0];
    Value w = ops[1];
    Value b;
    if (ops.size() >= 3) {
      b = ops[2];
      if (b && llvm::isa<NoneType>(b.getType()))
        b = Value();
      // [FIX] ONNX bias may be represented as `none` (from onnx.NoValue / ONNXNoneOp).
      // If so, materialize a zero bias tensor.
    }
    if (!b) {
      int64_t c = ShapedType::kDynamic;
      Type elemTy;
      if (auto wTy = llvm::dyn_cast<RankedTensorType>(w.getType())) {
        if (wTy.getRank() >= 1) {
          c = wTy.getShape()[0];
          elemTy = wTy.getElementType();
        }
      }
      if ((c == ShapedType::kDynamic || !elemTy) &&
          llvm::isa<RankedTensorType>(outTy)) {
        auto outRtt = llvm::cast<RankedTensorType>(outTy);
        if (outRtt.getRank() >= 2) {
          c = outRtt.getShape()[1];
          elemTy = outRtt.getElementType();
        }
      }
      if (c == ShapedType::kDynamic || !elemTy)
        return rewriter.notifyMatchFailure(
            op, "cannot materialize default conv bias without known output channels");
      auto biasTy = RankedTensorType::get({c}, elemTy);
      b = buildZeroPositTensorConst(rewriter, loc, biasTy, storageBits);
      if (!b)
        return rewriter.notifyMatchFailure(op, "failed to build default bias constant");
    }

    OperationState st(loc, mlir::posit::Conv2DOp::getOperationName());
    st.addOperands({x, w, b});
    st.addTypes({outTy});

    // attrs: strides / pads / dilations / group / auto_pad (照 ONNX attr)
    if (auto a = op->getAttr("strides"))   st.addAttribute("strides", a);
    if (auto a = op->getAttr("pads"))      st.addAttribute("pads", a);
    if (auto a = op->getAttr("dilations")) st.addAttribute("dilations", a);
    if (auto a = op->getAttr("kernel_shape")) st.addAttribute("kernel_shape", a);
    if (auto a = op->getAttr("group"))     st.addAttribute("group", a);
    if (auto a = op->getAttr("auto_pad"))  st.addAttribute("auto_pad", a);
    st.addAttribute("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));

    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

struct ONNXMatMulOpLowering : public OpConversionPattern<mlir::ONNXMatMulOp> {
  using OpConversionPattern<mlir::ONNXMatMulOp>::OpConversionPattern;
  ONNXMatMulOpLowering(TypeConverter &tc, MLIRContext *ctx, unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXMatMulOp>(tc, ctx), nbits(nbits), es(es),
        storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert matmul result type");

    Value a = adaptor.getOperands()[0];
    Value b = adaptor.getOperands()[1];

    Type elemTy;
    if (auto outRtt = llvm::dyn_cast<RankedTensorType>(outTy))
      elemTy = outRtt.getElementType();
    else if (auto outUtt = llvm::dyn_cast<UnrankedTensorType>(outTy))
      elemTy = outUtt.getElementType();
    if (!elemTy)
      return rewriter.notifyMatchFailure(op, "cannot infer matmul output element type");
    auto cTy = RankedTensorType::get({1}, elemTy);
    Value c = buildZeroPositTensorConst(rewriter, loc, cTy, storageBits);
    if (!c)
      return rewriter.notifyMatchFailure(op, "failed to build GEMM C=0 constant");

    OperationState st(loc, mlir::posit::GemmOp::getOperationName());
    st.addOperands({a, b, c});
    st.addTypes({outTy});

    st.addAttribute("alpha", FloatAttr::get(rewriter.getF32Type(), 1.0));
    st.addAttribute("beta",  FloatAttr::get(rewriter.getF32Type(), 0.0));
    st.addAttribute("transA", rewriter.getI64IntegerAttr(0));
    st.addAttribute("transB", rewriter.getI64IntegerAttr(0));
    st.addAttribute("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));

    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

struct ONNXGemmOpLowering : public OpConversionPattern<mlir::ONNXGemmOp> {
  using OpConversionPattern<mlir::ONNXGemmOp>::OpConversionPattern;
  ONNXGemmOpLowering(TypeConverter &tc, MLIRContext *ctx, unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXGemmOp>(tc, ctx), nbits(nbits), es(es),
        storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXGemmOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert gemm result type");

    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    if (!c || llvm::isa<NoneType>(c.getType())) {
      Type elemTy;
      if (auto outRtt = llvm::dyn_cast<RankedTensorType>(outTy))
        elemTy = outRtt.getElementType();
      else if (auto outUtt = llvm::dyn_cast<UnrankedTensorType>(outTy))
        elemTy = outUtt.getElementType();
      if (!elemTy)
        return rewriter.notifyMatchFailure(op, "cannot infer gemm output element type");
      auto cTy = RankedTensorType::get({1}, elemTy);
      c = buildZeroPositTensorConst(rewriter, loc, cTy, storageBits);
      if (!c)
        return rewriter.notifyMatchFailure(op, "failed to build GEMM default C");
    }

    OperationState st(loc, mlir::posit::GemmOp::getOperationName());
    st.addOperands({a, b, c});
    st.addTypes({outTy});
    st.addAttribute(
        "alpha", FloatAttr::get(rewriter.getF32Type(), op.getAlpha().convertToFloat()));
    st.addAttribute(
        "beta", FloatAttr::get(rewriter.getF32Type(), op.getBeta().convertToFloat()));
    st.addAttribute("transA", rewriter.getI64IntegerAttr(op.getTransA()));
    st.addAttribute("transB", rewriter.getI64IntegerAttr(op.getTransB()));
    st.addAttribute("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }

  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

struct ONNXReduceMeanV13OpLowering
    : public OpConversionPattern<mlir::ONNXReduceMeanV13Op> {
  using OpConversionPattern<mlir::ONNXReduceMeanV13Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXReduceMeanV13Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert reduce_mean result type");

    SmallVector<int64_t, 4> axesVals;
    if (std::optional<ArrayAttr> axesAttr = op.getAxes()) {
      for (Attribute a : *axesAttr)
        axesVals.push_back(llvm::cast<IntegerAttr>(a).getValue().getSExtValue());
    }

    OperationState st(loc, mlir::posit::ReduceMeanOp::getOperationName());
    st.addOperands({adaptor.getData()});
    st.addTypes({outTy});
    st.addAttribute("axes", rewriter.getI64ArrayAttr(axesVals));
    st.addAttribute("keepdims", rewriter.getI64IntegerAttr(op.getKeepdims()));
    st.addAttribute("noop_with_empty_axes", rewriter.getI64IntegerAttr(0));
    st.addAttribute("qalign_key",
        rewriter.getI64IntegerAttr(computeTensorQAlignKey(op)));
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXConstantOpLowering
    : public OpConversionPattern<mlir::ONNXConstantOp> {
  using OpConversionPattern<mlir::ONNXConstantOp>::OpConversionPattern;
  ONNXConstantOpLowering(TypeConverter &tc, MLIRContext *ctx, unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXConstantOp>(tc, ctx), nbits(nbits), es(es),
        storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXConstantOp op,
                              typename OpConversionPattern::OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  Location loc = op.getLoc();

	  // 1. 抓出 "value" attribute。
	  // [FIX] ONNX Constant 的 value 可能是 dense<...> 或 dense_resource<...>。
	  // dense_resource 不是 DenseFPElementsAttr，必須用 ElementsAttr +
	  // DenseResourceElementsAttrBase<T>::tryGetAsArrayRef() 來取資料。
	  Attribute valueAttr = op->getAttr("value");
  if (!valueAttr)
    return rewriter.notifyMatchFailure(
        op, "ONNXConstantOp has no 'value' attribute (other forms not handled yet)");

	  // [FIX] 接受任何 ElementsAttr（DenseElementsAttr / DenseResourceElementsAttr / splat...）。
	  auto elements = llvm::dyn_cast<ElementsAttr>(valueAttr);
	  if (!elements)
	    return rewriter.notifyMatchFailure(
	        op, "only ElementsAttr (dense/dense_resource) is handled for ONNXConstantOp");

		  // 2. 只接受浮點 element（整數走 shape-constant 路徑）
	  auto elemType = elements.getElementType();
  // Integer constants (typically shape tensors) are kept in ONNX dialect and
  // lowered by the later ONNX->Krnl pass, so this pattern only handles floats.
  if (llvm::isa<IntegerType>(elemType))
    return rewriter.notifyMatchFailure(op, "integer ONNXConstant is kept as ONNX");

		  if (!llvm::isa<FloatType>(elemType))
	    return rewriter.notifyMatchFailure(
	      op, "only floating-point element type is handled");

  // Scalar float constants are kept as ONNXConstant (see conversion target
  // dynamic legality) and lowered later by ONNX->Krnl.
  if (elements.getNumElements() == 1)
    return rewriter.notifyMatchFailure(op, "scalar float ONNXConstant is kept as ONNX");

	  // 3. 目前先只處理 RankedTensor，且 ONNXConstant 的 result 也應該是 tensor
	  auto outType = op.getResult().getType();
	  auto origOutType = llvm::dyn_cast<RankedTensorType>(outType);
	  if (!origOutType)
    return rewriter.notifyMatchFailure(
          op, "only ranked tensor results are handled");

	  // 4. 用 TypeConverter 把 result type 轉成 tensor<...x!posit.type<nbits,es>>
	  Type convertedType = getTypeConverter()->convertType(origOutType);
	  if (!convertedType)
	    return rewriter.notifyMatchFailure(
	        op, "failed to convert constant result type to posit tensor");
  if (isPositConstDebugEnabled()) {
    llvm::errs() << "[posit-const] lowering ONNXConstant nbits=" << nbits
                 << " es=" << es << " origType=" << origOutType
                 << " convertedType=" << convertedType
                 << " elems=" << elements.getNumElements() << "\n";
  }

  // 5. For the nqdq/f32->posit path, optionally emit compact posit raw-bit
  //    constants at compile time. This makes large f32 weights occupy 1 byte
  //    for p8 or 2 bytes for p16 in .rodata. QDQ builds leave this disabled
  //    and keep the existing f32-constant + posit.from_f32 behavior.
  if (Value compact = createCompactPositConstantIfEnabled(
          rewriter, loc, op, origOutType, convertedType, elements, nbits, es,
          storageBits)) {
    rewriter.replaceOp(op, compact);
    return success();
  }

	  // 6. Default path: keep an ONNX f32 tensor constant, then convert with
	  //    posit.from_f32 at runtime.
	  auto f32ElemTy = rewriter.getF32Type();
	  auto f32TensorTy = RankedTensorType::get(origOutType.getShape(), f32ElemTy);
      Value f32Const;
	  if (elemType.isF64()) {
	    SmallVector<double, 4> fpVals;
	    if (failed(collectFPValuesAsDouble(elements, fpVals)))
	      return rewriter.notifyMatchFailure(
	          op, "failed to read f64 constant payload from dense/dense_resource");
	    SmallVector<Attribute, 8> f32Attrs;
	    f32Attrs.reserve(fpVals.size());
	    for (double d : fpVals)
	      f32Attrs.push_back(FloatAttr::get(f32ElemTy, static_cast<float>(d)));
	    DenseElementsAttr f32Elements = DenseElementsAttr::get(f32TensorTy, f32Attrs);
        f32Const = createONNXTensorConstant(rewriter, loc, f32TensorTy, f32Elements);
	  } else if (!elemType.isF32()) {
	    return rewriter.notifyMatchFailure(op, "only f32/f64 constant tensors are handled");
	  } else {
        f32Const = createONNXTensorConstant(rewriter, loc, f32TensorTy, elements);
	  }

	  Value out = rewriter.create<mlir::posit::FromF32Op>(loc, convertedType, f32Const).getResult();
	  rewriter.replaceOp(op, out);
	  return success();                           
	    
	  }
  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

} // end anonymous namespace

// 對外提供一個 helper，讓 ONNXToPosit.cpp 呼叫來註冊 patterns。  ONNXAddOpLowering, ONNXConstantOpLowering
void populateONNXToPositConversionPattern(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          MLIRContext *ctx,
                                          unsigned nbits,
                                          unsigned es,
                                          bool alignToInt8QDomain,
                                          bool preferDirectF32FromQDQ,
                                          bool preferDirectPositFromQDQ,
                                          bool preferStoreAsPosit) {
  patterns.add<ONNXDequantizeLinearOpLowering>(
      typeConverter, ctx, nbits, es, alignToInt8QDomain,
      preferDirectF32FromQDQ, preferDirectPositFromQDQ, preferStoreAsPosit);
  patterns.add<ONNXAddOpLowering>(typeConverter, ctx, nbits, es);
  patterns.add<ONNXSubOpLowering, ONNXMulOpLowering, ONNXDivOpLowering>(
      typeConverter, ctx);
  patterns.add<ONNXReshapeOpLowering, ONNXUnsqueezeOpLowering,
      ONNXReluOpLowering, ONNXClipOpLowering, ONNXMaxPoolSingleOutOpLowering,
      ONNXFlattenOpLowering, ONNXReduceMeanV13OpLowering>(typeConverter, ctx);
  patterns.add<ONNXConvOpLowering, ONNXMatMulOpLowering, ONNXGemmOpLowering>(
      typeConverter, ctx, nbits, es);
  // Required by the nqdq/f32 -> posit path. When ONNX_MLIR_POSIT_FORCE_NQDQ
  // is set, large floating ONNX constants become illegal and are rewritten into
  // compact posit raw-bit constants (p8/p16) or f32->posit boundaries.
  patterns.add<ONNXConstantOpLowering>(typeConverter, ctx, nbits, es);
}

} // namespace onnx_mlir
