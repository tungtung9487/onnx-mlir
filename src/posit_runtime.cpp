// [EXTEND][2026-02-23] Posit runtime (SoftPosit or Universal backend)
// - Keeps MLIR CRunner ABI (`UnrankedMemRefType` / `DynamicMemRefType`)
// - Provides generic kernels instantiated for posit formats used by the generated .ll
// - Enable Universal backend with: -DPOSIT_USE_UNIVERSAL and include path to universal headers

#if defined(POSIT_USE_UNIVERSAL)
#include <universal/utility/compiler.hpp>
#include <universal/utility/architecture.hpp>
#include <universal/utility/bit_cast.hpp>
#include <universal/utility/long_double.hpp>
#include <universal/traits/number_traits.hpp>
#include <universal/traits/arithmetic_traits.hpp>
#include <universal/common/number_traits_reports.hpp>
#include <universal/number/posit/posit.hpp>
#if __has_include(<universal/number/posit/quire.hpp>)
#include <universal/number/posit/quire.hpp>
#else
#include <universal/number/quire/quire.hpp>
#endif
#else
extern "C" {
#include "softposit.h"
}
#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
#include <universal/utility/compiler.hpp>
#include <universal/utility/architecture.hpp>
#include <universal/utility/bit_cast.hpp>
#include <universal/utility/long_double.hpp>
#include <universal/traits/number_traits.hpp>
#include <universal/traits/arithmetic_traits.hpp>
#include <universal/common/number_traits_reports.hpp>
#include <universal/number/posit/exceptions.hpp>
#include <universal/number/posit/posit_fwd.hpp>
#include <universal/number/posit/posit_impl.hpp>
#include <universal/traits/posit_traits.hpp>
#include <universal/number/posit/numeric_limits.hpp>
#endif
#endif

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <atomic>
#include <chrono>
#include <algorithm>
#include <array>
#include <random>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <fstream>
#include <inttypes.h>
#include <limits>
#include <mutex>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

static std::atomic<uint64_t> gPositMemrefGuardRejectCount{0};
static std::atomic<uint64_t> gPositMappedRangeRejectCount{0};
static std::atomic<uint64_t> gPositQuireFallbackCount{0};
static std::atomic<uint64_t> gPositOpAlignFallbackCount{0};
static std::atomic<uint64_t> gPositMixedP16FallbackCount{0};
static std::atomic<uint64_t> gPositFallbackLogSeq{0};

// Coarse per-stage wall-time profiler (POSIT_PROFILE=1). Splits a forward pass
// into: conv weight/input decode+decompand (the posit->value step, incl. ALPS
// sinh), conv MAC (the fma inner loops), and gemm. Zero overhead when off.
static inline bool positProfileEnabled() {
  static int e = -1;
  if (e < 0) {
    const char *v = std::getenv("POSIT_PROFILE");
    e = (v && *v && std::string(v) != "0" && std::string(v) != "off") ? 1 : 0;
  }
  return e == 1;
}
static inline long long positNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
static std::atomic<long long> gProfConvDecodeNs{0};
static std::atomic<long long> gProfConvMacNs{0};
static std::atomic<long long> gProfGemmNs{0};
struct PositProfileDump {
  ~PositProfileDump() {
    if (!positProfileEnabled())
      return;
    std::fprintf(stderr,
                 "[POSIT_PROFILE] conv_decode=%.3fs conv_mac=%.3fs gemm=%.3fs\n",
                 gProfConvDecodeNs.load() / 1e9, gProfConvMacNs.load() / 1e9,
                 gProfGemmNs.load() / 1e9);
  }
};
static PositProfileDump gPositProfileDump;

struct QAlignBucketKey {
  int64_t key;
  int64_t channel;
  bool operator==(const QAlignBucketKey &o) const {
    return key == o.key && channel == o.channel;
  }
};

struct QAlignBucketKeyHash {
  size_t operator()(const QAlignBucketKey &k) const {
    uint64_t a = static_cast<uint64_t>(k.key);
    uint64_t b = static_cast<uint64_t>(k.channel);
    uint64_t x = a * 0x9E3779B185EBCA87ULL ^ (b + 0x9E3779B97F4A7C15ULL);
    return static_cast<size_t>(x ^ (x >> 33));
  }
};

struct QAlignCollectBucket {
  uint64_t seenOrig = 0;
  uint64_t seenDQ = 0;
  std::vector<float> origSamples;
  std::vector<float> dqSamples;
};

static std::unordered_map<QAlignBucketKey, QAlignCollectBucket, QAlignBucketKeyHash>
    gQAlignCollectBuckets;
static std::mutex gQAlignCollectMutex;

enum : int {
  kQAlignProbeOrig = 0,
  kQAlignProbeDQ = 1,
  kQAlignProbeFullRef = 2,
};

struct QAlignProbeCount {
  uint64_t orig = 0;
  uint64_t dq = 0;
};

struct QAlignProbeRow {
  int sourceKind = kQAlignProbeOrig;
  std::string format;
  int64_t key = 0;
  int64_t channel = -1;
  uint64_t sampleIndex = 0;
  double theta = 1.0;
  double gamma = 0.0;
  int compandMode = 0;
  int qalignFound = 0;
  int strictFallbackApplied = 0;
  int gpEnabled = 0;
  int gpRs = 7;
  int gpSc = 0;
  double origX = std::numeric_limits<double>::quiet_NaN();
  double int8XDQ = std::numeric_limits<double>::quiet_NaN();
  double qValue = std::numeric_limits<double>::quiet_NaN();
  double scale = std::numeric_limits<double>::quiet_NaN();
  double zeroPoint = std::numeric_limits<double>::quiet_NaN();
  double runtimeScaled = std::numeric_limits<double>::quiet_NaN();
  double runtimeYQ = std::numeric_limits<double>::quiet_NaN();
  double runtimeXDQ = std::numeric_limits<double>::quiet_NaN();
  double runtimeFinal = std::numeric_limits<double>::quiet_NaN();
  std::string chosenPath;
  std::string qStoreKind;
  std::string dqOutputKind;
  std::string dqOutputWritten;
};

struct TensorGPMetadata {
  bool valid = false;
  bool enabled = false;
  int rs = 7;
  int sc = 0;
  int64_t qalignKey = 0;
  int compandMode = 0;
  double theta = 1.0;
  double gamma = 0.0;
};

struct TensorMetaChannelKey {
  const void *ptr = nullptr;
  int64_t channel = -1;

  bool operator==(const TensorMetaChannelKey &o) const {
    return ptr == o.ptr && channel == o.channel;
  }
};

struct TensorMetaChannelKeyHash {
  size_t operator()(const TensorMetaChannelKey &k) const {
    size_t h1 = std::hash<const void *>{}(k.ptr);
    size_t h2 = std::hash<int64_t>{}(k.channel);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct RuntimeOutputAlpsBucketKey {
  std::string format;
  int64_t key = 0;
  int64_t channel = -1;

  bool operator==(const RuntimeOutputAlpsBucketKey &o) const {
    return format == o.format && key == o.key && channel == o.channel;
  }
};

struct RuntimeOutputAlpsBucketKeyHash {
  size_t operator()(const RuntimeOutputAlpsBucketKey &k) const {
    size_t h1 = std::hash<std::string>{}(k.format);
    size_t h2 = std::hash<int64_t>{}(k.key);
    size_t h3 = std::hash<int64_t>{}(k.channel);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2)) ^
           (h3 + 0x9e3779b97f4a7c15ULL + (h2 << 6) + (h2 >> 2));
  }
};

struct RuntimeOutputAlpsCollectBucket {
  uint64_t seen = 0;
  std::vector<float> samples;
};

struct RuntimeOutputAlpsOfflineRow {
  TensorGPMetadata meta;
  double directScore = std::numeric_limits<double>::quiet_NaN();
  double chosenScore = std::numeric_limits<double>::quiet_NaN();
};

struct DotProbeRow {
  std::string op;
  std::string format;
  std::string effectiveMathPath;
  std::string outputEncodedFormat;
  int64_t key = 0;
  std::string dotMode;
  int useQuire = 0;
  int useF32Math = 0;
  int useP16Mixed = 0;
  int useGPMetadata = 0;
  int experimentalGPAllowed = 0;
  int64_t dim0 = 0;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t dim3 = 0;
  int64_t k = 0;
  int64_t outputElements = 0;
};

static std::vector<QAlignProbeRow> gQAlignProbeRows;
static std::unordered_map<QAlignBucketKey, QAlignProbeCount, QAlignBucketKeyHash>
    gQAlignProbeCounts;
static std::mutex gQAlignProbeMutex;
static std::vector<DotProbeRow> gDotProbeRows;
static std::mutex gDotProbeMutex;
static std::unordered_map<const void *, TensorGPMetadata> gTensorGPMeta;
static std::unordered_map<TensorMetaChannelKey, TensorGPMetadata,
                          TensorMetaChannelKeyHash>
    gTensorGPChannelMeta;
static std::unordered_set<const void *> gTensorGPSpecialChannelMetaPtrs;
static std::unordered_set<const void *> gTensorGPAlpsChannelMetaPtrs;
static std::mutex gTensorGPMetaMutex;
static std::unordered_map<RuntimeOutputAlpsBucketKey,
    RuntimeOutputAlpsCollectBucket, RuntimeOutputAlpsBucketKeyHash>
    gRuntimeOutputAlpsCollectBuckets;
static std::mutex gRuntimeOutputAlpsCollectMutex;

// --- Post-output clamp (POSIT_OUTPUT_CLAMP_FILE: "key,lo,hi[,theta[,gamma]]") ---
// Extended format: optional theta/gamma trigger Version-B re-encoding.
// If theta > 0: after clamping, re-encode with ALPS using new theta (not original metadata).
// If theta == 0 (default): re-encode using the tensor's existing ALPS/GP metadata (Version A).
struct PositOutputClampEntry { float lo, hi, theta = 0.f, gamma = 1.f; };
static std::unordered_map<int64_t, PositOutputClampEntry> gPositOutputClampTable;
static std::once_flag gPositOutputClampLoadFlag;

// --- Per-layer activation range collection (POSIT_COLLECT_LAYER_RANGES_FILE) ---
// Triggered by setting POSIT_COLLECT_LAYER_RANGES_FILE=<output_path>.
// During forward pass, decoded activation values are reservoir-sampled per qalignKey.
// At program exit, the RAW reservoir samples are written (one part file per
// process). Because the dataset runner uses one process per image, the bash
// wrapper passes a unique part path per image and a Python step merges all parts
// and computes percentiles. (Earlier this wrote percentiles directly, which was
// wrong under the parallel one-process-per-image runner: each process overwrote
// the same file, leaving only the last image. Now it writes mergeable raw data.)
// Part file format (one line per key):
//   key,op,total_seen,n_samples,v0,v1,...,v{n-1}
// Intended usage: run with POSIT_QOP_F32_MATH=on on a clean f32 model to collect
// uncontaminated activation distributions, then derive per-layer clamp ranges + ALPS theta.
static constexpr int64_t kLayerRangesReservoirSize = 8000;
struct PositLayerRangeReservoir {
  std::string   opLabel;
  std::vector<float> samples;   // reservoir, sorted at write time
  int64_t       totalSeen = 0;  // total values observed (for Vitter reservoir sampling)
  std::mt19937  rng{42};
};
static std::unordered_map<int64_t, PositLayerRangeReservoir> gLayerRangeTable;
static std::mutex   gLayerRangeMutex;
static std::once_flag gLayerRangeInitFlag;
static bool         gLayerRangeEnabled = false;
static std::string  gLayerRangeOutputPath;

static inline double qalignCompandAlps(double x, double beta);
static inline double qalignDecompandAlps(double y, double beta);

static inline bool tensorCompandIsAlps(const TensorGPMetadata &meta) {
  return meta.compandMode == 1 && std::isfinite(meta.theta) &&
         meta.theta > 0.0 && std::isfinite(meta.gamma) && meta.gamma > 0.0;
}

static inline bool tensorMathUsesF32ForAlps(const TensorGPMetadata &a) {
  return tensorCompandIsAlps(a);
}

static inline bool tensorMathUsesF32ForAlps(const TensorGPMetadata &a,
                                            const TensorGPMetadata &b) {
  return tensorCompandIsAlps(a) || tensorCompandIsAlps(b);
}

static inline bool tensorMathUsesF32ForAlps(const TensorGPMetadata &a,
                                            const TensorGPMetadata &b,
                                            const TensorGPMetadata &c) {
  return tensorCompandIsAlps(a) || tensorCompandIsAlps(b) ||
         tensorCompandIsAlps(c);
}

static inline bool tensorMathUsesF32ForAlps(const TensorGPMetadata &a,
                                            const TensorGPMetadata &b,
                                            const TensorGPMetadata &c,
                                            const TensorGPMetadata &d) {
  return tensorCompandIsAlps(a) || tensorCompandIsAlps(b) ||
         tensorCompandIsAlps(c) || tensorCompandIsAlps(d);
}

static inline bool tensorHasSpecialDecodeMetadata(const TensorGPMetadata &meta) {
  return meta.enabled || tensorCompandIsAlps(meta);
}

static inline bool positFallbackLogEnabled() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  const char *e = std::getenv("POSIT_FALLBACK_LOG");
  if (!e || !*e) {
    enabled = 1;
    return true;
  }
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  enabled = (s == "0" || s == "off" || s == "false" || s == "no") ? 0 : 1;
  return enabled == 1;
}

static inline uint64_t positFallbackLogLimit() {
  static uint64_t lim = 0;
  if (lim > 0)
    return lim;
  const char *e = std::getenv("POSIT_FALLBACK_LOG_LIMIT");
  if (!e || !*e) {
    lim = 20;
    return lim;
  }
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(e, &end, 10);
  if (!end || *end != '\0' || parsed == 0)
    lim = 20;
  else
    lim = static_cast<uint64_t>(parsed);
  return lim;
}

static inline bool positQuireAutoDisableEnabled() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  const char *e = std::getenv("POSIT_QUIRE_AUTO_DISABLE");
  if (!e || !*e) {
    enabled = 1;
    return true;
  }
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  enabled = (s == "0" || s == "off" || s == "false" || s == "no") ? 0 : 1;
  return enabled == 1;
}

// Operator-level epilogue alignment:
// - default ON: compute conv/gemm epilogue in double and quantize once.
// - OFF: keep legacy posit-domain epilogue behavior.
static inline bool positOpAlignEnabled() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  const char *e = std::getenv("POSIT_OP_ALIGN");
  if (!e || !*e) {
    enabled = 1;
    return true;
  }
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  enabled = (s == "0" || s == "off" || s == "false" || s == "no") ? 0 : 1;
  return enabled == 1;
}

// True p16e1 mixed-precision dot accumulation:
// - OFF (default): keep existing dot path (quire / posit-domain fallback).
// - ON: for low-precision formats (p4..p9), compute dot in p16e1 domain and
//       quantize once back to target format. Storage format remains unchanged.
static inline bool positMixedPrecisionP16Enabled() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  const char *e = std::getenv("POSIT_MIXED_PRECISION_P16");
  if (!e || !*e) {
    enabled = 0;
    return false;
  }
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  enabled = (s == "0" || s == "off" || s == "false" || s == "no") ? 0 : 1;
  return enabled == 1;
}

enum class PositBuildMixedMode {
  Runtime = 0,
  Off = 1,
  P16E2 = 2,
  P32E2 = 3,
};

static inline PositBuildMixedMode positBuildMixedMode() {
#if defined(POSIT_BUILD_MIXED_OFF)
  return PositBuildMixedMode::Off;
#elif defined(POSIT_BUILD_MIXED_P16E2)
  return PositBuildMixedMode::P16E2;
#elif defined(POSIT_BUILD_MIXED_P32E2)
  return PositBuildMixedMode::P32E2;
#else
  return PositBuildMixedMode::Runtime;
#endif
}

static inline const char *positBuildMixedModeName() {
  switch (positBuildMixedMode()) {
  case PositBuildMixedMode::Runtime:
    return "mixed_p16";
  case PositBuildMixedMode::Off:
    return "off";
  case PositBuildMixedMode::P16E2:
    return "mixed_p16e2";
  case PositBuildMixedMode::P32E2:
    return "mixed_p32e2";
  }
  return "mixed_p16";
}

enum class PositDotOpKind {
  Gemm,
  Conv2d,
};

static inline bool positCsvEnvHasToken(const char *envName, const char *token,
                                       bool defaultWhenUnset) {
  const char *e = std::getenv(envName);
  if (!e || !*e)
    return defaultWhenUnset;
  std::string csv(e);
  std::string tok(token);
  for (char &ch : csv)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  for (char &ch : tok)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  std::stringstream ss(csv);
  std::string item;
  auto trimLocal = [](const std::string &in) -> std::string {
    size_t b = 0;
    while (b < in.size() &&
           std::isspace(static_cast<unsigned char>(in[b])))
      ++b;
    size_t e2 = in.size();
    while (e2 > b &&
           std::isspace(static_cast<unsigned char>(in[e2 - 1])))
      --e2;
    return in.substr(b, e2 - b);
  };
  while (std::getline(ss, item, ',')) {
    item = trimLocal(item);
    if (item == tok)
      return true;
  }
  return false;
}

static inline bool positMixedPrecisionP16OpEnabled(PositDotOpKind opKind) {
  switch (opKind) {
  case PositDotOpKind::Gemm:
    return positCsvEnvHasToken("POSIT_MIXED_PRECISION_P16_OPS", "gemm", true);
  case PositDotOpKind::Conv2d:
    return positCsvEnvHasToken("POSIT_MIXED_PRECISION_P16_OPS", "conv2d", true);
  }
  return true;
}

// QOperator higher-precision compute path for posit runtime:
// - This runtime Conv/Gemm path is used for QOperator-style lowering such as
//   QLinearConv/QLinearMatMul. Ordinary QDQ patterns dequantize back to f32 and
//   then use the normal ONNX/Krnl f32 operators, so they do not call these
//   posit runtime Conv/Gemm kernels.
// - OFF (default): keep existing posit/quire (or mixed-p16) dot path.
// - ON: compute QOperator conv2d/gemm dot+epilogue in f32 domain, then quantize
//   once back to the target posit format.
// Preferred env names:
//   POSIT_QOP_F32_MATH=on|off
//   POSIT_QOP_F32_MATH_OPS=conv2d,gemm
// Backward-compatible aliases still accepted:
//   POSIT_QDQ_F32_MATH
//   POSIT_QDQ_F32_MATH_OPS
static inline bool positBoolEnvEnabled(const char *preferredName,
                                       const char *legacyName,
                                       bool defaultValue) {
  const char *e = std::getenv(preferredName);
  if ((!e || !*e) && legacyName)
    e = std::getenv(legacyName);
  if (!e || !*e)
    return defaultValue;
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return !(s == "0" || s == "off" || s == "false" || s == "no");
}

static inline bool positQopF32MathEnabled() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  enabled = positBoolEnvEnabled("POSIT_QOP_F32_MATH",
                                "POSIT_QDQ_F32_MATH",
                                /*defaultValue=*/false) ? 1 : 0;
  return enabled == 1;
}

// Method B (intra-op parallelism): number of threads for the output-element
// loops of posit conv/gemm. Default 1 => serial, bit-identical, and identical
// behavior to before (so existing per-image --jobs commands are unaffected and
// never oversubscribe). Set POSIT_OMP_THREADS>1 to parallelize a single forward
// pass. Only effective if the .so was built with -fopenmp; otherwise the omp
// pragmas are ignored and this value is moot. We use an explicit thread count
// (via the num_threads clause) instead of OMP_NUM_THREADS, whose default is
// "all cores" and would silently oversubscribe under process-level --jobs.
static inline int positOmpThreadCount() {
  static int n = -1;
  if (n >= 0)
    return n;
  const char *e = std::getenv("POSIT_OMP_THREADS");
  int v = 1;
  if (e && *e) {
    v = std::atoi(e);
    if (v < 1)
      v = 1;
  }
  n = v;
  return n;
}

static inline bool positCsvEnvHasTokenWithLegacy(const char *preferredName,
                                                 const char *legacyName,
                                                 const char *token,
                                                 bool defaultWhenUnset) {
  const char *e = std::getenv(preferredName);
  if (!e || !*e)
    e = legacyName ? std::getenv(legacyName) : nullptr;
  if (!e || !*e)
    return defaultWhenUnset;
  std::string csv(e);
  std::string tok(token);
  for (char &ch : csv)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  for (char &ch : tok)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  std::stringstream ss(csv);
  std::string item;
  auto trimLocal = [](const std::string &in) -> std::string {
    size_t b = 0;
    while (b < in.size() &&
           std::isspace(static_cast<unsigned char>(in[b])))
      ++b;
    size_t e2 = in.size();
    while (e2 > b &&
           std::isspace(static_cast<unsigned char>(in[e2 - 1])))
      --e2;
    return in.substr(b, e2 - b);
  };
  while (std::getline(ss, item, ',')) {
    item = trimLocal(item);
    if (item == tok)
      return true;
  }
  return false;
}

static inline bool positQopF32MathOpEnabled(PositDotOpKind opKind) {
  switch (opKind) {
  case PositDotOpKind::Gemm:
    return positCsvEnvHasTokenWithLegacy("POSIT_QOP_F32_MATH_OPS",
                                         "POSIT_QDQ_F32_MATH_OPS",
                                         "gemm", true);
  case PositDotOpKind::Conv2d:
    return positCsvEnvHasTokenWithLegacy("POSIT_QOP_F32_MATH_OPS",
                                         "POSIT_QDQ_F32_MATH_OPS",
                                         "conv2d", true);
  }
  return true;
}


static inline double positParsePositiveDoubleEnv(
    const char *name, double fallback) {
  const char *e = std::getenv(name);
  if (!e || !*e)
    return fallback;
  char *end = nullptr;
  double v = std::strtod(e, &end);
  if (!end || *end != '\0' || !std::isfinite(v) || v <= 0.0)
    return fallback;
  return v;
}

template <typename Fmt>
static inline double positQAlignAlpha();

static inline uint64_t positParsePositiveU64Env(const char *name,
                                                uint64_t fallback) {
  const char *e = std::getenv(name);
  if (!e || !*e)
    return fallback;
  char *end = nullptr;
  unsigned long long v = std::strtoull(e, &end, 10);
  if (!end || *end != '\0' || v == 0)
    return fallback;
  return static_cast<uint64_t>(v);
}

static inline double positParseDoubleEnvWithFallback(const char *name,
                                                     double fallback) {
  const char *e = std::getenv(name);
  if (!e || !*e)
    return fallback;
  char *end = nullptr;
  double v = std::strtod(e, &end);
  if (!end || *end != '\0' || !std::isfinite(v))
    return fallback;
  return v;
}

static inline bool positParseBoolEnvWithFallback(const char *name,
                                                 bool fallback) {
  const char *e = std::getenv(name);
  if (!e || !*e)
    return fallback;
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (s == "1" || s == "on" || s == "true" || s == "yes")
    return true;
  if (s == "0" || s == "off" || s == "false" || s == "no")
    return false;
  return fallback;
}

static inline std::string trimAscii(std::string s) {
  size_t b = 0;
  while (b < s.size() &&
         std::isspace(static_cast<unsigned char>(s[b])))
    ++b;
  size_t e = s.size();
  while (e > b &&
         std::isspace(static_cast<unsigned char>(s[e - 1])))
    --e;
  return s.substr(b, e - b);
}

static inline std::string positQAlignCollectFilePath() {
  static std::string path = []() {
    const char *e = std::getenv("POSIT_QALIGN_COLLECT_FILE");
    return (e && *e) ? std::string(e) : std::string();
  }();
  return path;
}

static inline bool positQAlignCollectEnabled() {
  return !positQAlignCollectFilePath().empty();
}

static inline std::string positRuntimeOutputAlpsCollectFilePath() {
  static std::string path = []() {
    const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_COLLECT_FILE");
    return (e && *e) ? std::string(e) : std::string();
  }();
  return path;
}

static inline bool positRuntimeOutputAlpsCollectEnabled() {
  return !positRuntimeOutputAlpsCollectFilePath().empty();
}

static inline std::string positRuntimeOutputAlpsOfflineFilePath() {
  static std::string path = []() {
    const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_FILE");
    return (e && *e) ? std::string(e) : std::string();
  }();
  return path;
}

static inline bool positRuntimeOutputAlpsOfflineEnabled() {
  return !positRuntimeOutputAlpsOfflineFilePath().empty();
}

static inline std::string positRuntimeOutputAlpsCalibFilePath() {
  static std::string path = []() {
    const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_CALIB_FILE");
    return (e && *e) ? std::string(e) : std::string();
  }();
  return path;
}

static inline bool positRuntimeOutputAlpsCalibEnabled() {
  return !positRuntimeOutputAlpsCalibFilePath().empty();
}

static inline std::string positQAlignProbeFilePath() {
  static std::string path = []() {
    const char *e = std::getenv("POSIT_QALIGN_PROBE_FILE");
    return (e && *e) ? std::string(e) : std::string();
  }();
  return path;
}

static inline std::string positQAlignProbeTextFilePath() {
  static std::string path = []() {
    const char *e = std::getenv("POSIT_QALIGN_PROBE_TEXT_FILE");
    return (e && *e) ? std::string(e) : std::string();
  }();
  return path;
}

static inline bool positQAlignProbeEnabled() {
  return !positQAlignProbeFilePath().empty();
}

static inline uint64_t positQAlignProbeLimitPerBucket() {
  static uint64_t v =
      positParsePositiveU64Env("POSIT_QALIGN_PROBE_LIMIT", 8);
  return v;
}

static inline const std::unordered_set<int64_t> &positQAlignProbeKeys() {
  static std::unordered_set<int64_t> keys = []() {
    std::unordered_set<int64_t> out;
    const char *e = std::getenv("POSIT_QALIGN_PROBE_KEYS");
    if (!e || !*e)
      return out;
    std::stringstream ss(e);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      tok = trimAscii(tok);
      if (tok.empty())
        continue;
      char *end = nullptr;
      int64_t key = std::strtoll(tok.c_str(), &end, 10);
      if (end && *end == '\0')
        out.insert(key);
    }
    return out;
  }();
  return keys;
}

static inline bool positQAlignProbeWantsKey(int64_t key) {
  const auto &keys = positQAlignProbeKeys();
  return keys.empty() || keys.count(key) != 0;
}

static inline int positQAlignProbeSourceMode() {
  static int mode = []() -> int {
    const char *e = std::getenv("POSIT_QALIGN_PROBE_SOURCE");
    if (!e || !*e)
      return 2;
    std::string s(e);
    for (char &ch : s)
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (s == "orig")
      return kQAlignProbeOrig;
    if (s == "dq")
      return kQAlignProbeDQ;
    return 2;
  }();
  return mode;
}

static inline bool positQAlignProbeWantsSource(int sourceKind) {
  int mode = positQAlignProbeSourceMode();
  return mode == 2 || mode == sourceKind;
}

static inline const char *qalignCompandModeName(int mode) {
  switch (mode) {
  case 1:
    return "alps";
  default:
    return "off";
  }
}

static inline const char *qalignProbeSourceName(int sourceKind) {
  if (sourceKind == kQAlignProbeDQ)
    return "dq";
  if (sourceKind == kQAlignProbeFullRef)
    return "fullref";
  return "orig";
}

static inline bool qalignProbeIsAlps(const QAlignProbeRow &r) {
  return r.compandMode == 1 && std::isfinite(r.gamma) && r.gamma > 0.0 &&
         !r.strictFallbackApplied;
}

static std::vector<QAlignProbeRow> buildFullReferenceProbeRows() {
  std::unordered_map<std::string, const QAlignProbeRow *> origRows;
  std::unordered_map<std::string, const QAlignProbeRow *> dqRows;
  auto makeKey = [](const QAlignProbeRow &r) {
    std::ostringstream os;
    os << r.format << ":" << r.key << ":" << r.channel << ":" << r.sampleIndex;
    return os.str();
  };
  for (const auto &r : gQAlignProbeRows) {
    if (r.sourceKind == kQAlignProbeOrig)
      origRows.emplace(makeKey(r), &r);
    else if (r.sourceKind == kQAlignProbeDQ)
      dqRows.emplace(makeKey(r), &r);
  }
  std::vector<QAlignProbeRow> out;
  out.reserve(std::min(origRows.size(), dqRows.size()));
  for (const auto &kv : origRows) {
    auto it = dqRows.find(kv.first);
    if (it == dqRows.end())
      continue;
    const QAlignProbeRow &orig = *kv.second;
    const QAlignProbeRow &dq = *it->second;
    QAlignProbeRow row = qalignProbeIsAlps(orig) ? orig : dq;
    row.sourceKind = kQAlignProbeFullRef;
    row.origX = orig.origX;
    row.int8XDQ = dq.int8XDQ;
    row.qValue = dq.qValue;
    row.scale = dq.scale;
    row.zeroPoint = dq.zeroPoint;
    row.qalignFound = (orig.qalignFound || dq.qalignFound) ? 1 : 0;
    if (qalignProbeIsAlps(orig)) {
      row.theta = orig.theta;
      row.gamma = orig.gamma;
      row.compandMode = orig.compandMode;
      row.strictFallbackApplied = orig.strictFallbackApplied;
      row.gpEnabled = orig.gpEnabled;
      row.gpRs = orig.gpRs;
      row.gpSc = orig.gpSc;
      row.runtimeScaled = orig.runtimeScaled;
      row.runtimeYQ = orig.runtimeYQ;
      row.runtimeXDQ = orig.runtimeXDQ;
      row.runtimeFinal = orig.runtimeFinal;
      row.chosenPath = "alps_direct_fullref";
      row.qStoreKind = "posit_y_q_storage";
      row.dqOutputKind = dq.dqOutputKind;
      row.dqOutputWritten = dq.dqOutputWritten;
    } else {
      row.theta = 1.0;
      row.gamma = 0.0;
      row.compandMode = 0;
      row.strictFallbackApplied = 1;
      row.gpEnabled = 0;
      row.gpRs = 7;
      row.gpSc = 0;
      row.runtimeScaled = dq.runtimeScaled;
      row.runtimeYQ = dq.runtimeYQ;
      row.runtimeXDQ = dq.runtimeXDQ;
      row.runtimeFinal = dq.runtimeFinal;
      row.chosenPath = "int8_qdq_fallback_fullref";
      row.qStoreKind = dq.qStoreKind;
      row.dqOutputKind = dq.dqOutputKind;
      row.dqOutputWritten = dq.dqOutputWritten;
    }
    out.push_back(std::move(row));
  }
  std::sort(out.begin(), out.end(), [](const QAlignProbeRow &a, const QAlignProbeRow &b) {
    if (a.key != b.key) return a.key < b.key;
    if (a.channel != b.channel) return a.channel < b.channel;
    return a.sampleIndex < b.sampleIndex;
  });
  return out;
}

static void dumpQAlignProbeTextAtExit(const std::string &outPath) {
  if (outPath.empty()) return;
  std::ofstream fout(outPath);
  if (!fout.is_open()) {
    std::fprintf(stderr, "[QALIGN] failed to open probe text file: %s\n", outPath.c_str());
    return;
  }
  auto emitMaybe = [&](double v) {
    if (!std::isfinite(v)) return std::string();
    std::ostringstream os; os << v; return os.str();
  };
  fout << "QAlign runtime probe\n";
  fout << "Notes:\n";
  fout << "  runtime_x_dq is the dequantized value produced by qalign/ALPS.\n";
  fout << "  runtime_final is a diagnostic value after re-encoding runtime_x_dq to the target posit/GP storage.\n";
  fout << "  In QDQ f32-output DQ kernels, dq_output_written should be runtime_x_dq, not runtime_final.\n";
  fout << "  fullref rows merge orig and dq samples with the same key/channel/sample_index.\n";
  fout << "  If orig_x and int8_x_dq are present, compare |runtime_x_dq-orig_x| against |int8_x_dq-orig_x|.\n\n";
  std::vector<QAlignProbeRow> fullRefRows = buildFullReferenceProbeRows();
  std::vector<const QAlignProbeRow *> rowsForText;
  rowsForText.reserve(gQAlignProbeRows.size() + fullRefRows.size());
  for (const auto &row : fullRefRows) rowsForText.push_back(&row);
  for (const auto &row : gQAlignProbeRows) rowsForText.push_back(&row);
  std::unordered_map<std::string, std::vector<const QAlignProbeRow *>> groups;
  for (const QAlignProbeRow *rowPtr : rowsForText) {
    const auto &row = *rowPtr;
    std::ostringstream key; key << row.key << ":" << row.channel << ":" << qalignProbeSourceName(row.sourceKind);
    groups[key.str()].push_back(rowPtr);
  }
  struct GroupSummary { const QAlignProbeRow *first = nullptr; std::vector<const QAlignProbeRow *> rows; };
  std::vector<GroupSummary> summaries; summaries.reserve(groups.size());
  for (auto &kv : groups) if (!kv.second.empty()) summaries.push_back(GroupSummary{kv.second.front(), kv.second});
  std::sort(summaries.begin(), summaries.end(), [](const GroupSummary &a, const GroupSummary &b) {
    if (a.first->key != b.first->key) return a.first->key < b.first->key;
    if (a.first->channel != b.first->channel) return a.first->channel < b.first->channel;
    return a.first->sourceKind < b.first->sourceKind;
  });
  uint64_t bucketLimit = positParsePositiveU64Env("POSIT_QALIGN_PROBE_TEXT_BUCKET_LIMIT", 8);
  uint64_t emittedBuckets = 0;
  for (const auto &g : summaries) {
    if (emittedBuckets >= bucketLimit) break;
    const QAlignProbeRow &f = *g.first;
    double maeXDQvsDQ=0, maeFinalVsDQ=0, maeInt8VsOrig=0, maeXDQVsOrig=0, maeFinalVsOrig=0;
    uint64_t nDQ=0, nOrig=0, betterXDQThanInt8=0, betterFinalThanInt8=0;
    for (const QAlignProbeRow *rp : g.rows) {
      const QAlignProbeRow &r = *rp;
      if (std::isfinite(r.int8XDQ) && std::isfinite(r.runtimeXDQ)) {
        maeXDQvsDQ += std::fabs(r.runtimeXDQ-r.int8XDQ);
        if (std::isfinite(r.runtimeFinal)) maeFinalVsDQ += std::fabs(r.runtimeFinal-r.int8XDQ);
        ++nDQ;
      }
      if (std::isfinite(r.origX) && std::isfinite(r.int8XDQ) && std::isfinite(r.runtimeXDQ)) {
        double eInt8=std::fabs(r.int8XDQ-r.origX), eXDQ=std::fabs(r.runtimeXDQ-r.origX);
        double eFinal=std::isfinite(r.runtimeFinal)?std::fabs(r.runtimeFinal-r.origX):std::numeric_limits<double>::quiet_NaN();
        maeInt8VsOrig += eInt8; maeXDQVsOrig += eXDQ;
        if (std::isfinite(eFinal)) maeFinalVsOrig += eFinal;
        if (eXDQ < eInt8) ++betterXDQThanInt8;
        if (std::isfinite(eFinal) && eFinal < eInt8) ++betterFinalThanInt8;
        ++nOrig;
      }
    }
    fout << "[bucket " << (emittedBuckets+1) << "] key=" << f.key
         << " channel=" << f.channel << " source=" << qalignProbeSourceName(f.sourceKind)
         << " format=" << f.format << " mode=" << qalignCompandModeName(f.compandMode)
         << " theta=" << f.theta << " gamma=" << f.gamma
         << " qalign_found=" << f.qalignFound << " strict_fallback=" << f.strictFallbackApplied
         << " gp_enabled=" << f.gpEnabled << " gp_rs=" << f.gpRs << " gp_sc=" << f.gpSc
         << " chosen_path=" << f.chosenPath
         << " q_store_kind=" << f.qStoreKind
         << " dq_output_kind=" << f.dqOutputKind
         << " dq_output_written=" << f.dqOutputWritten
         << " samples=" << g.rows.size() << "\n";
    if (nDQ) fout << "  metrics_vs_int8_dq: mae(runtime_x_dq-int8_x_dq)=" << (maeXDQvsDQ/nDQ)
                  << " mae(runtime_final-int8_x_dq)=" << (maeFinalVsDQ/nDQ) << "\n";
    if (nOrig) fout << "  metrics_vs_orig: mae(int8_x_dq-orig_x)=" << (maeInt8VsOrig/nOrig)
                    << " mae(runtime_x_dq-orig_x)=" << (maeXDQVsOrig/nOrig)
                    << " mae(runtime_final-orig_x)=" << (maeFinalVsOrig/nOrig)
                    << " better_xdq_than_int8_pct=" << (100.0*static_cast<double>(betterXDQThanInt8)/nOrig)
                    << " better_final_than_int8_pct=" << (100.0*static_cast<double>(betterFinalThanInt8)/nOrig) << "\n";
    else fout << "  metrics_vs_orig: unavailable (orig_x was not captured; use dual-reference collect)\n";
    uint64_t sampleNo=0;
    for (const QAlignProbeRow *rp : g.rows) {
      const QAlignProbeRow &r=*rp;
      fout << "  sample[" << sampleNo++ << "] orig_x=" << emitMaybe(r.origX)
           << " int8_x_dq=" << emitMaybe(r.int8XDQ) << " q_value=" << emitMaybe(r.qValue)
           << " scale=" << emitMaybe(r.scale) << " zp=" << emitMaybe(r.zeroPoint)
           << " runtime_scaled=" << emitMaybe(r.runtimeScaled) << " runtime_y_q=" << emitMaybe(r.runtimeYQ)
           << " runtime_x_dq=" << emitMaybe(r.runtimeXDQ) << " runtime_final=" << emitMaybe(r.runtimeFinal)
           << " chosen_path=" << r.chosenPath
           << " q_store_kind=" << r.qStoreKind
           << " dq_output_kind=" << r.dqOutputKind
           << " dq_output_written=" << r.dqOutputWritten << "\n";
    }
    fout << "\n"; ++emittedBuckets;
  }
  fout << "Summary:\n  emitted_buckets=" << emittedBuckets << "\n  total_probe_rows=" << gQAlignProbeRows.size() << "\n";
}

static void dumpQAlignProbeAtExit() {
  if (!positQAlignProbeEnabled())
    return;
  std::string outPath = positQAlignProbeFilePath();
  std::ofstream fout(outPath);
  if (!fout.is_open()) {
    std::fprintf(stderr, "[QALIGN] failed to open probe file: %s\n",
                 outPath.c_str());
    return;
  }
  fout << "source_kind,format,key,channel,sample_index,theta,gamma,mode,"
          "qalign_found,strict_fallback_applied,gp_enabled,gp_rs,gp_sc,"
          "chosen_path,q_store_kind,dq_output_kind,dq_output_written,"
          "orig_x,int8_x_dq,q_value,scale,zp,runtime_scaled,runtime_y_q,"
          "runtime_x_dq,runtime_final,xdq_minus_orig_x,xdq_minus_int8_x_dq,"
          "final_minus_orig_x,final_minus_int8_x_dq,"
          "abs_int8_minus_orig_x,abs_xdq_minus_orig_x,abs_final_minus_orig_x,"
          "xdq_better_than_int8,final_better_than_int8\n";
  auto emitMaybe = [&](double v) {
    if (!std::isfinite(v))
      return std::string();
    std::ostringstream os;
    os << v;
    return os.str();
  };
  std::lock_guard<std::mutex> lock(gQAlignProbeMutex);
  auto writeRow = [&](const QAlignProbeRow &row) {
    double xdqMinusOrig =
        (std::isfinite(row.runtimeXDQ) && std::isfinite(row.origX))
            ? (row.runtimeXDQ - row.origX)
            : std::numeric_limits<double>::quiet_NaN();
    double xdqMinusDQ =
        (std::isfinite(row.runtimeXDQ) && std::isfinite(row.int8XDQ))
            ? (row.runtimeXDQ - row.int8XDQ)
            : std::numeric_limits<double>::quiet_NaN();
    double finalMinusOrig =
        (std::isfinite(row.runtimeFinal) && std::isfinite(row.origX))
            ? (row.runtimeFinal - row.origX)
            : std::numeric_limits<double>::quiet_NaN();
    double finalMinusDQ =
        (std::isfinite(row.runtimeFinal) && std::isfinite(row.int8XDQ))
            ? (row.runtimeFinal - row.int8XDQ)
            : std::numeric_limits<double>::quiet_NaN();
    double absInt8MinusOrig =
        (std::isfinite(row.int8XDQ) && std::isfinite(row.origX))
            ? std::fabs(row.int8XDQ - row.origX)
            : std::numeric_limits<double>::quiet_NaN();
    double absXDQMinusOrig = std::isfinite(xdqMinusOrig)
                                 ? std::fabs(xdqMinusOrig)
                                 : std::numeric_limits<double>::quiet_NaN();
    double absFinalMinusOrig = std::isfinite(finalMinusOrig)
                                   ? std::fabs(finalMinusOrig)
                                   : std::numeric_limits<double>::quiet_NaN();
    int xdqBetter = (std::isfinite(absInt8MinusOrig) &&
                     std::isfinite(absXDQMinusOrig) &&
                     absXDQMinusOrig < absInt8MinusOrig) ? 1 : 0;
    int finalBetter = (std::isfinite(absInt8MinusOrig) &&
                       std::isfinite(absFinalMinusOrig) &&
                       absFinalMinusOrig < absInt8MinusOrig) ? 1 : 0;
    fout << qalignProbeSourceName(row.sourceKind) << ","
         << row.format << "," << row.key << "," << row.channel << ","
         << row.sampleIndex << "," << row.theta << "," << row.gamma << ","
         << qalignCompandModeName(row.compandMode) << ","
         << row.qalignFound << "," << row.strictFallbackApplied << ","
         << row.gpEnabled << "," << row.gpRs << "," << row.gpSc << ","
         << row.chosenPath << "," << row.qStoreKind << ","
         << row.dqOutputKind << "," << row.dqOutputWritten << ","
         << emitMaybe(row.origX) << "," << emitMaybe(row.int8XDQ) << ","
         << emitMaybe(row.qValue) << "," << emitMaybe(row.scale) << ","
         << emitMaybe(row.zeroPoint) << "," << emitMaybe(row.runtimeScaled)
         << "," << emitMaybe(row.runtimeYQ) << ","
         << emitMaybe(row.runtimeXDQ) << ","
         << emitMaybe(row.runtimeFinal) << ","
         << emitMaybe(xdqMinusOrig) << ","
         << emitMaybe(xdqMinusDQ) << ","
         << emitMaybe(finalMinusOrig) << ","
         << emitMaybe(finalMinusDQ) << ","
         << emitMaybe(absInt8MinusOrig) << ","
         << emitMaybe(absXDQMinusOrig) << ","
         << emitMaybe(absFinalMinusOrig) << ","
         << xdqBetter << "," << finalBetter << "\n";
  };
  for (const auto &row : gQAlignProbeRows)
    writeRow(row);
  std::vector<QAlignProbeRow> fullRefRows = buildFullReferenceProbeRows();
  for (const auto &row : fullRefRows)
    writeRow(row);
  fout.flush();
  std::string textPath = positQAlignProbeTextFilePath();
  if (!textPath.empty()) dumpQAlignProbeTextAtExit(textPath);
}

static inline void ensureQAlignProbeRegistration() {
  static bool registered = false;
  if (registered)
    return;
  if (positQAlignProbeEnabled()) {
    std::atexit(dumpQAlignProbeAtExit);
    registered = true;
  }
}

static inline uint64_t positQAlignCollectMaxPerBucket() {
  static uint64_t v =
      positParsePositiveU64Env("POSIT_QALIGN_COLLECT_MAX_PER_BUCKET", 1024);
  return v;
}

static inline uint64_t positQAlignCollectPerCall() {
  static uint64_t v =
      positParsePositiveU64Env("POSIT_QALIGN_COLLECT_PER_CALL", 512);
  return v;
}

static inline uint64_t positRuntimeOutputAlpsCollectMaxPerKey() {
  static uint64_t v = positParsePositiveU64Env(
      "POSIT_RUNTIME_OUTPUT_ALPS_COLLECT_MAX_PER_KEY", 4096);
  return v;
}

static inline uint64_t positRuntimeOutputAlpsCollectPerCall() {
  static uint64_t v = positParsePositiveU64Env(
      "POSIT_RUNTIME_OUTPUT_ALPS_COLLECT_PER_CALL", 512);
  return v;
}

static void dumpRuntimeOutputAlpsCollectAtExit() {
  if (!positRuntimeOutputAlpsCollectEnabled())
    return;
  std::string outPath = positRuntimeOutputAlpsCollectFilePath();
  std::ofstream fout(outPath);
  if (!fout.is_open()) {
    std::fprintf(stderr,
        "[runtime-output-alps] failed to open collect file: %s\n",
        outPath.c_str());
    return;
  }
  fout << "format,key,channel,seen,sample_count,samples\n";
  std::lock_guard<std::mutex> lock(gRuntimeOutputAlpsCollectMutex);
  for (const auto &it : gRuntimeOutputAlpsCollectBuckets) {
    const auto &k = it.first;
    const auto &b = it.second;
    fout << k.format << "," << k.key << "," << k.channel << "," << b.seen
         << "," << b.samples.size() << ",";
    for (size_t i = 0; i < b.samples.size(); ++i) {
      if (i)
        fout << ";";
      fout << b.samples[i];
    }
    fout << "\n";
  }
  fout.flush();
}

static inline void ensureRuntimeOutputAlpsCollectRegistration() {
  static bool registered = false;
  if (registered)
    return;
  if (positRuntimeOutputAlpsCollectEnabled()) {
    std::atexit(dumpRuntimeOutputAlpsCollectAtExit);
    registered = true;
  }
}

enum : int {
  kQAlignCollectOrig = 0,
  kQAlignCollectDQ = 1,
};

static void dumpQAlignCollectAtExit() {
  if (!positQAlignCollectEnabled())
    return;
  std::string outPath = positQAlignCollectFilePath();
  std::ofstream fout(outPath);
  if (!fout.is_open()) {
    std::fprintf(stderr, "[QALIGN] failed to open collect file: %s\n",
                 outPath.c_str());
    return;
  }
  fout << "key,channel,seen_orig,sample_count_orig,orig_samples,seen_dq,sample_count_dq,dq_samples\n";
  std::lock_guard<std::mutex> lock(gQAlignCollectMutex);
  for (const auto &it : gQAlignCollectBuckets) {
    const auto &k = it.first;
    const auto &b = it.second;
    fout << k.key << "," << k.channel << "," << b.seenOrig << ","
         << b.origSamples.size() << ",";
    for (size_t i = 0; i < b.origSamples.size(); ++i) {
      if (i)
        fout << ";";
      fout << b.origSamples[i];
    }
    fout << "," << b.seenDQ << "," << b.dqSamples.size() << ",";
    for (size_t i = 0; i < b.dqSamples.size(); ++i) {
      if (i)
        fout << ";";
      fout << b.dqSamples[i];
    }
    fout << "\n";
  }
  fout.flush();
}

static inline void ensureQAlignCollectRegistration() {
  static bool registered = false;
  if (registered)
    return;
  if (positQAlignCollectEnabled()) {
    std::atexit(dumpQAlignCollectAtExit);
    registered = true;
  }
}

static inline void collectQAlignBatch(int collectKind, int64_t qalignKey,
                                      const std::vector<int64_t> &channels,
                                      const std::vector<double> &values) {
  if (!positQAlignCollectEnabled())
    return;
  ensureQAlignCollectRegistration();
  if (channels.empty() || values.empty() || channels.size() != values.size())
    return;
  uint64_t cap = positQAlignCollectMaxPerBucket();
  std::lock_guard<std::mutex> lock(gQAlignCollectMutex);
  for (size_t i = 0; i < values.size(); ++i) {
    QAlignBucketKey k{qalignKey, channels[i]};
    auto &bucket = gQAlignCollectBuckets[k];
    if (collectKind == kQAlignCollectDQ) {
      bucket.seenDQ++;
      if (bucket.dqSamples.size() < cap)
        bucket.dqSamples.push_back(static_cast<float>(values[i]));
    } else {
      bucket.seenOrig++;
      if (bucket.origSamples.size() < cap)
        bucket.origSamples.push_back(static_cast<float>(values[i]));
    }
  }
}

struct QAlignOutputSampleBuffer {
  bool enabled = false;
  int64_t qalignKey = 0;
  uint64_t budget = 0;
  int64_t stride = 1;
  std::vector<int64_t> channels;
  std::vector<double> orig;
  std::vector<double> dq;

  void maybeAppend(int64_t lin, int64_t channel, double origV, double dqV) {
    if (!enabled || budget == 0)
      return;
    if (orig.size() >= static_cast<size_t>(budget))
      return;
    if ((lin % stride) != 0)
      return;
    channels.push_back(channel);
    orig.push_back(origV);
    dq.push_back(dqV);
  }

  void flush() const {
    if (!enabled || qalignKey == 0 || orig.empty() || dq.empty())
      return;
    collectQAlignBatch(kQAlignCollectOrig, qalignKey, channels, orig);
    collectQAlignBatch(kQAlignCollectDQ, qalignKey, channels, dq);
  }
};

static inline QAlignOutputSampleBuffer makeQAlignOutputSampleBuffer(
    int64_t qalignKey, int64_t totalElems) {
  QAlignOutputSampleBuffer buf;
  buf.enabled = positQAlignCollectEnabled() && qalignKey != 0 && totalElems > 0;
  buf.qalignKey = qalignKey;
  if (!buf.enabled)
    return buf;
  buf.budget = positQAlignCollectPerCall();
  if (buf.budget == 0)
    return buf;
  if (totalElems > static_cast<int64_t>(buf.budget))
    buf.stride = std::max<int64_t>(
        1, totalElems / static_cast<int64_t>(buf.budget));
  buf.channels.reserve(static_cast<size_t>(buf.budget));
  buf.orig.reserve(static_cast<size_t>(buf.budget));
  buf.dq.reserve(static_cast<size_t>(buf.budget));
  return buf;
}

enum : int {
  kQAlignCompandOff = 0,
  kQAlignCompandAlps = 1,
};

struct QAlignParams {
  double alpha = 1.0;
  double beta = 0.0;
  int compandMode = kQAlignCompandOff;
  int gpRs = 7;
  int gpSc = 0;
  bool hasBeta = false;
  bool hasMode = false;
  bool hasGp = false;
  bool foundInCsv = false;
  bool strictFallbackApplied = false;
};

static inline int parseQAlignCompandModeToken(
    const std::string &s, int fallback = kQAlignCompandOff) {
  std::string t = trimAscii(s);
  if (t.empty())
    return fallback;
  std::string lower = t;
  for (char &ch : lower)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (lower == "alps" || lower == "1" || lower == "on" || lower == "true")
    return kQAlignCompandAlps;
  if (lower == "off" || lower == "0" || lower == "false")
    return kQAlignCompandOff;
  return fallback;
}

struct QAlignParamTable {
  bool hasFile = false;
  std::unordered_map<QAlignBucketKey, QAlignParams, QAlignBucketKeyHash> params;
  std::unordered_map<int64_t, QAlignParams> keyDefaults;
};

static const QAlignParamTable &getQAlignParamTable() {
  static QAlignParamTable table = []() {
    QAlignParamTable t;
    const char *e = std::getenv("POSIT_QALIGN_FILE");
    if (!e || !*e)
      return t;
    std::ifstream fin(e);
    if (!fin.is_open()) {
      std::fprintf(stderr, "[QALIGN] cannot open POSIT_QALIGN_FILE=%s\n", e);
      return t;
    }
    t.hasFile = true;
    std::unordered_map<int64_t,
        std::unordered_map<int64_t, uint64_t>> gpPairCounts;
    std::string line;
    while (std::getline(fin, line)) {
      line = trimAscii(line);
      if (line.empty() || line[0] == '#')
        continue;
      std::vector<std::string> cols;
      {
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ','))
          cols.push_back(trimAscii(tok));
      }
      if (cols.size() < 3)
        continue;
      const std::string &sKey = cols[0];
      const std::string &sCh = cols[1];
      const std::string &sAlpha = cols[2];
      char *end = nullptr;
      int64_t key = std::strtoll(sKey.c_str(), &end, 10);
      if (!end || *end != '\0')
        continue;
      end = nullptr;
      int64_t ch = std::strtoll(sCh.c_str(), &end, 10);
      if (!end || *end != '\0')
        continue;
      end = nullptr;
      double alpha = std::strtod(sAlpha.c_str(), &end);
      if (!end || *end != '\0' || !std::isfinite(alpha) || alpha <= 0.0)
        continue;
      QAlignParams p;
      p.alpha = alpha;
      if (cols.size() >= 4) {
        end = nullptr;
        double beta = std::strtod(cols[3].c_str(), &end);
        if (end && *end == '\0' && std::isfinite(beta) && beta >= 0.0) {
          p.beta = beta;
          p.hasBeta = true;
        } else {
          p.compandMode = parseQAlignCompandModeToken(cols[3], p.compandMode);
          p.hasMode = true;
        }
      }
      if (cols.size() >= 5) {
        p.compandMode = parseQAlignCompandModeToken(cols[4], p.compandMode);
        p.hasMode = true;
      }
      if (cols.size() >= 7) {
        end = nullptr;
        long gpRs = std::strtol(cols[5].c_str(), &end, 10);
        bool rsOk = end && *end == '\0';
        end = nullptr;
        long gpSc = std::strtol(cols[6].c_str(), &end, 10);
        bool scOk = end && *end == '\0';
        if (rsOk && scOk) {
          p.gpRs = std::min(7L, std::max(1L, gpRs));
          p.gpSc = std::min(16L, std::max(-16L, gpSc));
          p.hasGp = true;
        }
      }
      // Safe fallback rule: when qalign mode is off, generalized-posit metadata
      // must also be disabled. This makes "ALPS not selected" behave like the
      // original strict/standard posit path instead of continuing to use rs/sc.
      std::string gpSourceToken = (cols.size() >= 8) ? cols[7] : std::string();
      std::string gpSourceLower = gpSourceToken;
      for (char &ch2 : gpSourceLower)
        ch2 = static_cast<char>(std::tolower(static_cast<unsigned char>(ch2)));
      if (p.compandMode != kQAlignCompandAlps ||
          gpSourceLower.find("disabled") != std::string::npos ||
          (p.gpRs == 7 && p.gpSc == 0 && gpSourceLower.find("force") == std::string::npos)) {
        p.hasGp = false;
        p.gpRs = 7;
        p.gpSc = 0;
      }
      t.params[QAlignBucketKey{key, ch}] = p;
      if (p.hasGp) {
        int64_t gpPairKey =
            (static_cast<int64_t>(p.gpRs) << 32) ^
            static_cast<uint32_t>(p.gpSc - std::numeric_limits<int16_t>::min());
        gpPairCounts[key][gpPairKey]++;
      }
    }
    for (const auto &it : gpPairCounts) {
      int64_t bestPairKey = 0;
      uint64_t bestCount = 0;
      for (const auto &countIt : it.second) {
        if (countIt.second > bestCount) {
          bestCount = countIt.second;
          bestPairKey = countIt.first;
        }
      }
      if (bestCount == 0)
        continue;
      QAlignParams p;
      p.hasGp = true;
      p.gpRs = static_cast<int>(bestPairKey >> 32);
      p.gpSc = static_cast<int>(
          static_cast<int32_t>(bestPairKey & 0xffffffffULL) +
          std::numeric_limits<int16_t>::min());
      t.keyDefaults[it.first] = p;
    }
    return t;
  }();
  return table;
}

static inline bool positDescDebugEnabled() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  const char *e = std::getenv("POSIT_DESC_DEBUG");
  if (!e || !*e) {
    enabled = 0;
    return false;
  }
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  enabled = (s == "0" || s == "off" || s == "false" || s == "no") ? 0 : 1;
  return enabled == 1;
}

static void printPositFallbackSummaryAtExit() {
  uint64_t memref = gPositMemrefGuardRejectCount.load(std::memory_order_relaxed);
  uint64_t mapped = gPositMappedRangeRejectCount.load(std::memory_order_relaxed);
  uint64_t quire = gPositQuireFallbackCount.load(std::memory_order_relaxed);
  uint64_t opalign = gPositOpAlignFallbackCount.load(std::memory_order_relaxed);
  if (memref == 0 && mapped == 0 && quire == 0 && opalign == 0)
    return;
  std::fprintf(stderr,
               "[PFALLBACK_SUMMARY] memref_guard=%llu mapped_range=%llu quire=%llu op_align=%llu\n",
               static_cast<unsigned long long>(memref),
               static_cast<unsigned long long>(mapped),
               static_cast<unsigned long long>(quire),
               static_cast<unsigned long long>(opalign));
}

static inline void ensurePositFallbackSummaryRegistration() {
  static bool registered = false;
  if (registered)
    return;
  std::atexit(printPositFallbackSummaryAtExit);
  registered = true;
}

static inline void recordPositFallback(std::atomic<uint64_t> &counter,
                                       const char *kind, const char *detail) {
  ensurePositFallbackSummaryRegistration();
  uint64_t count = counter.fetch_add(1, std::memory_order_relaxed) + 1;
  if (!positFallbackLogEnabled())
    return;
  uint64_t seq = gPositFallbackLogSeq.fetch_add(1, std::memory_order_relaxed) + 1;
  if (seq > positFallbackLogLimit())
    return;
  if (detail && *detail) {
    std::fprintf(stderr, "[PFALLBACK] kind=%s count=%llu detail=%s\n", kind,
                 static_cast<unsigned long long>(count), detail);
  } else {
    std::fprintf(stderr, "[PFALLBACK] kind=%s count=%llu\n", kind,
                 static_cast<unsigned long long>(count));
  }
}

template <typename T>
static inline void dumpMemRefDesc(const char *tag, const DynamicMemRefType<T> &m);

template <typename T>
static inline int64_t num_elems(const DynamicMemRefType<T> &m) {
  int64_t n = 1;
  for (int i = 0; i < m.rank; ++i)
    n *= m.sizes[i];
  return n;
}

template <typename T>
static inline bool validate_memref_desc(const DynamicMemRefType<T> &m,
                                        int64_t maxRank = 8) {
  if (m.rank < 0 || m.rank > maxRank) {
    recordPositFallback(gPositMemrefGuardRejectCount, "memref_guard",
                        "invalid rank");
    if (positDescDebugEnabled())
      dumpMemRefDesc("memref_guard.invalid_rank", m);
    return false;
  }
  if (m.offset < 0 || m.offset > (1LL << 40)) {
    recordPositFallback(gPositMemrefGuardRejectCount, "memref_guard",
                        "offset out of range");
    if (positDescDebugEnabled())
      dumpMemRefDesc("memref_guard.bad_offset", m);
    return false;
  }
  for (int i = 0; i < m.rank; ++i) {
    if (m.sizes[i] < 0) {
      recordPositFallback(gPositMemrefGuardRejectCount, "memref_guard",
                          "negative size");
      if (positDescDebugEnabled())
        dumpMemRefDesc("memref_guard.negative_size", m);
      return false;
    }
    // Posit kernels here assume non-negative, forward strides.
    if (m.strides[i] < 0) {
      recordPositFallback(gPositMemrefGuardRejectCount, "memref_guard",
                          "negative stride");
      if (positDescDebugEnabled())
        dumpMemRefDesc("memref_guard.negative_stride", m);
      return false;
    }
    if (m.sizes[i] > (1LL << 30)) {
      recordPositFallback(gPositMemrefGuardRejectCount, "memref_guard",
                          "size too large");
      if (positDescDebugEnabled())
        dumpMemRefDesc("memref_guard.size_too_large", m);
      return false;
    }
    if (m.strides[i] > (1LL << 40)) {
      recordPositFallback(gPositMemrefGuardRejectCount, "memref_guard",
                          "stride too large");
      if (positDescDebugEnabled())
        dumpMemRefDesc("memref_guard.stride_too_large", m);
      return false;
    }
  }
  int64_t elems = 0;
  if (!num_elems_safe(m, elems)) {
    recordPositFallback(gPositMemrefGuardRejectCount, "memref_guard",
                        "invalid element count");
    if (positDescDebugEnabled())
      dumpMemRefDesc("memref_guard.invalid_elems", m);
    return false;
  }
  if (elems == 0)
    return true;
  if (!m.data) {
    recordPositFallback(gPositMemrefGuardRejectCount, "memref_guard",
                        "null data pointer");
    if (positDescDebugEnabled())
      dumpMemRefDesc("memref_guard.null_data", m);
    return false;
  }
  return true;
}

template <typename T>
static inline bool num_elems_safe(const DynamicMemRefType<T> &m, int64_t &out) {
  if (m.rank < 0)
    return false;
  int64_t n = 1;
  for (int i = 0; i < m.rank; ++i) {
    int64_t d = m.sizes[i];
    if (d < 0)
      return false;
    if (d == 0) {
      out = 0;
      return true;
    }
    if (n > std::numeric_limits<int64_t>::max() / d)
      return false;
    n *= d;
  }
  out = n;
  return true;
}

template <typename T>
static inline bool is_dense_contiguous_memref(const DynamicMemRefType<T> &m) {
  if (m.rank == 0)
    return true;
  int64_t expected = 1;
  for (int i = m.rank - 1; i >= 0; --i) {
    int64_t sz = m.sizes[i];
    if (sz < 0)
      return false;
    if (sz <= 1)
      continue;
    if (m.strides[i] != expected)
      return false;
    if (expected > std::numeric_limits<int64_t>::max() / sz)
      return false;
    expected *= sz;
  }
  return true;
}

static inline bool is_mapped_range(const void *ptr, size_t bytes) {
  if (!ptr || bytes == 0)
    return bytes == 0;
  uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t end = start + bytes - 1;
  if (end < start)
    return false;

#if defined(__linux__)
  long pageSizeLong = ::sysconf(_SC_PAGESIZE);
  if (pageSizeLong > 0) {
    uintptr_t pageSize = static_cast<uintptr_t>(pageSizeLong);
    uintptr_t pageStart = start & ~(pageSize - 1);
    uintptr_t pageEnd = end & ~(pageSize - 1);
    if (pageEnd < pageStart)
      return false;
    size_t pageCount = static_cast<size_t>((pageEnd - pageStart) / pageSize + 1);
    std::vector<unsigned char> vec(pageCount, 0);
    errno = 0;
    if (::mincore(reinterpret_cast<void *>(pageStart), pageCount * pageSize, vec.data()) == 0)
      return true;
    if (errno == ENOMEM)
      return false;
    // For non-ENOMEM failures, fall back to /proc/self/maps scan below.
  }
#endif

  FILE *maps = std::fopen("/proc/self/maps", "r");
  if (!maps)
    return true;

  uintptr_t covered = start;
  char line[512];
  while (std::fgets(line, sizeof(line), maps)) {
    uintptr_t lo = 0, hi = 0;
    if (std::sscanf(line, "%" SCNxPTR "-%" SCNxPTR, &lo, &hi) != 2)
      continue;
    if (covered < lo)
      continue;
    if (covered >= lo && covered < hi) {
      if (end < hi) {
        std::fclose(maps);
        return true;
      }
      covered = hi;
    }
  }
  std::fclose(maps);
  return false;
}

static inline bool positMappedGuardDisabled() {
  static int v = -1;
  if (v >= 0)
    return v == 1;
  const char *e = std::getenv("POSIT_DISABLE_MAPPED_GUARD");
  v = (e && *e && !(std::string(e) == "0" || std::string(e) == "off")) ? 1 : 0;
  return v == 1;
}

template <typename T>
static inline bool has_mapped_dense_storage(const DynamicMemRefType<T> &m) {
  int64_t elems = 0;
  if (!num_elems_safe(m, elems)) {
    recordPositFallback(gPositMappedRangeRejectCount, "mapped_range",
                        "num_elems overflow");
    return false;
  }
  if (elems == 0)
    return true;
  // Diagnostic/escape hatch: skip the mincore range check (treat any non-null
  // data as mapped). If GPT-2 runs correctly with this set, the guard was
  // falsely rejecting valid buffers; if it crashes, the descriptor is genuinely
  // over-large (a real shape/lowering bug).
  if (positMappedGuardDisabled())
    return m.data != nullptr;
  // Compute the ACTUAL element extent from strides instead of assuming a dense
  // [offset, offset+elems) block. Broadcast operands (stride-0 dims, e.g. a GPT-2
  // gemm bias broadcast to [N,M,S] while only N elements are stored) and strided
  // views have far fewer real elements than the logical product of sizes. The old
  // dense assumption made byteCount overshoot the real allocation, so the range
  // was wrongly reported "not mapped" and the op output was zeroed. For a dense
  // contiguous tensor this yields the same [offset, offset+elems) span as before.
  __int128 minRel = 0, maxRel = 0;
  for (int i = 0; i < m.rank; ++i) {
    if (m.sizes[i] <= 1)
      continue;
    __int128 step = static_cast<__int128>(m.sizes[i] - 1) *
                    static_cast<__int128>(m.strides[i]);
    if (step >= 0)
      maxRel += step;
    else
      minRel += step;
  }
  __int128 first = static_cast<__int128>(m.offset) + minRel;
  __int128 last = static_cast<__int128>(m.offset) + maxRel;
  if (first < 0 || last < first ||
      last > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
    recordPositFallback(gPositMappedRangeRejectCount, "mapped_range",
                        "element extent out of range");
    return false;
  }
  __int128 startBytes = first * static_cast<__int128>(sizeof(T));
  __int128 byteCount = (last - first + 1) * static_cast<__int128>(sizeof(T));
  if (startBytes < 0 || startBytes > static_cast<__int128>(std::numeric_limits<size_t>::max())) {
    recordPositFallback(gPositMappedRangeRejectCount, "mapped_range",
                        "start byte offset out of range");
    return false;
  }
  if (byteCount <= 0 || byteCount > static_cast<__int128>(std::numeric_limits<size_t>::max())) {
    recordPositFallback(gPositMappedRangeRejectCount, "mapped_range",
                        "byte size overflow");
    return false;
  }
  uintptr_t dataAddr = reinterpret_cast<uintptr_t>(m.data);
  uintptr_t startAddr = dataAddr + static_cast<size_t>(startBytes);
  if (startAddr < dataAddr) {
    recordPositFallback(gPositMappedRangeRejectCount, "mapped_range",
                        "start address overflow");
    return false;
  }
  bool ok = is_mapped_range(reinterpret_cast<const void *>(startAddr),
                            static_cast<size_t>(byteCount));
  if (!ok) {
    recordPositFallback(gPositMappedRangeRejectCount, "mapped_range",
                        "pointer range not mapped");
  }
  return ok;
}

template <typename T>
static inline void dumpMemRefDesc(const char *tag, const DynamicMemRefType<T> &m) {
  if (!positDescDebugEnabled())
    return;
  std::fprintf(stderr, "[PDESC] %s data=%p rank=%lld offset=%lld sizes=[", tag,
               static_cast<const void *>(m.data),
               static_cast<long long>(m.rank),
               static_cast<long long>(m.offset));
  for (int i = 0; i < m.rank; ++i) {
    std::fprintf(stderr, "%lld%s", static_cast<long long>(m.sizes[i]),
                 (i + 1 == m.rank) ? "" : ",");
  }
  std::fprintf(stderr, "] strides=[");
  for (int i = 0; i < m.rank; ++i) {
    std::fprintf(stderr, "%lld%s", static_cast<long long>(m.strides[i]),
                 (i + 1 == m.rank) ? "" : ",");
  }
  std::fprintf(stderr, "]\n");
}

template <typename T>
static inline int64_t offset_of(const DynamicMemRefType<T> &m, int64_t linear) {
  int64_t off = m.offset;
  int64_t rem = linear;
  for (int i = m.rank - 1; i >= 0; --i) {
    int64_t idx = rem % m.sizes[i];
    rem /= m.sizes[i];
    off += idx * m.strides[i];
  }
  return off;
}

template <typename T>
static inline bool checked_offset(const DynamicMemRefType<T> &m, const int64_t *idx,
                                  int64_t usedRank, int64_t &off) {
  if (!idx || usedRank < 0 || usedRank > m.rank)
    return false;
  __int128 sum = static_cast<__int128>(m.offset);
  for (int64_t d = 0; d < usedRank; ++d) {
    int64_t sz = m.sizes[d];
    int64_t id = idx[d];
    if (id < 0 || id >= sz)
      return false;
    sum += static_cast<__int128>(id) * static_cast<__int128>(m.strides[d]);
    if (sum < 0 || sum > static_cast<__int128>(1LL << 40)) {
      recordPositFallback(gPositMemrefGuardRejectCount, "memref_guard",
                          "computed offset overflow");
      return false;
    }
  }
  off = static_cast<int64_t>(sum);
  return true;
}

template <typename T>
static inline std::make_unsigned_t<T> to_bits(T v) {
  return static_cast<std::make_unsigned_t<T>>(v);
}

template <typename T>
static inline T from_bits(std::make_unsigned_t<T> v) {
  return static_cast<T>(v);
}

template <typename T>
static inline std::make_unsigned_t<T> load1(const DynamicMemRefType<T> &m, int64_t i) {
  if (!m.data)
    return 0;
  if (m.rank < 1)
    return 0;
  int64_t idx[1] = {i};
  int64_t off = 0;
  if (!checked_offset(m, idx, 1, off))
    return 0;
  return to_bits(m.data[off]);
}

template <typename T>
static inline std::make_unsigned_t<T> load2(const DynamicMemRefType<T> &m, int64_t i, int64_t j) {
  if (!m.data)
    return 0;
  if (m.rank < 2)
    return 0;
  int64_t idx[2] = {i, j};
  int64_t off = 0;
  if (!checked_offset(m, idx, 2, off))
    return 0;
  return to_bits(m.data[off]);
}

template <typename T>
static inline void store2(DynamicMemRefType<T> &m, int64_t i, int64_t j,
                          std::make_unsigned_t<T> bits) {
  if (!m.data)
    return;
  if (m.rank < 2)
    return;
  int64_t idx[2] = {i, j};
  int64_t off = 0;
  if (!checked_offset(m, idx, 2, off))
    return;
  m.data[off] = from_bits<T>(bits);
}

template <typename T>
static inline std::make_unsigned_t<T> load3(const DynamicMemRefType<T> &m, int64_t i,
                                            int64_t j, int64_t k) {
  if (!m.data)
    return 0;
  if (m.rank < 3)
    return 0;
  int64_t idx[3] = {i, j, k};
  int64_t off = 0;
  if (!checked_offset(m, idx, 3, off))
    return 0;
  return to_bits(m.data[off]);
}

template <typename T>
static inline void store3(DynamicMemRefType<T> &m, int64_t i, int64_t j, int64_t k,
                          std::make_unsigned_t<T> bits) {
  if (!m.data)
    return;
  if (m.rank < 3)
    return;
  int64_t idx[3] = {i, j, k};
  int64_t off = 0;
  if (!checked_offset(m, idx, 3, off))
    return;
  m.data[off] = from_bits<T>(bits);
}

template <typename T>
static inline std::make_unsigned_t<T> load_broadcast_nd(const DynamicMemRefType<T> &m,
                                                        const int64_t *outIdx,
                                                        int64_t outRank) {
  if (!m.data)
    return 0;
  if (m.rank == 0)
    return to_bits(m.data[m.offset]);
  int64_t delta = outRank - m.rank;
  int64_t midx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (m.rank > 8)
    return 0;
  for (int64_t od = 0; od < outRank; ++od) {
    int64_t md = od - delta;
    if (md < 0)
      continue;
    int64_t idx = outIdx[od];
    if (m.sizes[md] == 1)
      idx = 0;
    if (idx < 0 || idx >= m.sizes[md])
      return 0;
    midx[md] = idx;
  }
  int64_t off = 0;
  if (!checked_offset(m, midx, m.rank, off))
    return 0;
  return to_bits(m.data[off]);
}

template <typename T>
static inline std::make_unsigned_t<T> load4(const DynamicMemRefType<T> &m, int64_t n,
                                            int64_t c, int64_t h, int64_t w) {
  if (!m.data)
    return 0;
  if (m.rank < 4)
    return 0;
  int64_t idx[4] = {n, c, h, w};
  int64_t off = 0;
  if (!checked_offset(m, idx, 4, off))
    return 0;
  return to_bits(m.data[off]);
}

template <typename T>
static inline void store4(DynamicMemRefType<T> &m, int64_t n, int64_t c, int64_t h,
                          int64_t w, std::make_unsigned_t<T> bits) {
  if (!m.data)
    return;
  if (m.rank < 4)
    return;
  int64_t idx[4] = {n, c, h, w};
  int64_t off = 0;
  if (!checked_offset(m, idx, 4, off))
    return;
  m.data[off] = from_bits<T>(bits);
}

template <typename T>
static inline std::make_unsigned_t<T> loadC_broadcast(const DynamicMemRefType<T> &c, int64_t i,
                                                      int64_t j) {
  if (!c.data)
    return 0;
  if (c.rank == 0)
    return to_bits(c.data[c.offset]);
  if (c.rank == 1) {
    int64_t jj = (c.sizes[0] == 1) ? 0 : j;
    if (jj < 0 || jj >= c.sizes[0])
      return 0;
    return load1(c, jj);
  }
  int64_t ii = (c.sizes[0] == 1) ? 0 : i;
  int64_t jj = (c.sizes[1] == 1) ? 0 : j;
  if (ii < 0 || jj < 0 || ii >= c.sizes[0] || jj >= c.sizes[1])
    return 0;
  return load2(c, ii, jj);
}

static std::vector<std::string> parseCsvLowerTokensRuntime(
    const std::string &csv) {
  std::vector<std::string> out;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok = trimAscii(tok);
    std::transform(tok.begin(), tok.end(), tok.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (!tok.empty())
      out.push_back(tok);
  }
  return out;
}

static int parseIntEnvWithFallback(const char *name, int fallback) {
  const char *e = std::getenv(name);
  if (!e || !*e)
    return fallback;
  char *end = nullptr;
  long v = std::strtol(e, &end, 10);
  if (!end || *end != '\0')
    return fallback;
  return static_cast<int>(v);
}

static inline bool positDescDebugEnabled();

static inline bool runtimeEnvHasValue(const char *name) {
  const char *e = std::getenv(name);
  return e && *e;
}

static inline std::vector<std::string> splitCsvTrimmedRuntime(
    const std::string &line) {
  std::vector<std::string> cols;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ','))
    cols.push_back(trimAscii(tok));
  return cols;
}

static inline int findCsvColumnIndexRuntime(
    const std::vector<std::string> &header,
    std::initializer_list<const char *> names) {
  for (const char *name : names) {
    for (size_t i = 0; i < header.size(); ++i) {
      if (header[i] == name)
        return static_cast<int>(i);
    }
  }
  return -1;
}

static inline bool parseIntCellRuntime(const std::string &s, int &out) {
  if (s.empty())
    return false;
  char *end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (!end || *end != '\0')
    return false;
  out = static_cast<int>(v);
  return true;
}

static inline bool parseU64CellRuntime(const std::string &s, uint64_t &out) {
  if (s.empty())
    return false;
  char *end = nullptr;
  unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (!end || *end != '\0')
    return false;
  out = static_cast<uint64_t>(v);
  return true;
}

struct GP8AutoConfig {
  bool found = false;
  int rs = 7;
  int sc = 0;
  std::string qalignPath;
  std::string detailPath;
  std::string sourceKind;
};

static GP8AutoConfig loadGP8AutoConfigFromQAlignCsv(const char *formatName) {
  GP8AutoConfig cfg;
  const char *qalignFileEnv = std::getenv("POSIT_QALIGN_FILE");
  if (!qalignFileEnv || !*qalignFileEnv)
    return cfg;
  std::ifstream fin(qalignFileEnv);
  if (!fin.is_open())
    return cfg;
  struct Score {
    uint64_t rows = 0;
  };
  std::unordered_map<uint64_t, Score> pairScores;
  std::string line;
  while (std::getline(fin, line)) {
    line = trimAscii(line);
    if (line.empty() || line[0] == '#')
      continue;
    std::vector<std::string> cols = splitCsvTrimmedRuntime(line);
    if (cols.size() < 7)
      continue;
    int rs = 0;
    int sc = 0;
    if (!parseIntCellRuntime(cols[5], rs) || !parseIntCellRuntime(cols[6], sc))
      continue;
    rs = std::min(7, std::max(1, rs));
    sc = std::min(16, std::max(-16, sc));
    uint64_t packed =
        (static_cast<uint64_t>(static_cast<uint32_t>(rs)) << 32) |
        static_cast<uint32_t>(sc + 32768);
    pairScores[packed].rows += 1;
  }
  if (pairScores.empty())
    return cfg;
  uint64_t bestPacked = 0;
  Score bestScore{};
  bool haveBest = false;
  for (const auto &kv : pairScores) {
    const Score &s = kv.second;
    if (!haveBest || s.rows > bestScore.rows ||
        (s.rows == bestScore.rows && kv.first < bestPacked)) {
      bestPacked = kv.first;
      bestScore = s;
      haveBest = true;
    }
  }
  if (!haveBest)
    return cfg;
  cfg.found = true;
  cfg.rs = static_cast<int>((bestPacked >> 32) & 0xffffffffu);
  cfg.sc = static_cast<int>(static_cast<uint32_t>(bestPacked & 0xffffffffu)) -
           32768;
  cfg.qalignPath = qalignFileEnv;
  cfg.sourceKind = "qalign_csv_applied";
  if (positDescDebugEnabled()) {
    std::fprintf(stderr,
                 "[GP] auto-selected %s rs=%d sc=%d from %s (%s)\n",
                 formatName, cfg.rs, cfg.sc, cfg.qalignPath.c_str(),
                 cfg.sourceKind.c_str());
  }
  return cfg;
}

static GP8AutoConfig loadGP8AutoConfigFromQAlignDetail(
    const char *formatName) {
  GP8AutoConfig cfg;
  const char *qalignFileEnv = std::getenv("POSIT_QALIGN_FILE");
  if (!qalignFileEnv || !*qalignFileEnv)
    return cfg;
  std::string qalignPath(qalignFileEnv);
  size_t slash = qalignPath.find_last_of('/');
  std::string dir =
      (slash == std::string::npos) ? std::string(".") : qalignPath.substr(0, slash);
  std::string file =
      (slash == std::string::npos) ? qalignPath : qalignPath.substr(slash + 1);
  std::string detailFile = file;
  size_t dot = detailFile.rfind(".csv");
  if (dot != std::string::npos)
    detailFile.replace(dot, 4, "_detail.csv");
  else
    detailFile += "_detail.csv";
  std::string detailPath = dir + "/" + detailFile;
  std::ifstream fin(detailPath);
  if (!fin.is_open())
    return cfg;

  std::string autoSource = "from_ref";
  if (const char *e = std::getenv("POSIT_GP_AUTO_SOURCE"); e && *e) {
    autoSource = e;
    std::transform(autoSource.begin(), autoSource.end(), autoSource.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
  }
  std::vector<std::pair<std::string, std::string>> candidates;
  if (autoSource == "local") {
    candidates.push_back({"gp_rs_local", "gp_sc_local"});
    candidates.push_back({"gp_rs_from_ref", "gp_sc_from_ref"});
  } else {
    candidates.push_back({"gp_rs_from_ref", "gp_sc_from_ref"});
    candidates.push_back({"gp_rs_local", "gp_sc_local"});
  }

  std::string headerLine;
  if (!std::getline(fin, headerLine))
    return cfg;
  std::vector<std::string> header = splitCsvTrimmedRuntime(headerLine);
  int sampleCountIdx = findCsvColumnIndexRuntime(header, {"sample_count"});
  int rsIdx = -1;
  int scIdx = -1;
  std::string chosenSource = autoSource;
  for (const auto &cand : candidates) {
    rsIdx = findCsvColumnIndexRuntime(header, {cand.first.c_str()});
    scIdx = findCsvColumnIndexRuntime(header, {cand.second.c_str()});
    if (rsIdx >= 0 && scIdx >= 0) {
      chosenSource = cand.first + "+" + cand.second;
      break;
    }
  }
  if (rsIdx < 0 || scIdx < 0)
    return cfg;

  struct Score {
    uint64_t weight = 0;
    uint64_t rows = 0;
  };
  std::unordered_map<uint64_t, Score> pairScores;
  std::string line;
  while (std::getline(fin, line)) {
    line = trimAscii(line);
    if (line.empty())
      continue;
    std::vector<std::string> cols = splitCsvTrimmedRuntime(line);
    if (rsIdx >= static_cast<int>(cols.size()) ||
        scIdx >= static_cast<int>(cols.size()))
      continue;
    int rs = 0;
    int sc = 0;
    if (!parseIntCellRuntime(cols[rsIdx], rs) ||
        !parseIntCellRuntime(cols[scIdx], sc))
      continue;
    uint64_t weight = 1;
    if (sampleCountIdx >= 0 && sampleCountIdx < static_cast<int>(cols.size())) {
      uint64_t parsed = 0;
      if (parseU64CellRuntime(cols[sampleCountIdx], parsed) && parsed > 0)
        weight = parsed;
    }
    rs = std::min(7, std::max(1, rs));
    sc = std::min(16, std::max(-16, sc));
    uint64_t packed =
        (static_cast<uint64_t>(static_cast<uint32_t>(rs)) << 32) |
        static_cast<uint32_t>(sc + 32768);
    auto &score = pairScores[packed];
    score.weight += weight;
    score.rows += 1;
  }
  if (pairScores.empty())
    return cfg;

  uint64_t bestPacked = 0;
  Score bestScore{};
  bool haveBest = false;
  for (const auto &kv : pairScores) {
    const Score &s = kv.second;
    if (!haveBest || s.weight > bestScore.weight ||
        (s.weight == bestScore.weight && s.rows > bestScore.rows) ||
        (s.weight == bestScore.weight && s.rows == bestScore.rows &&
         kv.first < bestPacked)) {
      bestPacked = kv.first;
      bestScore = s;
      haveBest = true;
    }
  }
  if (!haveBest)
    return cfg;
  cfg.found = true;
  cfg.rs = static_cast<int>((bestPacked >> 32) & 0xffffffffu);
  cfg.sc = static_cast<int>(static_cast<uint32_t>(bestPacked & 0xffffffffu)) -
           32768;
  cfg.qalignPath = qalignPath;
  cfg.detailPath = detailPath;
  cfg.sourceKind = chosenSource;
  if (positDescDebugEnabled()) {
    std::fprintf(stderr,
                 "[GP] auto-selected %s rs=%d sc=%d from %s (%s)\n",
                 formatName, cfg.rs, cfg.sc, cfg.detailPath.c_str(),
                 cfg.sourceKind.c_str());
  }
  return cfg;
}

struct GP8RuntimeTable {
  bool enabled = false;
  int es = 0;
  int rs = 7;
  int sc = 0;
  std::array<double, 256> decode{};
  std::vector<std::pair<double, uint8_t>> sorted;
};

static double decodeApproxGeneralizedPosit8Runtime(
    uint8_t raw, int es, int rs, int sc) {
  constexpr int nbits = 8;
  if (raw == 0)
    return 0.0;
  if (raw == 0x80)
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

static uint8_t encodeApproxGeneralizedPosit8Runtime(
    double x, const GP8RuntimeTable &table) {
  if (!std::isfinite(x))
    return 0x80u;
  if (table.sorted.empty())
    return 0u;
  auto it = std::lower_bound(
      table.sorted.begin(), table.sorted.end(), x,
      [](const std::pair<double, uint8_t> &a, double v) { return a.first < v; });
  if (it == table.sorted.begin())
    return it->second;
  if (it == table.sorted.end())
    return table.sorted.back().second;
  auto prev = it - 1;
  return (std::fabs(prev->first - x) <= std::fabs(it->first - x)) ? prev->second
                                                                   : it->second;
}

template <int ES>
static const GP8RuntimeTable &getGP8RuntimeTableForFormat(const char *formatName) {
  static GP8RuntimeTable table = [formatName]() {
    GP8RuntimeTable t;
    t.es = ES;
    std::string formatsCsv;
    if (const char *e = std::getenv("POSIT_GP_EXPERIMENTAL_FORMATS"); e && *e)
      formatsCsv = e;
    else if (const char *e2 = std::getenv("POSIT_GP_FORMATS"); e2 && *e2)
      formatsCsv = e2;
    auto toks = parseCsvLowerTokensRuntime(formatsCsv);
    std::string want = formatName;
    std::transform(want.begin(), want.end(), want.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    t.enabled =
        std::find(toks.begin(), toks.end(), want) != toks.end();
    std::string upper = want;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) {
      return static_cast<char>(std::toupper(c));
    });
    std::replace(upper.begin(), upper.end(), '-', '_');
    std::string rsFormatEnv = "POSIT_GP_RS_" + upper;
    std::string scFormatEnv = "POSIT_GP_SC_" + upper;
    bool hasExplicitRs = runtimeEnvHasValue(rsFormatEnv.c_str()) ||
                         runtimeEnvHasValue("POSIT_GP_RS_P8") ||
                         runtimeEnvHasValue("POSIT_GP_RS");
    bool hasExplicitSc = runtimeEnvHasValue(scFormatEnv.c_str()) ||
                         runtimeEnvHasValue("POSIT_GP_SC_P8") ||
                         runtimeEnvHasValue("POSIT_GP_SC");
    int rs = 7;
    int sc = 0;
    if (hasExplicitRs || hasExplicitSc) {
      rs = parseIntEnvWithFallback(
          rsFormatEnv.c_str(),
          parseIntEnvWithFallback("POSIT_GP_RS_P8",
                                  parseIntEnvWithFallback("POSIT_GP_RS", 7)));
      sc = parseIntEnvWithFallback(
          scFormatEnv.c_str(),
          parseIntEnvWithFallback("POSIT_GP_SC_P8",
                                  parseIntEnvWithFallback("POSIT_GP_SC", 0)));
    } else {
      GP8AutoConfig autoCfg = loadGP8AutoConfigFromQAlignCsv(formatName);
      if (!autoCfg.found)
        autoCfg = loadGP8AutoConfigFromQAlignDetail(formatName);
      if (autoCfg.found) {
        rs = autoCfg.rs;
        sc = autoCfg.sc;
      }
    }
    t.rs = std::min(7, std::max(1, rs));
    t.sc = std::min(16, std::max(-16, sc));
    for (int i = 0; i < 256; ++i) {
      uint8_t raw = static_cast<uint8_t>(i);
      t.decode[i] = decodeApproxGeneralizedPosit8Runtime(raw, ES, t.rs, t.sc);
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
    return t;
  }();
  return table;
}

template <int NBits, int ES>
static inline bool positExperimentalGPEnabled() {
  if constexpr (NBits == 8 && ES == 0)
    return getGP8RuntimeTableForFormat<0>("p8e0").enabled;
  if constexpr (NBits == 8 && ES == 1)
    return getGP8RuntimeTableForFormat<1>("p8e1").enabled;
  if constexpr (NBits == 8 && ES == 2)
    return getGP8RuntimeTableForFormat<2>("p8e2").enabled;
  return false;
}

template <int NBits, int ES>
static inline const GP8RuntimeTable &positExperimentalGPTable() {
  if constexpr (NBits == 8 && ES == 0)
    return getGP8RuntimeTableForFormat<0>("p8e0");
  if constexpr (NBits == 8 && ES == 1)
    return getGP8RuntimeTableForFormat<1>("p8e1");
  return getGP8RuntimeTableForFormat<2>("p8e2");
}

template <int ES>
static const GP8RuntimeTable &getGP8RuntimeTableCustom(int rs, int sc) {
  static std::mutex cacheMutex;
  static std::unordered_map<int64_t, GP8RuntimeTable> cache;
  rs = std::min(7, std::max(1, rs));
  sc = std::min(16, std::max(-16, sc));
  int64_t key = (static_cast<int64_t>(rs) << 32) ^
                static_cast<uint32_t>(sc - std::numeric_limits<int16_t>::min());
  std::lock_guard<std::mutex> lock(cacheMutex);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  GP8RuntimeTable t;
  t.es = ES;
  t.enabled = true;
  t.rs = rs;
  t.sc = sc;
  for (int i = 0; i < 256; ++i) {
    uint8_t raw = static_cast<uint8_t>(i);
    t.decode[i] = decodeApproxGeneralizedPosit8Runtime(raw, ES, rs, sc);
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
  auto [inserted, _] = cache.emplace(key, std::move(t));
  return inserted->second;
}

static inline void registerTensorGPMetadata(const void *ptr,
                                            const TensorGPMetadata &meta) {
  if (!ptr)
    return;
  std::lock_guard<std::mutex> lock(gTensorGPMetaMutex);
  gTensorGPMeta[ptr] = meta;
}

static inline void registerTensorGPMetadataForChannel(
    const void *ptr, int64_t channel, const TensorGPMetadata &meta) {
  if (!ptr)
    return;
  std::lock_guard<std::mutex> lock(gTensorGPMetaMutex);
  gTensorGPChannelMeta[TensorMetaChannelKey{ptr, channel}] = meta;
  if (tensorHasSpecialDecodeMetadata(meta))
    gTensorGPSpecialChannelMetaPtrs.insert(ptr);
  if (tensorCompandIsAlps(meta))
    gTensorGPAlpsChannelMetaPtrs.insert(ptr);
}

static inline bool tensorHasSpecialChannelGPMetadata(const void *ptr) {
  if (!ptr)
    return false;
  std::lock_guard<std::mutex> lock(gTensorGPMetaMutex);
  return gTensorGPSpecialChannelMetaPtrs.find(ptr) !=
         gTensorGPSpecialChannelMetaPtrs.end();
}

static inline bool tensorHasAlpsChannelGPMetadata(const void *ptr) {
  if (!ptr)
    return false;
  std::lock_guard<std::mutex> lock(gTensorGPMetaMutex);
  return gTensorGPAlpsChannelMetaPtrs.find(ptr) !=
         gTensorGPAlpsChannelMetaPtrs.end();
}

static inline TensorGPMetadata lookupTensorGPMetadata(const void *ptr) {
  TensorGPMetadata meta;
  if (!ptr)
    return meta;
  std::lock_guard<std::mutex> lock(gTensorGPMetaMutex);
  auto it = gTensorGPMeta.find(ptr);
  if (it != gTensorGPMeta.end())
    return it->second;
  return meta;
}

static inline TensorGPMetadata lookupTensorGPMetadataForChannel(
    const void *ptr, int64_t channel, const TensorGPMetadata &fallback) {
  if (!ptr)
    return fallback;
  std::lock_guard<std::mutex> lock(gTensorGPMetaMutex);
  auto it = gTensorGPChannelMeta.find(TensorMetaChannelKey{ptr, channel});
  if (it != gTensorGPChannelMeta.end())
    return it->second;
  auto baseIt = gTensorGPMeta.find(ptr);
  if (baseIt != gTensorGPMeta.end())
    return baseIt->second;
  return fallback;
}

template <typename T>
static inline const void *tensorMetaPtr(const DynamicMemRefType<T> &m) {
  return static_cast<const void *>(m.data);
}

template <typename T>
static inline void registerTensorCompandMetadata(
    const DynamicMemRefType<T> &m, int64_t mode, double theta, double gamma) {
  TensorGPMetadata meta = lookupTensorGPMetadata(tensorMetaPtr(m));
  meta.valid = true;
  meta.compandMode = static_cast<int>(mode);
  meta.theta = theta;
  meta.gamma = gamma;
  registerTensorGPMetadata(tensorMetaPtr(m), meta);
}

template <typename T>
static inline void registerTensorConstMetadata(
    const DynamicMemRefType<T> &m, int64_t mode, double theta, double gamma,
    int64_t gpEnabled, int64_t gpRs, int64_t gpSc) {
  TensorGPMetadata meta = lookupTensorGPMetadata(tensorMetaPtr(m));
  meta.valid = true;
  meta.compandMode = static_cast<int>(mode);
  meta.theta = theta;
  meta.gamma = gamma;
  meta.enabled = (gpEnabled != 0);
  meta.rs = std::min(7, std::max(1, static_cast<int>(gpRs)));
  meta.sc = std::min(16, std::max(-16, static_cast<int>(gpSc)));
  registerTensorGPMetadata(tensorMetaPtr(m), meta);
}

template <typename T>
static inline void registerTensorConstMetadataForChannel(
    const DynamicMemRefType<T> &m, int64_t channel, int64_t mode, double theta,
    double gamma, int64_t gpEnabled, int64_t gpRs, int64_t gpSc) {
  const void *ptr = tensorMetaPtr(m);
  TensorGPMetadata meta = lookupTensorGPMetadata(ptr);
  meta.valid = true;
  meta.compandMode = static_cast<int>(mode);
  meta.theta = theta;
  meta.gamma = gamma;
  meta.enabled = (gpEnabled != 0);
  meta.rs = std::min(7, std::max(1, static_cast<int>(gpRs)));
  meta.sc = std::min(16, std::max(-16, static_cast<int>(gpSc)));
  registerTensorGPMetadataForChannel(ptr, channel, meta);
  TensorGPMetadata base = lookupTensorGPMetadata(ptr);
  if (!base.valid) {
    TensorGPMetadata directBase;
    directBase.valid = true;
    directBase.enabled = false;
    directBase.rs = 7;
    directBase.sc = 0;
    directBase.qalignKey = meta.qalignKey;
    directBase.compandMode = 0;
    directBase.theta = 1.0;
    directBase.gamma = 0.0;
    registerTensorGPMetadata(ptr, directBase);
  }
}


static inline bool positGpGlobalArithmeticEnabled() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  const char *e = std::getenv("POSIT_GP_GLOBAL_ARITHMETIC");
  if (!e || !*e) {
    enabled = 0;
    return false;
  }
  std::string v(e);
  for (char &ch : v)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  enabled = (v == "1" || v == "on" || v == "true" || v == "yes") ? 1 : 0;
  return enabled == 1;
}

//===----------------------------------------------------------------------===//
// Format traits (backend-specific)
//===----------------------------------------------------------------------===//

#if defined(POSIT_USE_UNIVERSAL)

template <int NBits, int ES, typename MemT_, typename UIntT_>
struct UniversalFmt {
  using MemT = MemT_;
  using UIntT = UIntT_;
  using PositT = sw::universal::posit<NBits, ES>;
  static constexpr int nbits = NBits;  // exposed for format-generic gates (e.g. ALPS)
  static constexpr int es = ES;

  static PositT fromRaw(UIntT b) {
    PositT p;
    p.setbits(static_cast<uint64_t>(b));
    return p;
  }
  static UIntT toRaw(const PositT &p) {
    return static_cast<UIntT>(p.bits().to_ull());
  }
  static PositT fromDouble(double x) {
    if constexpr (NBits == 8 && ES >= 0 && ES <= 2) {
      if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<NBits, ES>()) {
        UIntT raw = static_cast<UIntT>(
            encodeApproxGeneralizedPosit8Runtime(
                x, positExperimentalGPTable<NBits, ES>()));
        return fromRaw(raw);
      }
    }
    PositT p;
    p = x;
    return p;
  }
  static double toDouble(const PositT &p) {
    if constexpr (NBits == 8 && ES >= 0 && ES <= 2) {
      if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<NBits, ES>())
        return positExperimentalGPTable<NBits, ES>().decode[toRaw(p)];
    }
    return static_cast<double>(p);
  }
  static PositT add(const PositT &a, const PositT &b) {
    if constexpr (NBits == 8 && ES >= 0 && ES <= 2) {
      if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<NBits, ES>())
        return fromDouble(toDouble(a) + toDouble(b));
    }
    return a + b;
  }
  static PositT sub(const PositT &a, const PositT &b) {
    if constexpr (NBits == 8 && ES >= 0 && ES <= 2) {
      if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<NBits, ES>())
        return fromDouble(toDouble(a) - toDouble(b));
    }
    return a - b;
  }
  static PositT mul(const PositT &a, const PositT &b) {
    if constexpr (NBits == 8 && ES >= 0 && ES <= 2) {
      if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<NBits, ES>())
        return fromDouble(toDouble(a) * toDouble(b));
    }
    return a * b;
  }
  static PositT div(const PositT &a, const PositT &b) {
    if constexpr (NBits == 8 && ES >= 0 && ES <= 2) {
      if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<NBits, ES>())
        return fromDouble(toDouble(a) / toDouble(b));
    }
    return a / b;
  }
};

using FmtP8E0 = UniversalFmt<8, 0, int8_t, uint8_t>;
using FmtP8E1 = UniversalFmt<8, 1, int8_t, uint8_t>;
using FmtP8E2 = UniversalFmt<8, 2, int8_t, uint8_t>;
using FmtP4E0 = UniversalFmt<4, 0, int8_t, uint8_t>;
using FmtP4E1 = UniversalFmt<4, 1, int8_t, uint8_t>;
using FmtP4E2 = UniversalFmt<4, 2, int8_t, uint8_t>;
using FmtP4E3 = UniversalFmt<4, 3, int8_t, uint8_t>;
using FmtP5E0 = UniversalFmt<5, 0, int8_t, uint8_t>;
using FmtP5E1 = UniversalFmt<5, 1, int8_t, uint8_t>;
using FmtP5E2 = UniversalFmt<5, 2, int8_t, uint8_t>;
using FmtP5E3 = UniversalFmt<5, 3, int8_t, uint8_t>;
using FmtP6E0 = UniversalFmt<6, 0, int8_t, uint8_t>;
using FmtP6E1 = UniversalFmt<6, 1, int8_t, uint8_t>;
using FmtP6E2 = UniversalFmt<6, 2, int8_t, uint8_t>;
using FmtP6E3 = UniversalFmt<6, 3, int8_t, uint8_t>;
using FmtP7E0 = UniversalFmt<7, 0, int8_t, uint8_t>;
using FmtP7E1 = UniversalFmt<7, 1, int8_t, uint8_t>;
using FmtP7E2 = UniversalFmt<7, 2, int8_t, uint8_t>;
using FmtP7E3 = UniversalFmt<7, 3, int8_t, uint8_t>;
using FmtP9E0 = UniversalFmt<9, 0, int16_t, uint16_t>;
using FmtP9E1 = UniversalFmt<9, 1, int16_t, uint16_t>;
using FmtP9E2 = UniversalFmt<9, 2, int16_t, uint16_t>;
using FmtP9E3 = UniversalFmt<9, 3, int16_t, uint16_t>;
using FmtP10E0 = UniversalFmt<10, 0, int16_t, uint16_t>;
using FmtP10E1 = UniversalFmt<10, 1, int16_t, uint16_t>;
using FmtP10E2 = UniversalFmt<10, 2, int16_t, uint16_t>;
using FmtP11E0 = UniversalFmt<11, 0, int16_t, uint16_t>;
using FmtP11E1 = UniversalFmt<11, 1, int16_t, uint16_t>;
using FmtP11E2 = UniversalFmt<11, 2, int16_t, uint16_t>;
using FmtP12E0 = UniversalFmt<12, 0, int16_t, uint16_t>;
using FmtP12E1 = UniversalFmt<12, 1, int16_t, uint16_t>;
using FmtP12E2 = UniversalFmt<12, 2, int16_t, uint16_t>;
using FmtP13E0 = UniversalFmt<13, 0, int16_t, uint16_t>;
using FmtP13E1 = UniversalFmt<13, 1, int16_t, uint16_t>;
using FmtP13E2 = UniversalFmt<13, 2, int16_t, uint16_t>;
using FmtP14E0 = UniversalFmt<14, 0, int16_t, uint16_t>;
using FmtP14E1 = UniversalFmt<14, 1, int16_t, uint16_t>;
using FmtP14E2 = UniversalFmt<14, 2, int16_t, uint16_t>;
using FmtP15E0 = UniversalFmt<15, 0, int16_t, uint16_t>;
using FmtP15E1 = UniversalFmt<15, 1, int16_t, uint16_t>;
using FmtP15E2 = UniversalFmt<15, 2, int16_t, uint16_t>;
using FmtP16E0 = UniversalFmt<16, 0, int16_t, uint16_t>;
using FmtP16E1 = UniversalFmt<16, 1, int16_t, uint16_t>;
using FmtP16E2 = UniversalFmt<16, 2, int16_t, uint16_t>;
using FmtP32E0 = UniversalFmt<32, 0, int32_t, uint32_t>;
using FmtP32E1 = UniversalFmt<32, 1, int32_t, uint32_t>;
using FmtP32E2 = UniversalFmt<32, 2, int32_t, uint32_t>;

#else

// SoftPosit dynamic posit path for p8e1: map 8-bit raw to/from posit_1_t with x=8.
// This requires SoftPosit to export pX1_* symbols.
// Build with -DPOSIT_USE_SOFTPOSIT_PX1 to enable.
struct FmtP8E1ViaPX1 {
  using MemT = int8_t;
  using UIntT = uint8_t;
  using PositT = posit_1_t;
  static PositT fromRaw(UIntT b) {
    PositT p;
    p.v = static_cast<uint32_t>(b) << 24;
    return p;
  }
  static UIntT toRaw(PositT p) {
    p = pX1_to_pX1(p, 8);
    return static_cast<UIntT>((p.v >> 24) & 0xFFu);
  }
  static PositT fromDouble(double x) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 1>())
      return fromRaw(static_cast<UIntT>(
          encodeApproxGeneralizedPosit8Runtime(
              x, positExperimentalGPTable<8, 1>())));
    return convertDoubleToPX1(x, 8);
  }
  static double toDouble(PositT p) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 1>())
      return positExperimentalGPTable<8, 1>().decode[toRaw(p)];
    return convertPX1ToDouble(p);
  }
  static PositT add(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 1>())
      return fromDouble(toDouble(a) + toDouble(b));
    return pX1_add(a, b, 8);
  }
  static PositT sub(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 1>())
      return fromDouble(toDouble(a) - toDouble(b));
    return pX1_sub(a, b, 8);
  }
  static PositT mul(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 1>())
      return fromDouble(toDouble(a) * toDouble(b));
    return pX1_mul(a, b, 8);
  }
  static PositT div(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 1>())
      return fromDouble(toDouble(a) / toDouble(b));
    return pX1_div(a, b, 8);
  }
};

// SoftPosit dynamic posit path for p8e2: map 8-bit raw to/from posit_2_t with x=8.
// This requires SoftPosit to export pX2/qX2 symbols.
// Build with -DPOSIT_USE_SOFTPOSIT_PX2 to enable.
struct FmtP8E2ViaPX2 {
  using MemT = int8_t;
  using UIntT = uint8_t;
  using PositT = posit_2_t;
  static PositT fromRaw(UIntT b) {
    PositT p;
    p.v = static_cast<uint32_t>(b) << 24;
    return p;
  }
  static UIntT toRaw(PositT p) {
    p = pX2_to_pX2(p, 8);
    return static_cast<UIntT>((p.v >> 24) & 0xFFu);
  }
  static PositT fromDouble(double x) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 2>())
      return fromRaw(static_cast<UIntT>(
          encodeApproxGeneralizedPosit8Runtime(
              x, positExperimentalGPTable<8, 2>())));
    return convertDoubleToPX2(x, 8);
  }
  static double toDouble(PositT p) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 2>())
      return positExperimentalGPTable<8, 2>().decode[toRaw(p)];
    return convertPX2ToDouble(p);
  }
  static PositT add(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 2>())
      return fromDouble(toDouble(a) + toDouble(b));
    return pX2_add(a, b, 8);
  }
  static PositT sub(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 2>())
      return fromDouble(toDouble(a) - toDouble(b));
    return pX2_sub(a, b, 8);
  }
  static PositT mul(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 2>())
      return fromDouble(toDouble(a) * toDouble(b));
    return pX2_mul(a, b, 8);
  }
  static PositT div(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 2>())
      return fromDouble(toDouble(a) / toDouble(b));
    return pX2_div(a, b, 8);
  }
};

struct FmtP8E0 {
  using MemT = int8_t;
  using UIntT = uint8_t;
  using PositT = posit8_t;
  static PositT fromRaw(UIntT b) { return castP8(b); }
  static UIntT toRaw(PositT p) { return static_cast<UIntT>(castUI(p) & 0xFFu); }
  static PositT fromDouble(double x) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 0>())
      return fromRaw(static_cast<UIntT>(
          encodeApproxGeneralizedPosit8Runtime(
              x, positExperimentalGPTable<8, 0>())));
    return convertDoubleToP8(x);
  }
  static double toDouble(PositT p) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 0>())
      return positExperimentalGPTable<8, 0>().decode[toRaw(p)];
    return convertP8ToDouble(p);
  }
  static PositT add(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 0>())
      return fromDouble(toDouble(a) + toDouble(b));
    return p8_add(a, b);
  }
  static PositT sub(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 0>())
      return fromDouble(toDouble(a) - toDouble(b));
    return p8_sub(a, b);
  }
  static PositT mul(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 0>())
      return fromDouble(toDouble(a) * toDouble(b));
    return p8_mul(a, b);
  }
  static PositT div(PositT a, PositT b) {
    if (positGpGlobalArithmeticEnabled() && positExperimentalGPEnabled<8, 0>())
      return fromDouble(toDouble(a) / toDouble(b));
    return p8_div(a, b);
  }
};

struct FmtP16E1 {
  using MemT = int16_t;
  using UIntT = uint16_t;
  using PositT = posit16_t;
  static PositT fromRaw(UIntT b) { return castP16(b); }
  static UIntT toRaw(PositT p) { return static_cast<UIntT>(castUI(p) & 0xFFFFu); }
  static PositT fromDouble(double x) { return convertDoubleToP16(x); }
  static double toDouble(PositT p) { return convertP16ToDouble(p); }
  static PositT add(PositT a, PositT b) { return p16_add(a, b); }
  static PositT sub(PositT a, PositT b) { return p16_sub(a, b); }
  static PositT mul(PositT a, PositT b) { return p16_mul(a, b); }
  static PositT div(PositT a, PositT b) { return p16_div(a, b); }
};

// SoftPosit dynamic posit path for p16e2: map 16-bit raw to/from posit_2_t with x=16.
#if defined(POSIT_USE_SOFTPOSIT_PX2)
struct FmtP16E2ViaPX2 {
  using MemT = int16_t;
  using UIntT = uint16_t;
  using PositT = posit_2_t;
  static PositT fromRaw(UIntT b) {
    PositT p;
    p.v = static_cast<uint32_t>(b) << 16;
    return p;
  }
  static UIntT toRaw(PositT p) {
    p = pX2_to_pX2(p, 16);
    return static_cast<UIntT>((p.v >> 16) & 0xFFFFu);
  }
  static PositT fromDouble(double x) { return convertDoubleToPX2(x, 16); }
  static double toDouble(PositT p) { return convertPX2ToDouble(p); }
  static PositT add(PositT a, PositT b) { return pX2_add(a, b, 16); }
  static PositT sub(PositT a, PositT b) { return pX2_sub(a, b, 16); }
  static PositT mul(PositT a, PositT b) { return pX2_mul(a, b, 16); }
  static PositT div(PositT a, PositT b) { return pX2_div(a, b, 16); }
};
#endif

#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
template <int NBits, int ES, typename MemT_, typename UIntT_>
struct SoftPositUniversalFallbackFmt {
  using MemT = MemT_;
  using UIntT = UIntT_;
  using PositT = sw::universal::posit<NBits, ES>;

  static PositT fromRaw(UIntT b) {
    PositT p;
    p.setbits(static_cast<uint64_t>(b));
    return p;
  }
  static UIntT toRaw(const PositT &p) {
    return static_cast<UIntT>(p.bits().to_ull());
  }
  static PositT fromDouble(double x) {
    PositT p;
    p = x;
    return p;
  }
  static double toDouble(const PositT &p) { return static_cast<double>(p); }
  static PositT add(const PositT &a, const PositT &b) { return a + b; }
  static PositT sub(const PositT &a, const PositT &b) { return a - b; }
  static PositT mul(const PositT &a, const PositT &b) { return a * b; }
  static PositT div(const PositT &a, const PositT &b) { return a / b; }
};

using FmtP16E0 = SoftPositUniversalFallbackFmt<16, 0, int16_t, uint16_t>;
using FmtP32E0 = SoftPositUniversalFallbackFmt<32, 0, int32_t, uint32_t>;
#endif

// SoftPosit default backend is kept for legacy formats only.
// Full p8/p16/p32 x e0/e1/e2 coverage requires POSIT_USE_UNIVERSAL.

#if defined(POSIT_USE_SOFTPOSIT_PX1)
struct FmtP32E1ViaPX1 {
  using MemT = int32_t;
  using UIntT = uint32_t;
  using PositT = posit_1_t;
  static PositT fromRaw(UIntT b) {
    PositT p;
    p.v = static_cast<uint32_t>(b);
    return p;
  }
  static UIntT toRaw(PositT p) {
    p = pX1_to_pX1(p, 32);
    return static_cast<UIntT>(p.v);
  }
  static PositT fromDouble(double x) { return convertDoubleToPX1(x, 32); }
  static double toDouble(PositT p) { return convertPX1ToDouble(p); }
  static PositT add(PositT a, PositT b) { return pX1_add(a, b, 32); }
  static PositT sub(PositT a, PositT b) { return pX1_sub(a, b, 32); }
  static PositT mul(PositT a, PositT b) { return pX1_mul(a, b, 32); }
  static PositT div(PositT a, PositT b) { return pX1_div(a, b, 32); }
};
#endif

struct FmtP32E2 {
  using MemT = int32_t;
  using UIntT = uint32_t;
  using PositT = posit32_t;
  static PositT fromRaw(UIntT b) { return castP32(b); }
  static UIntT toRaw(PositT p) { return static_cast<UIntT>(castUI(p)); }
  static PositT fromDouble(double x) { return convertDoubleToP32(x); }
  static double toDouble(PositT p) { return convertP32ToDouble(p); }
  static PositT add(PositT a, PositT b) { return p32_add(a, b); }
  static PositT sub(PositT a, PositT b) { return p32_sub(a, b); }
  static PositT mul(PositT a, PositT b) { return p32_mul(a, b); }
  static PositT div(PositT a, PositT b) { return p32_div(a, b); }
};

#endif

template <typename Fmt>
static inline typename Fmt::UIntT bits_from_double_fmt(double x) {
  return Fmt::toRaw(Fmt::fromDouble(x));
}

template <typename Fmt>
static inline double double_from_bits_fmt(typename Fmt::UIntT b) {
  // Speedup for 8-bit posit (p8e0/e1/e2): the raw->value decode has only 256
  // possible inputs, so precompute a lookup table once and index it instead of
  // calling the Universal library's fromRaw/toDouble on every element. The LUT
  // is filled from the exact same Fmt::toDouble(Fmt::fromRaw(i)) => bit-identical
  // to the direct call; it only removes the per-decode library-call cost. This is
  // the single decode entry point used by conv/gemm (and the plain/ALPS-yq
  // decode), so tabulating here accelerates every 8-bit decode path. Thread-safe:
  // C++ guarantees the function-local static is initialized exactly once.
  if constexpr (sizeof(typename Fmt::UIntT) == 1) {
    static const std::array<double, 256> lut = [] {
      std::array<double, 256> t{};
      for (int i = 0; i < 256; ++i)
        t[static_cast<size_t>(i)] =
            Fmt::toDouble(Fmt::fromRaw(static_cast<typename Fmt::UIntT>(i)));
      return t;
    }();
    return lut[static_cast<uint8_t>(b)];
  }
  return Fmt::toDouble(Fmt::fromRaw(b));
}

static inline bool positQAlignStrictFallbackWhenModeOffEnabled() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  const char *e = std::getenv("POSIT_QALIGN_STRICT_FALLBACK_WHEN_MODE_OFF");
  if (!e || !*e)
    e = std::getenv("QALIGN_STRICT_FALLBACK_WHEN_COMPAND_OFF");
  if (!e || !*e)
    e = std::getenv("QALIGN_STRICT_FALLBACK_WHEN_MODE_OFF");
  if (!e || !*e) {
    enabled = 1;
    return true;
  }
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  enabled = (s == "0" || s == "off" || s == "false" || s == "no") ? 0 : 1;
  return enabled == 1;
}

static inline void applyStrictFallbackIfNeeded(QAlignParams &p) {
  if (!positQAlignStrictFallbackWhenModeOffEnabled())
    return;
  const bool validAlps =
      p.compandMode == kQAlignCompandAlps && std::isfinite(p.beta) && p.beta > 0.0;
  if (validAlps)
    return;

  // Safety rule: mode=off means no ALPS, no legacy theta scaling, and no GP.
  // Otherwise mode=off would still do x -> theta*x -> posit_round -> /theta.
  p.alpha = 1.0;
  p.beta = 0.0;
  p.compandMode = kQAlignCompandOff;
  p.hasBeta = true;
  p.hasMode = true;
  p.gpRs = 7;
  p.gpSc = 0;
  p.hasGp = false;
  p.strictFallbackApplied = true;
}

template <typename Fmt>
static inline QAlignParams resolveQAlignParams(int64_t qalignKey,
                                               int64_t channel) {
  // CSV-only runtime behavior:
  // If the bucket is not explicitly present in POSIT_QALIGN_FILE, keep identity/off.
  // This avoids accidental global ALPS fallback on uncalibrated buckets.
  (void)sizeof(Fmt);
  QAlignParams out;
  out.alpha = 1.0;
  out.beta = 0.0;
  out.compandMode = kQAlignCompandOff;
  out.hasBeta = false;
  out.hasMode = false;
  if (qalignKey == 0)
    return out;
  const auto &tab = getQAlignParamTable();
  if (!tab.hasFile)
    return out;

  auto applyIfFound = [&](int64_t ch) -> bool {
    auto it = tab.params.find(QAlignBucketKey{qalignKey, ch});
    if (it == tab.params.end())
      return false;
    out.alpha = it->second.alpha;
    out.beta = it->second.hasBeta ? it->second.beta : 0.0;
    out.compandMode = it->second.hasMode ? it->second.compandMode : kQAlignCompandOff;
    out.hasBeta = it->second.hasBeta;
    out.hasMode = it->second.hasMode;
    out.gpRs = it->second.gpRs;
    out.gpSc = it->second.gpSc;
    out.hasGp = it->second.hasGp;
    out.foundInCsv = true;
    applyStrictFallbackIfNeeded(out);
    return true;
  };
  if (applyIfFound(channel))
    return out;
  if (channel != -1 && applyIfFound(-1))
    return out;
  return out;
}

static inline double qalignCompandAlps(double x, double beta) {
  if (!(beta > 0.0) || !std::isfinite(x))
    return x;
  double y = std::asinh(beta * x);
  if (!std::isfinite(y))
    return x;
  return y;
}

static inline double qalignDecompandAlps(double y, double beta) {
  if (!(beta > 0.0) || !std::isfinite(y))
    return y;
  double x = std::sinh(y) / beta;
  if (!std::isfinite(x))
    return y;
  return x;
}

template <typename Fmt>
static inline const char *qalignFormatName() {
  if constexpr (std::is_same_v<Fmt, FmtP8E0>)
    return "p8e0";
  if constexpr (std::is_same_v<Fmt, FmtP8E1>)
    return "p8e1";
  if constexpr (std::is_same_v<Fmt, FmtP8E2>)
    return "p8e2";
  if constexpr (std::is_same_v<Fmt, FmtP4E0>)
    return "p4e0";
  if constexpr (std::is_same_v<Fmt, FmtP4E1>)
    return "p4e1";
  if constexpr (std::is_same_v<Fmt, FmtP4E2>)
    return "p4e2";
  if constexpr (std::is_same_v<Fmt, FmtP4E3>)
    return "p4e3";
  if constexpr (std::is_same_v<Fmt, FmtP5E0>)
    return "p5e0";
  if constexpr (std::is_same_v<Fmt, FmtP5E1>)
    return "p5e1";
  if constexpr (std::is_same_v<Fmt, FmtP5E2>)
    return "p5e2";
  if constexpr (std::is_same_v<Fmt, FmtP5E3>)
    return "p5e3";
  if constexpr (std::is_same_v<Fmt, FmtP6E0>)
    return "p6e0";
  if constexpr (std::is_same_v<Fmt, FmtP6E1>)
    return "p6e1";
  if constexpr (std::is_same_v<Fmt, FmtP6E2>)
    return "p6e2";
  if constexpr (std::is_same_v<Fmt, FmtP6E3>)
    return "p6e3";
  if constexpr (std::is_same_v<Fmt, FmtP7E0>)
    return "p7e0";
  if constexpr (std::is_same_v<Fmt, FmtP7E1>)
    return "p7e1";
  if constexpr (std::is_same_v<Fmt, FmtP7E2>)
    return "p7e2";
  if constexpr (std::is_same_v<Fmt, FmtP7E3>)
    return "p7e3";
  if constexpr (std::is_same_v<Fmt, FmtP9E0>)
    return "p9e0";
  if constexpr (std::is_same_v<Fmt, FmtP9E1>)
    return "p9e1";
  if constexpr (std::is_same_v<Fmt, FmtP9E2>)
    return "p9e2";
  if constexpr (std::is_same_v<Fmt, FmtP9E3>)
    return "p9e3";
  if constexpr (std::is_same_v<Fmt, FmtP10E0>)
    return "p10e0";
  if constexpr (std::is_same_v<Fmt, FmtP10E1>)
    return "p10e1";
  if constexpr (std::is_same_v<Fmt, FmtP10E2>)
    return "p10e2";
  if constexpr (std::is_same_v<Fmt, FmtP11E0>)
    return "p11e0";
  if constexpr (std::is_same_v<Fmt, FmtP11E1>)
    return "p11e1";
  if constexpr (std::is_same_v<Fmt, FmtP11E2>)
    return "p11e2";
  if constexpr (std::is_same_v<Fmt, FmtP12E0>)
    return "p12e0";
  if constexpr (std::is_same_v<Fmt, FmtP12E1>)
    return "p12e1";
  if constexpr (std::is_same_v<Fmt, FmtP12E2>)
    return "p12e2";
  if constexpr (std::is_same_v<Fmt, FmtP13E0>)
    return "p13e0";
  if constexpr (std::is_same_v<Fmt, FmtP13E1>)
    return "p13e1";
  if constexpr (std::is_same_v<Fmt, FmtP13E2>)
    return "p13e2";
  if constexpr (std::is_same_v<Fmt, FmtP14E0>)
    return "p14e0";
  if constexpr (std::is_same_v<Fmt, FmtP14E1>)
    return "p14e1";
  if constexpr (std::is_same_v<Fmt, FmtP14E2>)
    return "p14e2";
  if constexpr (std::is_same_v<Fmt, FmtP15E0>)
    return "p15e0";
  if constexpr (std::is_same_v<Fmt, FmtP15E1>)
    return "p15e1";
  if constexpr (std::is_same_v<Fmt, FmtP15E2>)
    return "p15e2";
  if constexpr (std::is_same_v<Fmt, FmtP16E0>)
    return "p16e0";
  if constexpr (std::is_same_v<Fmt, FmtP16E1>)
    return "p16e1";
  if constexpr (std::is_same_v<Fmt, FmtP16E2>)
    return "p16e2";
  if constexpr (std::is_same_v<Fmt, FmtP32E0>)
    return "p32e0";
  if constexpr (std::is_same_v<Fmt, FmtP32E1>)
    return "p32e1";
  if constexpr (std::is_same_v<Fmt, FmtP32E2>)
    return "p32e2";
  return "unknown";
}

template <typename Fmt>
static inline int qalignExperimentalGpEnabledForFmt() {
  if constexpr (std::is_same_v<Fmt, FmtP8E0>)
    return positExperimentalGPEnabled<8, 0>() ? 1 : 0;
  if constexpr (std::is_same_v<Fmt, FmtP8E1>)
    return positExperimentalGPEnabled<8, 1>() ? 1 : 0;
  if constexpr (std::is_same_v<Fmt, FmtP8E2>)
    return positExperimentalGPEnabled<8, 2>() ? 1 : 0;
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
  if constexpr (std::is_same_v<Fmt, FmtP8E1ViaPX1>)
    return positExperimentalGPEnabled<8, 1>() ? 1 : 0;
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
  if constexpr (std::is_same_v<Fmt, FmtP8E2ViaPX2>)
    return positExperimentalGPEnabled<8, 2>() ? 1 : 0;
#endif
#endif
  return 0;
}

static inline std::string positDotProbeFilePath() {
  if (const char *e = std::getenv("POSIT_DOT_PROBE_FILE")) return std::string(e);
  if (const char *e = std::getenv("POSIT_DOT_PROBE")) {
    std::string s(e); for (char &ch:s) ch=static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (!(s=="0" || s=="off" || s=="false" || s=="no")) return std::string("/tmp/posit_dot_probe.csv");
  }
  return std::string();
}
static inline bool positDotProbeEnabled(){ return !positDotProbeFilePath().empty(); }
static inline uint64_t positDotProbeLimit(){ return positParsePositiveU64Env("POSIT_DOT_PROBE_LIMIT", 256); }
static inline bool positDotProbeAppendHeader(const std::string &path){ std::ifstream fin(path, std::ios::binary|std::ios::ate); return !fin.good() || fin.tellg()==std::streampos(0); }
static void dumpDotProbeAtExit(){
  if(!positDotProbeEnabled()) return;
  std::string outPath=positDotProbeFilePath(); bool writeHeader=positDotProbeAppendHeader(outPath);
  std::ofstream fout(outPath, std::ios::app);
  if(!fout.is_open()){ std::fprintf(stderr,"[DOTPROBE] failed to open dot probe file: %s\n",outPath.c_str()); return; }
  if(writeHeader) fout << "pid,op,format,effective_math_path,output_encoded_format,key,dot_mode,use_quire,use_f32_math,use_p16_mixed,use_gp_metadata,experimental_gp_allowed,dim0,dim1,dim2,dim3,k,output_elements\n";
#if defined(__linux__)
  long pid=static_cast<long>(getpid());
#else
  long pid=0;
#endif
  std::lock_guard<std::mutex> lock(gDotProbeMutex);
  for(const auto &row:gDotProbeRows) fout << pid << "," << row.op << "," << row.format << "," << row.effectiveMathPath << "," << row.outputEncodedFormat << "," << row.key << "," << row.dotMode << "," << row.useQuire << "," << row.useF32Math << "," << row.useP16Mixed << "," << row.useGPMetadata << "," << row.experimentalGPAllowed << "," << row.dim0 << "," << row.dim1 << "," << row.dim2 << "," << row.dim3 << "," << row.k << "," << row.outputElements << "\n";
}
static inline void ensureDotProbeRegistration(){ static std::once_flag once; if(positDotProbeEnabled()) std::call_once(once,[](){std::atexit(dumpDotProbeAtExit);}); }
template <typename Fmt>
static inline void recordDotProbe(const char *op,int64_t qalignKey,const std::string &dotMode,bool useQuire,bool useF32Math,bool useP16Mixed,bool useGPMetadata,int64_t dim0,int64_t dim1,int64_t dim2,int64_t dim3,int64_t k,int64_t outputElements){
  if(!positDotProbeEnabled()) return; ensureDotProbeRegistration();
  std::lock_guard<std::mutex> lock(gDotProbeMutex); if(gDotProbeRows.size()>=positDotProbeLimit()) return;
  DotProbeRow row; row.op=op; row.format=qalignFormatName<Fmt>(); row.effectiveMathPath=dotMode; row.outputEncodedFormat=qalignFormatName<Fmt>(); row.key=qalignKey; row.dotMode=dotMode; row.useQuire=useQuire?1:0; row.useF32Math=useF32Math?1:0; row.useP16Mixed=useP16Mixed?1:0; row.useGPMetadata=useGPMetadata?1:0; row.experimentalGPAllowed=qalignExperimentalGpEnabledForFmt<Fmt>(); row.dim0=dim0; row.dim1=dim1; row.dim2=dim2; row.dim3=dim3; row.k=k; row.outputElements=outputElements; gDotProbeRows.push_back(std::move(row));
}

template <typename Fmt>
static inline int qalignExperimentalGpRsForFmt() {
  if constexpr (std::is_same_v<Fmt, FmtP8E0>)
    return positExperimentalGPTable<8, 0>().rs;
  if constexpr (std::is_same_v<Fmt, FmtP8E1>)
    return positExperimentalGPTable<8, 1>().rs;
  if constexpr (std::is_same_v<Fmt, FmtP8E2>)
    return positExperimentalGPTable<8, 2>().rs;
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
  if constexpr (std::is_same_v<Fmt, FmtP8E1ViaPX1>)
    return positExperimentalGPTable<8, 1>().rs;
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
  if constexpr (std::is_same_v<Fmt, FmtP8E2ViaPX2>)
    return positExperimentalGPTable<8, 2>().rs;
#endif
#endif
  return 7;
}

template <typename Fmt>
static inline int qalignExperimentalGpScForFmt() {
  if constexpr (std::is_same_v<Fmt, FmtP8E0>)
    return positExperimentalGPTable<8, 0>().sc;
  if constexpr (std::is_same_v<Fmt, FmtP8E1>)
    return positExperimentalGPTable<8, 1>().sc;
  if constexpr (std::is_same_v<Fmt, FmtP8E2>)
    return positExperimentalGPTable<8, 2>().sc;
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
  if constexpr (std::is_same_v<Fmt, FmtP8E1ViaPX1>)
    return positExperimentalGPTable<8, 1>().sc;
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
  if constexpr (std::is_same_v<Fmt, FmtP8E2ViaPX2>)
    return positExperimentalGPTable<8, 2>().sc;
#endif
#endif
  return 0;
}

template <typename Fmt>
static inline constexpr bool fmtSupportsPerTensorGP() {
  return std::is_same_v<Fmt, FmtP8E0> || std::is_same_v<Fmt, FmtP8E1> ||
         std::is_same_v<Fmt, FmtP8E2>
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
         || std::is_same_v<Fmt, FmtP8E1ViaPX1>
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
         || std::is_same_v<Fmt, FmtP8E2ViaPX2>
#endif
#endif
      ;
}

#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_MIN
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_MIN 0.25
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_MAX
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_MAX 4.0
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_STEPS
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_STEPS 9
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GAMMA_TARGET
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GAMMA_TARGET 1.0
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GAMMA_PERCENTILE
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GAMMA_PERCENTILE 0.95
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_MIN_GAIN
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_MIN_GAIN 0.0
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_MAX_SAMPLES
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_MAX_SAMPLES 4096
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8 ""
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8 ""
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E0
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E0 ""
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E0
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E0 ""
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E1
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E1 ""
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E1
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E1 ""
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E2
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E2 ""
#endif
#ifndef POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E2
#define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E2 ""
#endif

enum class PositRuntimeOutputAlpsMode {
  Off = 0,
  Sampled = 1,
  Full = 2,
  Offline = 3,
};

static inline PositRuntimeOutputAlpsMode positRuntimeOutputAlpsMode() {
#if defined(POSIT_RUNTIME_OUTPUT_ALPS_FULL)
  return PositRuntimeOutputAlpsMode::Full;
#elif defined(POSIT_RUNTIME_OUTPUT_ALPS_OFFLINE)
  return PositRuntimeOutputAlpsMode::Offline;
#elif defined(POSIT_RUNTIME_OUTPUT_ALPS_SAMPLED)
  return PositRuntimeOutputAlpsMode::Sampled;
#else
  return PositRuntimeOutputAlpsMode::Off;
#endif
}

static inline bool positRuntimeOutputAlpsDebugEnabled() {
  static bool enabled =
      positParseBoolEnvWithFallback("POSIT_RUNTIME_OUTPUT_ALPS_DEBUG", false);
  return enabled;
}

static inline uint64_t positRuntimeOutputAlpsDebugLimit() {
  static uint64_t v =
      positParsePositiveU64Env("POSIT_RUNTIME_OUTPUT_ALPS_DEBUG_LIMIT", 64);
  return v;
}

static inline const char *positRuntimeOutputAlpsModeName() {
  switch (positRuntimeOutputAlpsMode()) {
  case PositRuntimeOutputAlpsMode::Sampled:
    return "sampled";
  case PositRuntimeOutputAlpsMode::Full:
    return "full";
  case PositRuntimeOutputAlpsMode::Offline:
    return "offline";
  default:
    return "off";
  }
}

static inline bool positRuntimeOutputAlpsDebugShouldEmit() {
  static std::mutex mu;
  static uint64_t emitted = 0;
  if (!positRuntimeOutputAlpsDebugEnabled())
    return false;
  std::lock_guard<std::mutex> lock(mu);
  if (emitted >= positRuntimeOutputAlpsDebugLimit())
    return false;
  ++emitted;
  return true;
}

static inline double clamp01Runtime(double v) {
  if (!std::isfinite(v))
    return 0.95;
  if (v < 0.0)
    return 0.0;
  if (v > 1.0)
    return 1.0;
  return v;
}

static inline std::vector<int> parseIntCsvValuesRuntime(
    const std::string &csv) {
  std::vector<int> vals;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok = trimAscii(tok);
    if (tok.empty())
      continue;
    char *end = nullptr;
    long v = std::strtol(tok.c_str(), &end, 10);
    if (!end || *end != '\0')
      continue;
    vals.push_back(static_cast<int>(v));
  }
  std::sort(vals.begin(), vals.end());
  vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
  return vals;
}

template <typename Fmt>
static inline constexpr bool fmtSupportsRuntimeOutputAlps() {
  // Runtime/offline output ALPS (activation companding) is supported for all
  // sub-16-bit universal posit formats (p4..p15). The encode/decode path is
  // format-generic (asinh companding + Fmt bit decode), so any instantiated
  // UniversalFmt with nbits in [4,15] qualifies. (Previously hard-limited to p8.)
  if constexpr (requires { Fmt::nbits; }) {
    return Fmt::nbits >= 4 && Fmt::nbits <= 15;
  } else {
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
    if constexpr (std::is_same_v<Fmt, FmtP8E1ViaPX1>)
      return true;
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
    if constexpr (std::is_same_v<Fmt, FmtP8E2ViaPX2>)
      return true;
#endif
#endif
    return false;
  }
}

template <typename Fmt>
static inline const char *runtimeOutputAlpsDefaultRsCsvForFmt() {
  if constexpr (std::is_same_v<Fmt, FmtP8E0>)
    return (*POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E0)
               ? POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E0
               : POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8;
  if constexpr (std::is_same_v<Fmt, FmtP8E1>
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
                || std::is_same_v<Fmt, FmtP8E1ViaPX1>
#endif
#endif
  )
    return (*POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E1)
               ? POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E1
               : POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8;
  return (*POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E2)
             ? POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E2
             : POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8;
}

template <typename Fmt>
static inline const char *runtimeOutputAlpsDefaultScCsvForFmt() {
  if constexpr (std::is_same_v<Fmt, FmtP8E0>)
    return (*POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E0)
               ? POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E0
               : POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8;
  if constexpr (std::is_same_v<Fmt, FmtP8E1>
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
                || std::is_same_v<Fmt, FmtP8E1ViaPX1>
#endif
#endif
  )
    return (*POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E1)
               ? POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E1
               : POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8;
  return (*POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E2)
             ? POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E2
             : POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8;
}

template <typename Fmt>
static inline std::vector<int> runtimeOutputAlpsRsValuesForFmt() {
  if constexpr (!fmtSupportsRuntimeOutputAlps<Fmt>())
    return {};
  std::string envCsv;
  if constexpr (std::is_same_v<Fmt, FmtP8E0>) {
    if (const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8E0"); e && *e)
      envCsv = e;
  } else if constexpr (std::is_same_v<Fmt, FmtP8E1>
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
                        || std::is_same_v<Fmt, FmtP8E1ViaPX1>
#endif
#endif
  ) {
    if (const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8E1"); e && *e)
      envCsv = e;
  } else {
    if (const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8E2"); e && *e)
      envCsv = e;
  }
  if (envCsv.empty()) {
    if (const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8"); e && *e)
      envCsv = e;
  }
  if (envCsv.empty())
    envCsv = runtimeOutputAlpsDefaultRsCsvForFmt<Fmt>();
  auto vals = parseIntCsvValuesRuntime(envCsv);
  std::vector<int> out;
  for (int v : vals) {
    v = std::min(7, std::max(1, v));
    if (std::find(out.begin(), out.end(), v) == out.end())
      out.push_back(v);
  }
  return out;
}

template <typename Fmt>
static inline std::vector<int> runtimeOutputAlpsScValuesForFmt() {
  if constexpr (!fmtSupportsRuntimeOutputAlps<Fmt>())
    return {};
  std::string envCsv;
  if constexpr (std::is_same_v<Fmt, FmtP8E0>) {
    if (const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8E0"); e && *e)
      envCsv = e;
  } else if constexpr (std::is_same_v<Fmt, FmtP8E1>
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
                        || std::is_same_v<Fmt, FmtP8E1ViaPX1>
#endif
#endif
  ) {
    if (const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8E1"); e && *e)
      envCsv = e;
  } else {
    if (const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8E2"); e && *e)
      envCsv = e;
  }
  if (envCsv.empty()) {
    if (const char *e = std::getenv("POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8"); e && *e)
      envCsv = e;
  }
  if (envCsv.empty())
    envCsv = runtimeOutputAlpsDefaultScCsvForFmt<Fmt>();
  auto vals = parseIntCsvValuesRuntime(envCsv);
  std::vector<int> out;
  for (int v : vals) {
    v = std::min(16, std::max(-16, v));
    if (std::find(out.begin(), out.end(), v) == out.end())
      out.push_back(v);
  }
  return out;
}

static inline double runtimeOutputAlpsThetaMin() {
  return positParsePositiveDoubleEnv(
      "POSIT_RUNTIME_OUTPUT_ALPS_THETA_MIN",
      POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_MIN);
}
static inline double runtimeOutputAlpsThetaMax() {
  return positParsePositiveDoubleEnv(
      "POSIT_RUNTIME_OUTPUT_ALPS_THETA_MAX",
      POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_MAX);
}
static inline int runtimeOutputAlpsThetaSteps() {
  return std::max(
      1, parseIntEnvWithFallback("POSIT_RUNTIME_OUTPUT_ALPS_THETA_STEPS",
                                 POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_STEPS));
}
static inline double runtimeOutputAlpsGammaTarget() {
  return positParsePositiveDoubleEnv(
      "POSIT_RUNTIME_OUTPUT_ALPS_GAMMA_TARGET",
      POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GAMMA_TARGET);
}
static inline double runtimeOutputAlpsGammaPercentile() {
  return clamp01Runtime(positParseDoubleEnvWithFallback(
      "POSIT_RUNTIME_OUTPUT_ALPS_GAMMA_PERCENTILE",
      POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GAMMA_PERCENTILE));
}
static inline double runtimeOutputAlpsMinGain() {
  return positParseDoubleEnvWithFallback(
      "POSIT_RUNTIME_OUTPUT_ALPS_MIN_GAIN",
      POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_MIN_GAIN);
}
static inline uint64_t runtimeOutputAlpsMaxSamples() {
  return positParsePositiveU64Env(
      "POSIT_RUNTIME_OUTPUT_ALPS_MAX_SAMPLES",
      static_cast<uint64_t>(POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_MAX_SAMPLES));
}

static inline double runtimeOutputAlpsPercentileAbs(
    const std::vector<double> &xs, double pct) {
  if (xs.empty())
    return 1.0;
  std::vector<double> mags;
  mags.reserve(xs.size());
  for (double x : xs) {
    if (std::isfinite(x))
      mags.push_back(std::fabs(x));
  }
  if (mags.empty())
    return 1.0;
  std::sort(mags.begin(), mags.end());
  double pp = clamp01Runtime(pct);
  size_t idx = static_cast<size_t>(
      std::llround(pp * static_cast<double>(mags.size() - 1)));
  idx = std::min(idx, mags.size() - 1);
  return std::max(1e-12, mags[idx]);
}

template <typename Fmt>
static inline std::vector<double> runtimeOutputAlpsScoringValues(
    const std::vector<double> &fullValues) {
  if (fullValues.empty())
    return {};
  if (positRuntimeOutputAlpsMode() == PositRuntimeOutputAlpsMode::Full)
    return fullValues;
  uint64_t limit = runtimeOutputAlpsMaxSamples();
  if (limit == 0 || fullValues.size() <= limit)
    return fullValues;
  std::vector<double> out;
  out.reserve(static_cast<size_t>(limit));
  size_t stride = std::max<size_t>(1, fullValues.size() / static_cast<size_t>(limit));
  for (size_t i = 0; i < fullValues.size() && out.size() < static_cast<size_t>(limit);
       i += stride)
    out.push_back(fullValues[i]);
  if (out.empty())
    out.push_back(fullValues.front());
  return out;
}

static inline void ensureRuntimeOutputAlpsCalibRegistration();

template <typename Fmt>
static inline void collectRuntimeOutputAlpsValues(
    int64_t qalignKey, const std::vector<double> &fullValues) {
  if ((!positRuntimeOutputAlpsCollectEnabled() &&
       !positRuntimeOutputAlpsCalibEnabled()) ||
      qalignKey == 0 || fullValues.empty())
    return;
  ensureRuntimeOutputAlpsCollectRegistration();
  ensureRuntimeOutputAlpsCalibRegistration();
  RuntimeOutputAlpsBucketKey bucketKey{qalignFormatName<Fmt>(), qalignKey, -1};
  std::vector<double> sampled = fullValues;
  uint64_t perCall = positRuntimeOutputAlpsCollectPerCall();
  if (perCall > 0 && sampled.size() > perCall) {
    sampled = runtimeOutputAlpsScoringValues<Fmt>(fullValues);
    if (sampled.size() > perCall)
      sampled.resize(static_cast<size_t>(perCall));
  }
  uint64_t cap = positRuntimeOutputAlpsCollectMaxPerKey();
  std::lock_guard<std::mutex> lock(gRuntimeOutputAlpsCollectMutex);
  auto &bucket = gRuntimeOutputAlpsCollectBuckets[bucketKey];
  bucket.seen += sampled.size();
  for (double v : sampled) {
    if (!std::isfinite(v))
      continue;
    if (bucket.samples.size() >= cap)
      break;
    bucket.samples.push_back(static_cast<float>(v));
  }
}

static inline std::vector<std::string> splitCsvLineSimple(const std::string &line) {
  std::vector<std::string> cols;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ','))
    cols.push_back(trimAscii(tok));
  return cols;
}

static inline bool parseInt64Strict(const std::string &s, int64_t &out) {
  char *end = nullptr;
  long long v = std::strtoll(s.c_str(), &end, 10);
  if (!end || *end != '\0')
    return false;
  out = static_cast<int64_t>(v);
  return true;
}

static inline bool parseIntStrict(const std::string &s, int &out) {
  char *end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (!end || *end != '\0')
    return false;
  out = static_cast<int>(v);
  return true;
}

static inline bool parseDoubleStrict(const std::string &s, double &out) {
  char *end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  if (!end || *end != '\0' || !std::isfinite(v))
    return false;
  out = v;
  return true;
}

static inline RuntimeOutputAlpsOfflineRow makeRuntimeOutputAlpsOfflineDirectRow() {
  RuntimeOutputAlpsOfflineRow row;
  row.meta.valid = true;
  row.meta.enabled = false;
  row.meta.rs = 7;
  row.meta.sc = 0;
  row.meta.compandMode = 0;
  row.meta.theta = 1.0;
  row.meta.gamma = 0.0;
  return row;
}

static const std::unordered_map<RuntimeOutputAlpsBucketKey,
    RuntimeOutputAlpsOfflineRow, RuntimeOutputAlpsBucketKeyHash> &
getRuntimeOutputAlpsOfflineTable() {
  static std::unordered_map<RuntimeOutputAlpsBucketKey,
      RuntimeOutputAlpsOfflineRow, RuntimeOutputAlpsBucketKeyHash> table = []() {
    std::unordered_map<RuntimeOutputAlpsBucketKey,
        RuntimeOutputAlpsOfflineRow, RuntimeOutputAlpsBucketKeyHash> t;
    std::string path = positRuntimeOutputAlpsOfflineFilePath();
    if (path.empty())
      return t;
    std::ifstream fin(path);
    if (!fin.is_open()) {
      std::fprintf(stderr,
          "[runtime-output-alps] cannot open POSIT_RUNTIME_OUTPUT_ALPS_FILE=%s\n",
          path.c_str());
      return t;
    }
    std::string line;
    while (std::getline(fin, line)) {
      line = trimAscii(line);
      if (line.empty() || line[0] == '#')
        continue;
      auto cols = splitCsvLineSimple(line);
      if (cols.size() < 9)
        continue;
      if (cols[0] == "format")
        continue;
      RuntimeOutputAlpsBucketKey key;
      RuntimeOutputAlpsOfflineRow row = makeRuntimeOutputAlpsOfflineDirectRow();
      key.format = cols[0];
      if (!parseInt64Strict(cols[1], key.key))
        continue;
      if (!parseInt64Strict(cols[2], key.channel))
        continue;
      if (!parseIntStrict(cols[3], row.meta.compandMode))
        continue;
      if (!parseDoubleStrict(cols[4], row.meta.theta))
        continue;
      if (!parseDoubleStrict(cols[5], row.meta.gamma))
        continue;
      int gpEnabled = 0;
      if (!parseIntStrict(cols[6], gpEnabled))
        continue;
      row.meta.enabled = gpEnabled != 0;
      if (!parseIntStrict(cols[7], row.meta.rs))
        continue;
      if (!parseIntStrict(cols[8], row.meta.sc))
        continue;
      row.meta.valid = true;
      row.meta.qalignKey = key.key;
      if (cols.size() >= 10)
        parseDoubleStrict(cols[9], row.directScore);
      if (cols.size() >= 11)
        parseDoubleStrict(cols[10], row.chosenScore);
      t[key] = row;
    }
    return t;
  }();
  return table;
}

template <typename Fmt>
static inline std::optional<RuntimeOutputAlpsOfflineRow>
lookupRuntimeOutputAlpsOfflineRow(int64_t qalignKey) {
  if (!positRuntimeOutputAlpsOfflineEnabled() || qalignKey == 0)
    return std::nullopt;
  const auto &tab = getRuntimeOutputAlpsOfflineTable();
  RuntimeOutputAlpsBucketKey key{qalignFormatName<Fmt>(), qalignKey, -1};
  auto it = tab.find(key);
  if (it == tab.end())
    return std::nullopt;
  return it->second;
}

static inline TensorGPMetadata emptyTensorGPMetadata() {
  return TensorGPMetadata{};
}

template <typename Fmt>
static inline TensorGPMetadata defaultTensorGPMetadataForFormat() {
  // Conservative default: do not globally reinterpret every p8 tensor as GP.
  // Per-layer GP is enabled only when a qalign CSV row explicitly carries
  // gp_rs/gp_sc and the format is allowed by POSIT_GP_EXPERIMENTAL_FORMATS.
  TensorGPMetadata meta;
  meta.valid = true;
  meta.enabled = false;
  meta.rs = 7;
  meta.sc = 0;
  meta.compandMode = 0;
  meta.theta = 1.0;
  meta.gamma = 0.0;
  return meta;
}

template <typename Fmt>
static inline TensorGPMetadata resolveTensorGPMetadataForQAlignKey(
    int64_t qalignKey) {
  if constexpr (!fmtSupportsPerTensorGP<Fmt>()) {
    return emptyTensorGPMetadata();
  } else {
    TensorGPMetadata meta = defaultTensorGPMetadataForFormat<Fmt>();
    meta.qalignKey = qalignKey;
    const auto &tab = getQAlignParamTable();
    if (qalignKey != 0 && tab.hasFile) {
      // CSV present means CSV is authoritative. If a key has no enabled GP row,
      // return a valid-but-disabled metadata object so downstream ops do not
      // inherit GP from earlier tensors.
      meta.valid = true;
      meta.enabled = false;
      auto it = tab.keyDefaults.find(qalignKey);
      if (it != tab.keyDefaults.end() && it->second.hasGp &&
          qalignExperimentalGpEnabledForFmt<Fmt>() != 0) {
        meta.enabled = true;
        meta.rs = it->second.gpRs;
        meta.sc = it->second.gpSc;
      }
      return meta;
    }
    return meta;
  }
}

template <typename Fmt>
static inline TensorGPMetadata chooseOutputTensorGPMetadata(
    int64_t qalignKey, const TensorGPMetadata &a = TensorGPMetadata{},
    const TensorGPMetadata &b = TensorGPMetadata{},
    const TensorGPMetadata &c = TensorGPMetadata{}) {
  TensorGPMetadata meta = resolveTensorGPMetadataForQAlignKey<Fmt>(qalignKey);
  if (qalignKey != 0 && meta.valid)
    return meta;
  if (a.valid && tensorHasSpecialDecodeMetadata(a))
    return a;
  if (b.valid && tensorHasSpecialDecodeMetadata(b))
    return b;
  if (c.valid && tensorHasSpecialDecodeMetadata(c))
    return c;
  return meta;
}

template <typename Fmt>
static inline typename Fmt::UIntT encodeTensorValueNoCompand(
    double x, const TensorGPMetadata &meta) {
  if constexpr (fmtSupportsPerTensorGP<Fmt>()) {
    if (meta.enabled) {
      if constexpr (std::is_same_v<Fmt, FmtP8E0>)
        return static_cast<typename Fmt::UIntT>(
            encodeApproxGeneralizedPosit8Runtime(
                x, getGP8RuntimeTableCustom<0>(meta.rs, meta.sc)));
      if constexpr (std::is_same_v<Fmt, FmtP8E1>
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
                    || std::is_same_v<Fmt, FmtP8E1ViaPX1>
#endif
#endif
      )
        return static_cast<typename Fmt::UIntT>(
            encodeApproxGeneralizedPosit8Runtime(
                x, getGP8RuntimeTableCustom<1>(meta.rs, meta.sc)));
      return static_cast<typename Fmt::UIntT>(
          encodeApproxGeneralizedPosit8Runtime(
              x, getGP8RuntimeTableCustom<2>(meta.rs, meta.sc)));
    }
  }
  return bits_from_double_fmt<Fmt>(x);
}

template <typename Fmt>
static inline double decodeTensorValue(
    typename Fmt::UIntT bits, const TensorGPMetadata &meta) {
  if (tensorCompandIsAlps(meta)) {
    double yq = 0.0;
    if constexpr (fmtSupportsPerTensorGP<Fmt>()) {
      if (meta.enabled) {
        if constexpr (std::is_same_v<Fmt, FmtP8E0>)
          yq = getGP8RuntimeTableCustom<0>(meta.rs, meta.sc).decode[bits];
        else if constexpr (std::is_same_v<Fmt, FmtP8E1>
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
                           || std::is_same_v<Fmt, FmtP8E1ViaPX1>
#endif
#endif
        )
          yq = getGP8RuntimeTableCustom<1>(meta.rs, meta.sc).decode[bits];
        else
          yq = getGP8RuntimeTableCustom<2>(meta.rs, meta.sc).decode[bits];
      } else {
        yq = double_from_bits_fmt<Fmt>(bits);
      }
    } else {
      yq = double_from_bits_fmt<Fmt>(bits);
    }
    return qalignDecompandAlps(meta.gamma * yq, meta.theta);
  }
  if constexpr (fmtSupportsPerTensorGP<Fmt>()) {
    if (meta.enabled) {
      if constexpr (std::is_same_v<Fmt, FmtP8E0>)
        return getGP8RuntimeTableCustom<0>(meta.rs, meta.sc).decode[bits];
      if constexpr (std::is_same_v<Fmt, FmtP8E1>
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
                    || std::is_same_v<Fmt, FmtP8E1ViaPX1>
#endif
#endif
      )
        return getGP8RuntimeTableCustom<1>(meta.rs, meta.sc).decode[bits];
      return getGP8RuntimeTableCustom<2>(meta.rs, meta.sc).decode[bits];
    }
  }
  return double_from_bits_fmt<Fmt>(bits);
}

template <typename Fmt>
static inline typename Fmt::UIntT encodeTensorValue(
    double x, const TensorGPMetadata &meta) {
  if (tensorCompandIsAlps(meta)) {
    double y = qalignCompandAlps(x, meta.theta) / meta.gamma;
    if (!std::isfinite(y))
      y = 0.0;
    return encodeTensorValueNoCompand<Fmt>(y, meta);
  }
  return encodeTensorValueNoCompand<Fmt>(x, meta);
}

static void loadPositOutputClampTable() {
  const char *path = std::getenv("POSIT_OUTPUT_CLAMP_FILE");
  if (!path || !*path) return;
  std::ifstream f(path);
  if (!f.is_open()) { fprintf(stderr, "[posit_clamp] cannot open %s\n", path); return; }
  std::string line;
  while (std::getline(f, line)) {
    // Strip inline comments (anything after '#' or whitespace+comment)
    auto hpos = line.find('#');
    if (hpos != std::string::npos) line = line.substr(0, hpos);
    if (line.empty()) continue;
    // Parse: key,lo,hi[,theta[,gamma]]
    int64_t key; float lo, hi, theta = 0.f, gamma = 1.f;
    int parsed = sscanf(line.c_str(), "%" PRId64 ",%f,%f,%f,%f",
                        &key, &lo, &hi, &theta, &gamma);
    if (parsed >= 3)
      gPositOutputClampTable[key] = {lo, hi, theta, gamma};
  }
  int vb = 0, va = 0;
  for (auto &kv : gPositOutputClampTable)
    (kv.second.theta > 0.f ? vb : va)++;
  fprintf(stderr, "[posit_clamp] loaded %zu entries from %s (versionA=%d versionB=%d)\n",
          gPositOutputClampTable.size(), path, va, vb);
}

static inline const PositOutputClampEntry *lookupPositOutputClamp(int64_t qalignKey) {
  std::call_once(gPositOutputClampLoadFlag, loadPositOutputClampTable);
  if (qalignKey == 0 || gPositOutputClampTable.empty()) return nullptr;
  auto it = gPositOutputClampTable.find(qalignKey);
  return it != gPositOutputClampTable.end() ? &it->second : nullptr;
}

// Version A: clamp then re-encode with the tensor's existing ALPS/GP metadata.
// Version B: clamp then re-encode with a new ALPS theta from the clamp entry.
// Version is selected per-entry: if clamp.theta > 0 → Version B.
template <typename Fmt>
static void applyPositOutputClamp(DynamicMemRefType<typename Fmt::MemT> &t,
                                  const PositOutputClampEntry &clamp) {
  TensorGPMetadata decodeMeta = lookupTensorGPMetadata(tensorMetaPtr(t));
  int64_t n = num_elems(t);
  double lo = static_cast<double>(clamp.lo);
  double hi = static_cast<double>(clamp.hi);

  // Version B: build a new TensorGPMetadata with the calibrated theta/gamma for re-encoding.
  const bool versionB = (clamp.theta > 0.f);
  TensorGPMetadata encodeMeta = decodeMeta;
  if (versionB) {
    encodeMeta = TensorGPMetadata{};  // reset GP, use ALPS only
    encodeMeta.compandMode = 1;       // kQAlignCompandAlps
    encodeMeta.theta = static_cast<double>(clamp.theta);
    encodeMeta.gamma = static_cast<double>(clamp.gamma > 0.f ? clamp.gamma : 1.f);
    encodeMeta.enabled = false;       // no GP override in re-encode path
  }

  for (int64_t i = 0; i < n; ++i) {
    auto raw = to_bits(t.data[offset_of(t, i)]);
    double v = decodeTensorValue<Fmt>(raw, decodeMeta);
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    t.data[offset_of(t, i)] = from_bits<typename Fmt::MemT>(encodeTensorValue<Fmt>(v, encodeMeta));
  }

  // Version B re-encodes with a new theta/gamma. The producing kernel already
  // registered its own outMeta for this tensor pointer; downstream ops decode
  // via lookupTensorGPMetadata(tensorMetaPtr(input)). We must update the
  // registration so the next op decodes these re-encoded bits with the SAME
  // metadata they were encoded with. Without this, the next conv/gemm/to_f32
  // would decode with the stale outMeta → wrong values.
  if (versionB)
    registerTensorGPMetadata(tensorMetaPtr(t), encodeMeta);
}

// --- Layer range collection helpers ---

static void initLayerRangeCollection() {
  const char *path = std::getenv("POSIT_COLLECT_LAYER_RANGES_FILE");
  if (!path || !*path) return;
  gLayerRangeEnabled = true;
  gLayerRangeOutputPath = path;
  fprintf(stderr, "[posit_collect] layer range collection enabled → %s\n", path);
}

// Reservoir-sample decoded activation values for later percentile computation.
// opLabel is stored on first call for this key (e.g., "conv2d", "gemm3d").
template <typename Fmt>
static void collectLayerRangeValues(int64_t qalignKey, const char *opLabel,
                                    DynamicMemRefType<typename Fmt::MemT> &t) {
  std::call_once(gLayerRangeInitFlag, initLayerRangeCollection);
  if (!gLayerRangeEnabled || qalignKey == 0) return;

  TensorGPMetadata meta = lookupTensorGPMetadata(tensorMetaPtr(t));
  int64_t n = num_elems(t);

  std::lock_guard<std::mutex> lock(gLayerRangeMutex);
  auto &res = gLayerRangeTable[qalignKey];
  if (res.opLabel.empty()) res.opLabel = opLabel;

  for (int64_t i = 0; i < n; ++i) {
    auto raw = to_bits(t.data[offset_of(t, i)]);
    float v = static_cast<float>(decodeTensorValue<Fmt>(raw, meta));
    if (!std::isfinite(v)) continue;
    ++res.totalSeen;
    if (static_cast<int64_t>(res.samples.size()) < kLayerRangesReservoirSize) {
      res.samples.push_back(v);
    } else {
      // Vitter's Algorithm R: replace with probability k/n
      std::uniform_int_distribution<int64_t> dist(0, res.totalSeen - 1);
      int64_t j = dist(res.rng);
      if (j < kLayerRangesReservoirSize)
        res.samples[static_cast<size_t>(j)] = v;
    }
  }
}

static void writeLayerRangeCSV() {
  if (!gLayerRangeEnabled || gLayerRangeOutputPath.empty()) return;
  std::ofstream f(gLayerRangeOutputPath);
  if (!f.is_open()) {
    fprintf(stderr, "[posit_collect] cannot write ranges to %s\n",
            gLayerRangeOutputPath.c_str());
    return;
  }
  // Write RAW reservoir samples (mergeable across parts). One line per key:
  //   key,op,total_seen,n_samples,v0,v1,...
  // The Python merge step concatenates all parts per key and computes percentiles.
  f << "# key,op,total_seen,n_samples,values...\n";
  for (auto &kv : gLayerRangeTable) {
    auto &res = kv.second;
    if (res.samples.empty()) continue;
    f << kv.first << "," << res.opLabel << "," << res.totalSeen << ","
      << res.samples.size();
    for (float v : res.samples)
      f << "," << v;
    f << "\n";
  }
  fprintf(stderr, "[posit_collect] wrote %zu layer range entries (raw samples) to %s\n",
          gLayerRangeTable.size(), gLayerRangeOutputPath.c_str());
}

// Register writeLayerRangeCSV to run at program exit.
static const bool gLayerRangeAtExitRegistered = []() {
  std::atexit(writeLayerRangeCSV);
  return true;
}();

template <typename Fmt>
static inline double runtimeOutputAlpsScoreForMeta(
    const std::vector<double> &values, const TensorGPMetadata &meta) {
  // Score = NSR (noise-to-signal ratio) = sum((q(x)-x)^2) / sum(x^2) = 1/SQNR.
  // Lower is better (keeps the existing minimize + minGain logic), and
  // maximizing SQNR == minimizing NSR.
  //
  // Previously this was MAE = mean(|q(x)-x|), averaged over ALL values. For
  // ReLU6 activations (~98% zeros) the MAE is dominated by the many zeros, so
  // the search picks a theta that keeps zeros accurate but CRUSHES the few large
  // discriminative values -> features shrink -> accuracy collapse. NSR weights by
  // signal power (x^2), so the large discriminative values dominate the choice,
  // preserving them. (Matches the signal-aware metric used for ALPS calibration.)
  if (values.empty())
    return 0.0;
  long double errPow = 0.0L;
  long double sigPow = 0.0L;
  for (double x : values) {
    if (!std::isfinite(x))
      continue;
    double q = requantizeTensorValue<Fmt>(x, meta);
    long double d = static_cast<long double>(q) - static_cast<long double>(x);
    errPow += d * d;
    sigPow += static_cast<long double>(x) * static_cast<long double>(x);
  }
  if (sigPow <= 0.0L)
    return static_cast<double>(errPow);  // all-zero signal: fall back to abs error
  return static_cast<double>(errPow / sigPow);
}

template <typename Fmt>
static inline double runtimeOutputAlpsEstimateGamma(
    const std::vector<double> &values, double theta, double gammaTarget,
    double gammaPct) {
  std::vector<double> zs;
  zs.reserve(values.size());
  for (double x : values) {
    if (!std::isfinite(x))
      continue;
    double z = qalignCompandAlps(x, theta);
    if (std::isfinite(z))
      zs.push_back(z);
  }
  double denom = runtimeOutputAlpsPercentileAbs(zs, gammaPct);
  double target = (gammaTarget > 0.0) ? gammaTarget : 1.0;
  double gamma = denom / target;
  if (!(gamma > 0.0) || !std::isfinite(gamma))
    gamma = 1.0;
  return gamma;
}

template <typename Fmt>
static inline RuntimeOutputAlpsOfflineRow searchRuntimeOutputAlpsMetadata(
    const std::vector<double> &scoreValues, const TensorGPMetadata &seedMeta) {
  RuntimeOutputAlpsOfflineRow best;
  best.meta = seedMeta;
  best.meta.valid = true;
  best.meta.enabled = false;
  best.meta.rs = 7;
  best.meta.sc = 0;
  best.meta.compandMode = 0;
  best.meta.theta = 1.0;
  best.meta.gamma = 0.0;
  if (scoreValues.empty())
    return best;

  auto makeBaseMeta = [&](bool enableGp, int rs, int sc) {
    TensorGPMetadata m = seedMeta;
    m.valid = true;
    m.enabled = enableGp;
    m.rs = rs;
    m.sc = sc;
    m.compandMode = 0;
    m.theta = 1.0;
    m.gamma = 0.0;
    return m;
  };

  std::vector<TensorGPMetadata> directCandidates;
  directCandidates.push_back(makeBaseMeta(false, 7, 0));
  for (int rs : runtimeOutputAlpsRsValuesForFmt<Fmt>()) {
    for (int sc : runtimeOutputAlpsScValuesForFmt<Fmt>()) {
      bool dup = false;
      for (const auto &m : directCandidates) {
        if (m.enabled && m.rs == rs && m.sc == sc) {
          dup = true;
          break;
        }
      }
      if (!dup)
        directCandidates.push_back(makeBaseMeta(true, rs, sc));
    }
  }
  if (seedMeta.enabled) {
    bool dup = false;
    for (const auto &m : directCandidates) {
      if (m.enabled && m.rs == seedMeta.rs && m.sc == seedMeta.sc) {
        dup = true;
        break;
      }
    }
    if (!dup)
      directCandidates.push_back(makeBaseMeta(true, seedMeta.rs, seedMeta.sc));
  }

  TensorGPMetadata bestMeta = directCandidates.front();
  double bestDirectScore =
      runtimeOutputAlpsScoreForMeta<Fmt>(scoreValues, bestMeta);
  double bestScore = bestDirectScore;
  for (size_t i = 1; i < directCandidates.size(); ++i) {
    double score =
        runtimeOutputAlpsScoreForMeta<Fmt>(scoreValues, directCandidates[i]);
    if (score < bestDirectScore) {
      bestDirectScore = score;
      bestMeta = directCandidates[i];
      bestScore = score;
    }
  }

  const double thetaMin = runtimeOutputAlpsThetaMin();
  const double thetaMax = std::max(thetaMin, runtimeOutputAlpsThetaMax());
  const int thetaSteps = runtimeOutputAlpsThetaSteps();
  const double gammaTarget = runtimeOutputAlpsGammaTarget();
  const double gammaPct = runtimeOutputAlpsGammaPercentile();
  const double minGain = runtimeOutputAlpsMinGain();

  for (const auto &baseMeta : directCandidates) {
    for (int i = 0; i < thetaSteps; ++i) {
      double t = (thetaSteps == 1) ? 0.0
                                   : static_cast<double>(i) /
                                         static_cast<double>(thetaSteps - 1);
      double logTheta = std::log2(thetaMin) +
                        (std::log2(thetaMax) - std::log2(thetaMin)) * t;
      double theta = std::pow(2.0, logTheta);
      if (!(theta > 0.0) || !std::isfinite(theta))
        continue;
      double gamma = runtimeOutputAlpsEstimateGamma<Fmt>(
          scoreValues, theta, gammaTarget, gammaPct);
      TensorGPMetadata cand = baseMeta;
      cand.compandMode = 1;
      cand.theta = theta;
      cand.gamma = gamma;
      double score = runtimeOutputAlpsScoreForMeta<Fmt>(scoreValues, cand);
      if ((bestDirectScore - score) >= minGain && score < bestScore) {
        bestMeta = cand;
        bestScore = score;
      }
    }
  }

  best.meta = bestMeta;
  best.directScore = bestDirectScore;
  best.chosenScore = bestScore;
  return best;
}

template <typename Fmt>
static inline TensorGPMetadata chooseRuntimeOutputAlpsMetadata(
    int64_t qalignKey, const std::vector<double> &fullValues,
    const TensorGPMetadata &seedMeta) {
  if constexpr (!fmtSupportsRuntimeOutputAlps<Fmt>()) {
    return seedMeta;
  } else {
    if (positRuntimeOutputAlpsMode() == PositRuntimeOutputAlpsMode::Off ||
        fullValues.empty())
      return seedMeta;
    collectRuntimeOutputAlpsValues<Fmt>(qalignKey, fullValues);
    if (auto offline = lookupRuntimeOutputAlpsOfflineRow<Fmt>(qalignKey)) {
      if (positRuntimeOutputAlpsDebugShouldEmit()) {
        std::fprintf(stderr,
            "[runtime-output-alps] fmt=%s mode=%s key=%lld full_elems=%zu "
            "source=offline chosen_compand=%s gp=%d rs=%d sc=%d theta=%.9g "
            "gamma=%.9g direct_score=%.9g chosen_score=%.9g\n",
            qalignFormatName<Fmt>(), positRuntimeOutputAlpsModeName(),
            static_cast<long long>(qalignKey), fullValues.size(),
            tensorCompandIsAlps(offline->meta) ? "alps" : "direct",
            offline->meta.enabled ? 1 : 0, offline->meta.rs, offline->meta.sc,
            offline->meta.theta, offline->meta.gamma, offline->directScore,
            offline->chosenScore);
      }
      return offline->meta;
    }
    if (positRuntimeOutputAlpsMode() == PositRuntimeOutputAlpsMode::Offline)
      return seedMeta;
    std::vector<double> scoreValues =
        runtimeOutputAlpsScoringValues<Fmt>(fullValues);
    if (scoreValues.empty())
      return seedMeta;
    RuntimeOutputAlpsOfflineRow best =
        searchRuntimeOutputAlpsMetadata<Fmt>(scoreValues, seedMeta);
    if (positRuntimeOutputAlpsDebugShouldEmit()) {
      std::fprintf(stderr,
          "[runtime-output-alps] fmt=%s mode=%s key=%lld full_elems=%zu "
          "score_elems=%zu source=online direct_score=%.9g chosen_score=%.9g "
          "chosen_compand=%s gp=%d rs=%d sc=%d theta=%.9g gamma=%.9g\n",
          qalignFormatName<Fmt>(), positRuntimeOutputAlpsModeName(),
          static_cast<long long>(qalignKey), fullValues.size(),
          scoreValues.size(), best.directScore, best.chosenScore,
          tensorCompandIsAlps(best.meta) ? "alps" : "direct",
          best.meta.enabled ? 1 : 0, best.meta.rs, best.meta.sc,
          best.meta.theta, best.meta.gamma);
    }
    return best.meta;
  }
}

template <typename Fmt>
static inline double requantizeTensorValue(double x,
                                           const TensorGPMetadata &meta) {
  return decodeTensorValue<Fmt>(encodeTensorValue<Fmt>(x, meta), meta);
}

template <typename Fmt>
static inline bool emitRuntimeOutputAlpsCalibRowForFormat(
    std::ofstream &fout, const RuntimeOutputAlpsBucketKey &key,
    const RuntimeOutputAlpsCollectBucket &bucket) {
  if (key.format != qalignFormatName<Fmt>())
    return false;
  std::vector<double> values;
  values.reserve(bucket.samples.size());
  for (float v : bucket.samples) {
    if (std::isfinite(v))
      values.push_back(static_cast<double>(v));
  }
  TensorGPMetadata seed = defaultTensorGPMetadataForFormat<Fmt>();
  seed.valid = true;
  seed.qalignKey = key.key;
  RuntimeOutputAlpsOfflineRow best =
      searchRuntimeOutputAlpsMetadata<Fmt>(values, seed);
  fout << key.format << "," << key.key << "," << key.channel << ","
       << best.meta.compandMode << "," << best.meta.theta << ","
       << best.meta.gamma << "," << (best.meta.enabled ? 1 : 0) << ","
       << best.meta.rs << "," << best.meta.sc << "," << best.directScore
       << "," << best.chosenScore << "," << values.size() << "\n";
  return true;
}

static void dumpRuntimeOutputAlpsCalibToPath(const std::string &outPath) {
  std::ofstream fout(outPath);
  if (!fout.is_open()) {
    std::fprintf(stderr,
        "[runtime-output-alps] failed to open calib file: %s\n",
        outPath.c_str());
    return;
  }
  fout << "format,key,channel,compand_mode,theta,gamma,gp_enabled,gp_rs,gp_sc,"
          "direct_score,chosen_score,sample_count\n";
  std::lock_guard<std::mutex> lock(gRuntimeOutputAlpsCollectMutex);
  for (const auto &it : gRuntimeOutputAlpsCollectBuckets) {
    const auto &key = it.first;
    const auto &bucket = it.second;
    bool wrote = false;
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP8E0>(fout, key, bucket) || wrote;
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP8E1>(fout, key, bucket) || wrote;
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP8E2>(fout, key, bucket) || wrote;
#if defined(POSIT_USE_UNIVERSAL)
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP4E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP4E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP4E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP5E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP5E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP5E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP6E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP6E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP6E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP7E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP7E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP7E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP9E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP9E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP9E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP10E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP10E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP10E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP11E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP11E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP11E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP12E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP12E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP12E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP13E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP13E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP13E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP14E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP14E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP14E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP15E0>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP15E1>(fout, key, bucket) || wrote;  // ALPS p4-p15
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP15E2>(fout, key, bucket) || wrote;  // ALPS p4-p15
#endif
#if !defined(POSIT_USE_UNIVERSAL)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP8E1ViaPX1>(fout, key, bucket) || wrote;
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
    wrote = emitRuntimeOutputAlpsCalibRowForFormat<FmtP8E2ViaPX2>(fout, key, bucket) || wrote;
#endif
#endif
    (void)wrote;
  }
  fout.flush();
}

static void dumpRuntimeOutputAlpsCalibAtExit() {
  if (!positRuntimeOutputAlpsCalibEnabled())
    return;
  dumpRuntimeOutputAlpsCalibToPath(positRuntimeOutputAlpsCalibFilePath());
}

static inline void ensureRuntimeOutputAlpsCalibRegistration() {
  static bool registered = false;
  if (registered)
    return;
  if (positRuntimeOutputAlpsCalibEnabled()) {
    std::atexit(dumpRuntimeOutputAlpsCalibAtExit);
    registered = true;
  }
}

template <typename Fmt>
static inline void recordQAlignProbe(int sourceKind, int64_t qalignKey,
                                     int64_t channel, const QAlignParams &p,
                                     double origX, double int8XDQ,
                                     double qValue, double scale, double zp,
                                     double runtimeScaled, double runtimeYQ,
                                     double runtimeXDQ, double runtimeFinal,
                                     const char *flowKind = nullptr) {
  if (!positQAlignProbeEnabled())
    return;
  if (!positQAlignProbeWantsSource(sourceKind))
    return;
  if (!positQAlignProbeWantsKey(qalignKey))
    return;
  ensureQAlignProbeRegistration();
  uint64_t limit = positQAlignProbeLimitPerBucket();
  std::lock_guard<std::mutex> lock(gQAlignProbeMutex);
  QAlignBucketKey bucket{qalignKey, channel};
  auto &count = gQAlignProbeCounts[bucket];
  uint64_t &slot = (sourceKind == kQAlignProbeDQ) ? count.dq : count.orig;
  if (slot >= limit)
    return;
  QAlignProbeRow row;
  TensorGPMetadata probeMeta =
      p.hasGp ? TensorGPMetadata{true, true, p.gpRs, p.gpSc, qalignKey}
              : resolveTensorGPMetadataForQAlignKey<Fmt>(qalignKey);
  row.sourceKind = sourceKind;
  row.format = qalignFormatName<Fmt>();
  row.key = qalignKey;
  row.channel = channel;
  row.sampleIndex = slot++;
  row.theta = p.alpha;
  row.gamma = p.beta;
  row.compandMode = p.compandMode;
  row.qalignFound = p.foundInCsv ? 1 : 0;
  row.strictFallbackApplied = p.strictFallbackApplied ? 1 : 0;
  row.gpEnabled = probeMeta.enabled ? 1 : 0;
  row.gpRs = probeMeta.rs;
  row.gpSc = probeMeta.sc;
  row.origX = origX;
  row.int8XDQ = int8XDQ;
  row.qValue = qValue;
  row.scale = scale;
  row.zeroPoint = zp;
  row.runtimeScaled = runtimeScaled;
  row.runtimeYQ = runtimeYQ;
  row.runtimeXDQ = runtimeXDQ;
  row.runtimeFinal = runtimeFinal;
  std::string flow = flowKind ? std::string(flowKind) : std::string();
  if (sourceKind == kQAlignProbeFullRef) {
    row.chosenPath = (p.compandMode == kQAlignCompandAlps && p.beta > 0.0)
                         ? "alps_direct_fullref"
                         : "int8_qdq_fallback_fullref";
    row.qStoreKind = "int8_q_input";
    row.dqOutputKind = (flow == "dq_f32") ? "f32" : ((flow == "dq_posit_storage") ? "posit_storage" : "unknown");
    row.dqOutputWritten = (flow == "dq_f32") ? "runtime_x_dq" : ((flow == "dq_posit_storage") ? "runtime_final" : "unknown");
  } else if (sourceKind == kQAlignProbeDQ) {
    row.chosenPath = (p.compandMode == kQAlignCompandAlps && p.beta > 0.0)
                         ? "alps_direct_runtime_x_dq"
                         : "int8_qdq_fallback";
    row.qStoreKind = "int8_q_input";
    row.dqOutputKind = (flow == "dq_f32") ? "f32" : ((flow == "dq_posit_storage") ? "posit_storage" : "unknown");
    row.dqOutputWritten = (flow == "dq_f32") ? "runtime_x_dq" : ((flow == "dq_posit_storage") ? "runtime_final" : "unknown");
  } else {
    row.chosenPath = (p.compandMode == kQAlignCompandAlps && p.beta > 0.0)
                         ? "orig_alps_direct"
                         : "orig_identity_or_standard_posit";
    row.qStoreKind = (flow == "q_posit_storage") ? "posit_y_q_storage" : "unknown";
    row.dqOutputKind = "not_dq";
    row.dqOutputWritten = "not_dq";
  }
  gQAlignProbeRows.push_back(std::move(row));
}


static inline bool positQAlignAlpsTargetIsOrig() {
  // Default target is dq/int8_x_dq.  This keeps an existing INT8-QDQ model's
  // downstream activation distribution stable: ALPS is used to make posit better
  // approximate int8_x_dq, not to pull values back toward pre-quantization orig_x.
  //
  // To explicitly experiment with the older direct-from-orig behavior, set:
  //   QALIGN_ALPS_TARGET=orig
  // or
  //   POSIT_QALIGN_ALPS_TARGET=orig
  static int targetOrig = -1;
  if (targetOrig >= 0)
    return targetOrig == 1;
  const char *e = std::getenv("POSIT_QALIGN_ALPS_TARGET");
  if (!e || !*e)
    e = std::getenv("QALIGN_ALPS_TARGET");
  if (!e || !*e) {
    targetOrig = 0;
    return false;
  }
  std::string v(e);
  for (char &ch : v)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (v == "orig" || v == "origin" || v == "preq" || v == "pre_q" ||
      v == "from_orig") {
    targetOrig = 1;
  } else {
    targetOrig = 0;
  }
  return targetOrig == 1;
}

template <typename Fmt>
static inline double applyQAlignScaleSnap(
    double v, int64_t qalignKey, int64_t channel,
    int sourceKind = kQAlignProbeOrig,
    double qValue = std::numeric_limits<double>::quiet_NaN(),
    double scale = std::numeric_limits<double>::quiet_NaN(),
    double zp = std::numeric_limits<double>::quiet_NaN(),
    const char *flowKind = nullptr,
    double origRefX = std::numeric_limits<double>::quiet_NaN()) {
  if (!std::isfinite(v))
    return v;

  QAlignParams p = resolveQAlignParams<Fmt>(qalignKey, channel);
  applyStrictFallbackIfNeeded(p);
  double theta = (std::isfinite(p.alpha) && p.alpha > 0.0) ? p.alpha : 1.0;
  bool useCompand = (p.compandMode == kQAlignCompandAlps &&
                     std::isfinite(p.beta) && p.beta > 0.0);

  const double int8XDQInput = v;
  const bool directAlpsFromOrig =
      useCompand && sourceKind == kQAlignProbeDQ &&
      std::isfinite(origRefX) && positQAlignAlpsTargetIsOrig();
  const double qalignInput = directAlpsFromOrig ? origRefX : int8XDQInput;

  double runtimeScaled = qalignInput;
  double runtimeYQ = std::numeric_limits<double>::quiet_NaN();
  double runtimeXDQ = qalignInput;
  double runtimeFinal = qalignInput;

  if (useCompand) {
    // Paper-style ALPS compander. By default qalignInput is int8_x_dq,
    // matching the dq-target calibrator:
    //   int8_x_dq -> ALPS -> posit -> inverse ALPS.
    // If QALIGN_ALPS_TARGET=orig is explicitly set and an orig_ref is available,
    // qalignInput becomes orig_x for direct-from-orig experiments.
    runtimeScaled = theta * qalignInput;
    double y = qalignCompandAlps(qalignInput, theta) / p.beta;
    if (!std::isfinite(y))
      return int8XDQInput;
    typename Fmt::UIntT yBits = bits_from_double_fmt<Fmt>(y);
    double yq = double_from_bits_fmt<Fmt>(yBits);
    double back = qalignDecompandAlps(p.beta * yq, theta);
    runtimeYQ = yq;
    runtimeXDQ = std::isfinite(back) ? back : int8XDQInput;
  } else {
#if defined(POSIT_RUNTIME_QALIGN_ALPS_ONLY)
    runtimeScaled = qalignInput;
    runtimeYQ = double_from_bits_fmt<Fmt>(bits_from_double_fmt<Fmt>(qalignInput));
    runtimeXDQ = qalignInput;
#else
    runtimeScaled = theta * qalignInput;
    if (!std::isfinite(runtimeScaled))
      runtimeScaled = qalignInput;
    if (std::fabs(theta - 1.0) < 1e-12) {
      runtimeYQ = double_from_bits_fmt<Fmt>(bits_from_double_fmt<Fmt>(qalignInput));
      runtimeXDQ = qalignInput;
    } else {
      double snapped =
          double_from_bits_fmt<Fmt>(bits_from_double_fmt<Fmt>(runtimeScaled));
      runtimeYQ = snapped;
      double back = snapped / theta;
      runtimeXDQ = std::isfinite(back) ? back : int8XDQInput;
    }
#endif
  }

  TensorGPMetadata probeMeta =
      p.hasGp ? TensorGPMetadata{true, true, p.gpRs, p.gpSc, qalignKey}
              : resolveTensorGPMetadataForQAlignKey<Fmt>(qalignKey);
  runtimeFinal = decodeTensorValue<Fmt>(
      encodeTensorValue<Fmt>(runtimeXDQ, probeMeta), probeMeta);

  recordQAlignProbe<Fmt>(
      sourceKind, qalignKey, channel, p,
      sourceKind == kQAlignProbeOrig ? int8XDQInput
                                     : std::numeric_limits<double>::quiet_NaN(),
      sourceKind == kQAlignProbeDQ ? int8XDQInput
                                   : std::numeric_limits<double>::quiet_NaN(),
      qValue, scale, zp, runtimeScaled, runtimeYQ, runtimeXDQ, runtimeFinal, flowKind);
  if (sourceKind == kQAlignProbeDQ && std::isfinite(origRefX)) {
    recordQAlignProbe<Fmt>(
        kQAlignProbeFullRef, qalignKey, channel, p, origRefX, int8XDQInput,
        qValue, scale, zp, runtimeScaled, runtimeYQ, runtimeXDQ,
        runtimeFinal, flowKind);
  }
  return runtimeXDQ;
}

template <typename Fmt>
static inline typename Fmt::UIntT qalignBitsFromDouble(double v, int64_t qalignKey,
                                                       int64_t channel = -1) {
  return bits_from_double_fmt<Fmt>(applyQAlignScaleSnap<Fmt>(v, qalignKey, channel));
}

template <typename Fmt>
static inline typename Fmt::UIntT qalignBitsFromRaw(typename Fmt::UIntT bits,
                                                    int64_t qalignKey,
                                                    int64_t channel = -1) {
  if (qalignKey == 0)
    return bits;
  double v = double_from_bits_fmt<Fmt>(bits);
  return qalignBitsFromDouble<Fmt>(v, qalignKey, channel);
}

static bool positQuireEnabledForP8() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  const char *e = std::getenv("POSIT_QUIRE_P8");
  if (!e || !*e) {
    enabled = 1;
    return true;
  }
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (s == "0" || s == "off" || s == "false" || s == "no")
    enabled = 0;
  else
    enabled = 1;
  return enabled == 1;
}

static bool positQuireEnabledForSmall() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  // Dedicated switch for extra low-bit formats.
  const char *e = std::getenv("POSIT_QUIRE_SMALL");
  // Backward-compatible: if not set, fall back to p8 switch.
  if (!e || !*e)
    e = std::getenv("POSIT_QUIRE_P8");
  if (!e || !*e) {
    enabled = 1;
    return true;
  }
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (s == "0" || s == "off" || s == "false" || s == "no")
    enabled = 0;
  else
    enabled = 1;
  return enabled == 1;
}

// Dot-product accumulator:
// - default: plain posit accumulate (acc += a*b in posit domain)
// - SoftPosit formats with quire: accumulate products in quire and convert once
//   at the end. This reduces rounding loss in GEMM/Conv inner loops.
template <typename Fmt>
struct DotAccumulator {
  using PositT = typename Fmt::PositT;
  struct State {
    PositT acc;
  };
  static inline State init() { return State{Fmt::fromDouble(0.0)}; }
  static inline void fdp(State &s, PositT a, PositT b) {
    s.acc = Fmt::add(s.acc, Fmt::mul(a, b));
  }
  static inline PositT finish(const State &s) { return s.acc; }
};

#if defined(POSIT_USE_UNIVERSAL)
template <typename PositT, typename QuireT>
static inline PositT finishUniversalQuire(const QuireT &q) {
  return sw::universal::quire_resolve(q);
}

template <>
struct DotAccumulator<FmtP8E0> {
  using PositT = typename FmtP8E0::PositT;
  using QuireT = sw::universal::quire<PositT, 20>;
  struct State {
    bool useQuire;
    QuireT q;
    PositT acc;
  };
  static inline State init() {
    return State{positQuireEnabledForP8() && !positExperimentalGPEnabled<8, 0>(),
                 QuireT(0), FmtP8E0::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.q += sw::universal::quire_mul(a, b);
      return;
    }
    s.acc = FmtP8E0::add(s.acc, FmtP8E0::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? finishUniversalQuire<PositT>(s.q) : s.acc;
  }
};

template <>
struct DotAccumulator<FmtP8E1> {
  using PositT = typename FmtP8E1::PositT;
  using QuireT = sw::universal::quire<PositT, 20>;
  struct State {
    bool useQuire;
    QuireT q;
    PositT acc;
  };
  static inline State init() {
    return State{positQuireEnabledForP8() && !positExperimentalGPEnabled<8, 1>(),
                 QuireT(0), FmtP8E1::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.q += sw::universal::quire_mul(a, b);
      return;
    }
    s.acc = FmtP8E1::add(s.acc, FmtP8E1::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? finishUniversalQuire<PositT>(s.q) : s.acc;
  }
};

template <>
struct DotAccumulator<FmtP8E2> {
  using PositT = typename FmtP8E2::PositT;
  using QuireT = sw::universal::quire<PositT, 20>;
  struct State {
    bool useQuire;
    QuireT q;
    PositT acc;
  };
  static inline State init() {
    return State{positQuireEnabledForP8() && !positExperimentalGPEnabled<8, 2>(),
                 QuireT(0), FmtP8E2::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.q += sw::universal::quire_mul(a, b);
      return;
    }
    s.acc = FmtP8E2::add(s.acc, FmtP8E2::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? finishUniversalQuire<PositT>(s.q) : s.acc;
  }
};

// Use quire accumulation for extra formats with promoted posit storage.
// - p4..p7 -> promote to posit<8,es> + quire<8,es,20>
// - p9     -> promote to posit<16,es> + quire<16,es,20>
// This keeps quire-based accumulation while avoiding unstable small-nbits
// quire template instantiations.
#define DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FMT_ALIAS, QNBITS, ESBITS)                                 \
  template <>                                                                                        \
  struct DotAccumulator<FMT_ALIAS> {                                                                 \
    using PositT = typename FMT_ALIAS::PositT;                                                       \
    using QPositT = sw::universal::posit<QNBITS, ESBITS>;                                            \
    using QuireT = sw::universal::quire<QPositT, 20>;                                                \
    struct State {                                                                                   \
      bool useQuire;                                                                                 \
      QuireT q;                                                                                      \
      PositT acc;                                                                                    \
    };                                                                                               \
    static inline State init() {                                                                     \
      return State{positQuireEnabledForSmall(), QuireT(0), FMT_ALIAS::fromDouble(0.0)};            \
    }                                                                                                \
    static inline void fdp(State &s, PositT a, PositT b) {                                          \
      if (s.useQuire) {                                                                              \
        QPositT qa = static_cast<double>(a);                                                         \
        QPositT qb = static_cast<double>(b);                                                         \
        s.q += sw::universal::quire_mul(qa, qb);                                                     \
        return;                                                                                      \
      }                                                                                              \
      s.acc = FMT_ALIAS::add(s.acc, FMT_ALIAS::mul(a, b));                                          \
    }                                                                                                \
    static inline PositT finish(const State &s) {                                                    \
      if (!s.useQuire)                                                                               \
        return s.acc;                                                                                \
      QPositT qOut = sw::universal::quire_resolve(s.q);                                              \
      return FMT_ALIAS::fromDouble(static_cast<double>(qOut));                                       \
    }                                                                                                \
  };

DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP4E0, 8, 0)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP4E1, 8, 1)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP4E2, 8, 2)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP4E3, 8, 3)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP5E0, 8, 0)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP5E1, 8, 1)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP5E2, 8, 2)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP5E3, 8, 3)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP6E0, 8, 0)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP6E1, 8, 1)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP6E2, 8, 2)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP6E3, 8, 3)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP7E0, 8, 0)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP7E1, 8, 1)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP7E2, 8, 2)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP7E3, 8, 3)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP9E0, 16, 0)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP9E1, 16, 1)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP9E2, 16, 2)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP9E3, 16, 3)

#undef DEFINE_UNIVERSAL_SMALL_DOT_ACCUM
#endif

#if !defined(POSIT_USE_UNIVERSAL)
template <>
struct DotAccumulator<FmtP8E0> {
  using PositT = typename FmtP8E0::PositT;
  struct State {
    bool useQuire;
    quire8_t q;
    PositT acc;
  };
  static inline State init() {
    return State{positQuireEnabledForP8() && !positExperimentalGPEnabled<8, 0>(),
                 q8Clr(), FmtP8E0::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.q = q8_fdp_add(s.q, a, b);
      return;
    }
    s.acc = FmtP8E0::add(s.acc, FmtP8E0::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? q8_to_p8(s.q) : s.acc;
  }
};

#if defined(POSIT_USE_SOFTPOSIT_PX1)
template <>
struct DotAccumulator<FmtP8E1ViaPX1> {
  using PositT = typename FmtP8E1ViaPX1::PositT;
  struct State {
    bool useQuire;
    long double qacc;
    PositT acc;
  };
  static inline State init() {
    return State{positQuireEnabledForP8() && !positExperimentalGPEnabled<8, 1>(),
                 0.0L, FmtP8E1ViaPX1::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.qacc += static_cast<long double>(FmtP8E1ViaPX1::toDouble(a)) *
                static_cast<long double>(FmtP8E1ViaPX1::toDouble(b));
      return;
    }
    s.acc = FmtP8E1ViaPX1::add(s.acc, FmtP8E1ViaPX1::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? FmtP8E1ViaPX1::fromDouble(static_cast<double>(s.qacc)) : s.acc;
  }
};
#endif

#if defined(POSIT_USE_SOFTPOSIT_PX2)
template <>
struct DotAccumulator<FmtP8E2ViaPX2> {
  using PositT = typename FmtP8E2ViaPX2::PositT;
  struct State {
    bool useQuire;
    quire_2_t q;
    PositT acc;
  };
  static inline State init() {
    return State{
        positQuireEnabledForP8() && !positExperimentalGPEnabled<8, 2>(),
        qX2Clr(), FmtP8E2ViaPX2::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.q = qX2_fdp_add(s.q, a, b);
      return;
    }
    s.acc = FmtP8E2ViaPX2::add(s.acc, FmtP8E2ViaPX2::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? qX2_to_pX2(s.q, 8) : s.acc;
  }
};
#endif
#endif

template <typename Fmt>
static inline bool dotAccumulatorWouldUseQuire() {
  auto st = DotAccumulator<Fmt>::init();
  if constexpr (requires { st.useQuire; }) return st.useQuire;
  else return false;
}

template <typename Fmt, typename LoadA, typename LoadB>
static inline typename Fmt::PositT dot_product_accumulate(
    int64_t K, LoadA &&loadA, LoadB &&loadB) {
  using Acc = DotAccumulator<Fmt>;
  auto st = Acc::init();
  for (int64_t k = 0; k < K; ++k)
    Acc::fdp(st, loadA(k), loadB(k));
  return Acc::finish(st);
}

template <typename Fmt>
static inline constexpr bool isLowPrecisionFmtForBuildMixed() {
  return std::is_same_v<Fmt, FmtP4E0> || std::is_same_v<Fmt, FmtP4E1> ||
         std::is_same_v<Fmt, FmtP4E2> || std::is_same_v<Fmt, FmtP4E3> ||
         std::is_same_v<Fmt, FmtP5E0> || std::is_same_v<Fmt, FmtP5E1> ||
         std::is_same_v<Fmt, FmtP5E2> || std::is_same_v<Fmt, FmtP5E3> ||
         std::is_same_v<Fmt, FmtP6E0> || std::is_same_v<Fmt, FmtP6E1> ||
         std::is_same_v<Fmt, FmtP6E2> || std::is_same_v<Fmt, FmtP6E3> ||
         std::is_same_v<Fmt, FmtP7E0> || std::is_same_v<Fmt, FmtP7E1> ||
         std::is_same_v<Fmt, FmtP7E2> || std::is_same_v<Fmt, FmtP7E3> ||
         std::is_same_v<Fmt, FmtP8E0> || std::is_same_v<Fmt, FmtP8E1> ||
         std::is_same_v<Fmt, FmtP8E2> ||
#if defined(POSIT_USE_SOFTPOSIT_PX1)
         std::is_same_v<Fmt, FmtP8E1ViaPX1> ||
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
         std::is_same_v<Fmt, FmtP8E2ViaPX2> ||
#endif
         std::is_same_v<Fmt, FmtP9E0> || std::is_same_v<Fmt, FmtP9E1> ||
         std::is_same_v<Fmt, FmtP9E2> || std::is_same_v<Fmt, FmtP9E3>;
}

template <typename Fmt>
static inline bool shouldUseBuildMixedDot(PositDotOpKind opKind) {
  if constexpr (!isLowPrecisionFmtForBuildMixed<Fmt>()) {
    return false;
  } else {
    switch (positBuildMixedMode()) {
    case PositBuildMixedMode::Off:
      return false;
    case PositBuildMixedMode::P16E2:
    case PositBuildMixedMode::P32E2:
      return true;
    case PositBuildMixedMode::Runtime:
      return positMixedPrecisionP16Enabled() &&
             positMixedPrecisionP16OpEnabled(opKind);
    }
  }
  return false;
}

template <typename Fmt, typename MixedFmt, typename LoadA, typename LoadB>
static inline typename Fmt::PositT dot_product_accumulate_mixed_fmt(
    int64_t K, LoadA &&loadA, LoadB &&loadB) {
  using MixedT = typename MixedFmt::PositT;
  MixedT acc = MixedFmt::fromDouble(0.0);
  for (int64_t k = 0; k < K; ++k) {
    auto a = loadA(k);
    auto b = loadB(k);
    MixedT am = MixedFmt::fromDouble(Fmt::toDouble(a));
    MixedT bm = MixedFmt::fromDouble(Fmt::toDouble(b));
    acc = MixedFmt::add(acc, MixedFmt::mul(am, bm));
  }
  return Fmt::fromDouble(MixedFmt::toDouble(acc));
}

template <typename Fmt, typename LoadA, typename LoadB>
static inline typename Fmt::PositT dot_product_accumulate_selected_mixed(
    int64_t K, LoadA &&loadA, LoadB &&loadB) {
  switch (positBuildMixedMode()) {
  case PositBuildMixedMode::P16E2:
    return dot_product_accumulate_mixed_fmt<Fmt, FmtP16E2>(
        K, std::forward<LoadA>(loadA), std::forward<LoadB>(loadB));
  case PositBuildMixedMode::P32E2:
    return dot_product_accumulate_mixed_fmt<Fmt, FmtP32E2>(
        K, std::forward<LoadA>(loadA), std::forward<LoadB>(loadB));
  case PositBuildMixedMode::Off:
  case PositBuildMixedMode::Runtime:
  default:
    return dot_product_accumulate_mixed_fmt<Fmt, FmtP16E1>(
        K, std::forward<LoadA>(loadA), std::forward<LoadB>(loadB));
  }
}

template <typename Fmt, typename LoadA, typename LoadB>
static inline typename Fmt::PositT dot_product_accumulate_safe(
    int64_t K, LoadA &&loadA, LoadB &&loadB,
    PositDotOpKind opKind = PositDotOpKind::Gemm) {
  if (shouldUseBuildMixedDot<Fmt>(opKind)) {
    try {
      return dot_product_accumulate_selected_mixed<Fmt>(
          K, std::forward<LoadA>(loadA), std::forward<LoadB>(loadB));
    } catch (...) {
      recordPositFallback(
          gPositMixedP16FallbackCount, "mixed_p16_fallback",
          "build/runtime mixed dot threw, fallback to base dot");
    }
  }
#if defined(POSIT_USE_UNIVERSAL)
  static std::atomic<bool> disableQuireForFmt{false};
  const bool autoDisable = positQuireAutoDisableEnabled();
  const bool skipQuireTry =
      autoDisable && disableQuireForFmt.load(std::memory_order_relaxed);
  if (!skipQuireTry) {
    try {
      return dot_product_accumulate<Fmt>(K, loadA, loadB);
    } catch (...) {
      // Fallback for universal quire runtime exceptions (e.g. NaR inputs).
      if (autoDisable) {
        disableQuireForFmt.store(true, std::memory_order_relaxed);
        recordPositFallback(gPositQuireFallbackCount, "quire_fallback",
                            "dot accumulate disabled quire after exception");
      } else {
        recordPositFallback(gPositQuireFallbackCount, "quire_fallback",
                            "dot accumulate switched to non-quire path");
      }
    }
  }
#endif
  auto acc = Fmt::fromDouble(0.0);
  bool opException = false;
  for (int64_t k = 0; k < K; ++k) {
    auto a = loadA(k);
    auto b = loadB(k);
#if defined(POSIT_USE_UNIVERSAL)
    try {
      acc = Fmt::add(acc, Fmt::mul(a, b));
    } catch (...) {
      // Keep previous accumulator on exceptional operands.
      if (!opException) {
        recordPositFallback(gPositQuireFallbackCount, "quire_fallback",
                            "non-quire multiply/add threw");
        opException = true;
      }
    }
#else
    acc = Fmt::add(acc, Fmt::mul(a, b));
#endif
  }
  return acc;
}

static bool positTraceEnabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char *e = std::getenv("POSIT_TRACE");
    enabled = (e && *e && std::string(e) != "0") ? 1 : 0;
  }
  return enabled == 1;
}

static int64_t positTraceMaxElems() {
  static int64_t v = -1;
  if (v >= 0)
    return v;
  const char *e = std::getenv("POSIT_TRACE_MAX_ELEMS");
  if (!e || !*e) {
    v = 8;
    return v;
  }
  char *end = nullptr;
  long long parsed = std::strtoll(e, &end, 10);
  if (!end || *end != '\0' || parsed <= 0)
    v = 8;
  else
    v = parsed;
  return v;
}

template <typename Fmt>
static void traceMemrefSummary(const char *opName,
                               const DynamicMemRefType<typename Fmt::MemT> &m) {
  if (!positTraceEnabled())
    return;

  TensorGPMetadata meta = lookupTensorGPMetadata(tensorMetaPtr(m));
  int64_t n = num_elems(m);
  double mn = std::numeric_limits<double>::infinity();
  double mx = -std::numeric_limits<double>::infinity();
  double sum = 0.0;
  double absmax = 0.0;
  int64_t k = std::min<int64_t>(n, positTraceMaxElems());
  std::vector<double> sample;
  sample.reserve(static_cast<size_t>(k));

  for (int64_t i = 0; i < n; ++i) {
    auto bits = to_bits(m.data[offset_of(m, i)]);
    double v = decodeTensorValue<Fmt>(bits, meta);
    mn = std::min(mn, v);
    mx = std::max(mx, v);
    sum += v;
    absmax = std::max(absmax, std::fabs(v));
    if (i < k)
      sample.push_back(v);
  }

  double mean = (n > 0) ? (sum / static_cast<double>(n)) : 0.0;
  std::fprintf(stderr, "[PTRACE] op=%s rank=%lld shape=[", opName,
               static_cast<long long>(m.rank));
  for (int i = 0; i < m.rank; ++i) {
    std::fprintf(stderr, "%lld%s", static_cast<long long>(m.sizes[i]),
                 (i + 1 == m.rank) ? "" : ",");
  }
  std::fprintf(stderr, "] n=%lld min=%g max=%g mean=%g absmax=%g sample=[",
               static_cast<long long>(n), mn, mx, mean, absmax);
  for (size_t i = 0; i < sample.size(); ++i) {
    std::fprintf(stderr, "%g%s", sample[i], (i + 1 == sample.size()) ? "" : ",");
  }
  std::fprintf(stderr, "]\n");
}

//===----------------------------------------------------------------------===//
// Generic kernels
//===----------------------------------------------------------------------===//

template <typename Fmt>
static void posit_from_f32_kernel(UnrankedMemRefType<float> *In,
                                  UnrankedMemRefType<typename Fmt::MemT> *Out,
                                  int64_t qalignKey) {
  DynamicMemRefType<float> in(*In);
  DynamicMemRefType<typename Fmt::MemT> out(*Out);
  if (!validate_memref_desc(in) || !validate_memref_desc(out))
    return;
  int64_t inElems = 0, outElems = 0;
  if (!num_elems_safe(in, inElems) || !num_elems_safe(out, outElems))
    return;
  TensorGPMetadata outMeta = chooseOutputTensorGPMetadata<Fmt>(qalignKey);
  if (inElems == 1 && outElems > 1) {
    double rawV = static_cast<double>(in.data[offset_of(in, 0)]);
    if (positQAlignCollectEnabled())
      collectQAlignBatch(kQAlignCollectOrig, qalignKey, std::vector<int64_t>{-1},
          std::vector<double>{rawV});
    double v =
        applyQAlignScaleSnap<Fmt>(rawV, qalignKey, -1, kQAlignProbeOrig,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(), "q_posit_storage");
    auto bits = encodeTensorValue<Fmt>(v, outMeta);
    for (int64_t i = 0; i < outElems; ++i)
      out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(bits);
    registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
    return;
  }
  if (!is_dense_contiguous_memref(in) || !is_dense_contiguous_memref(out)) {
    if (inElems > 0 && outElems > 0) {
      double rawV = static_cast<double>(in.data[in.offset]);
      if (positQAlignCollectEnabled())
        collectQAlignBatch(kQAlignCollectOrig, qalignKey, std::vector<int64_t>{-1},
            std::vector<double>{rawV});
      double v =
          applyQAlignScaleSnap<Fmt>(rawV, qalignKey, -1, kQAlignProbeOrig,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(), "q_posit_storage");
      auto bits = encodeTensorValue<Fmt>(v, outMeta);
      for (int64_t i = 0; i < outElems; ++i)
        out.data[out.offset + i] = from_bits<typename Fmt::MemT>(bits);
      registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
    }
    return;
  }
  int64_t n = std::min(inElems, outElems);
  bool collect = positQAlignCollectEnabled();
  uint64_t perCallBudget = collect ? positQAlignCollectPerCall() : 0;
  int64_t sampleStride = 1;
  if (collect && perCallBudget > 0 && n > static_cast<int64_t>(perCallBudget))
    sampleStride = std::max<int64_t>(1, n / static_cast<int64_t>(perCallBudget));
  std::vector<int64_t> sampleChannels;
  std::vector<double> sampleValues;
  std::vector<double> sampleOrigValues;
  if (collect && perCallBudget > 0) {
    sampleChannels.reserve(static_cast<size_t>(perCallBudget));
    sampleValues.reserve(static_cast<size_t>(perCallBudget));
    sampleOrigValues.reserve(static_cast<size_t>(perCallBudget));
  }
  // Method B: element-wise f32->posit encode, independent per element. Parallel
  // only when NOT collecting qalign samples (the sampleValues.push_back below is
  // not thread-safe); collection is off in normal inference. Bit-identical.
#pragma omp parallel for if(!collect) num_threads(positOmpThreadCount()) schedule(static)
  for (int64_t i = 0; i < n; ++i) {
    double rawV = static_cast<double>(in.data[in.offset + i]);
    if (collect && perCallBudget > 0 &&
        sampleValues.size() < static_cast<size_t>(perCallBudget) &&
        (i % sampleStride == 0)) {
      sampleChannels.push_back(-1);
      sampleValues.push_back(rawV);
    }
    double v =
        applyQAlignScaleSnap<Fmt>(rawV, qalignKey, -1, kQAlignProbeOrig,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(), "q_posit_storage");
    auto bits = encodeTensorValue<Fmt>(v, outMeta);
    out.data[out.offset + i] = from_bits<typename Fmt::MemT>(bits);
  }
  if (collect && !sampleValues.empty())
    collectQAlignBatch(kQAlignCollectOrig, qalignKey, sampleChannels, sampleValues);
  registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
}

template <typename Fmt>
static void posit_to_f32_kernel(UnrankedMemRefType<typename Fmt::MemT> *In,
                                UnrankedMemRefType<float> *Out) {
  DynamicMemRefType<typename Fmt::MemT> in(*In);
  DynamicMemRefType<float> out(*Out);
  TensorGPMetadata inMeta = lookupTensorGPMetadata(tensorMetaPtr(in));
  if (!validate_memref_desc(in) || !validate_memref_desc(out))
    return;
  int64_t inElems = 0, outElems = 0;
  if (!num_elems_safe(in, inElems) || !num_elems_safe(out, outElems))
    return;
  if (inElems == 1 && outElems > 1) {
    auto bits = to_bits(in.data[offset_of(in, 0)]);
    float v = static_cast<float>(decodeTensorValue<Fmt>(bits, inMeta));
    for (int64_t i = 0; i < outElems; ++i)
      out.data[offset_of(out, i)] = v;
    return;
  }
  if (!is_dense_contiguous_memref(in) || !is_dense_contiguous_memref(out)) {
    if (inElems > 0 && outElems > 0) {
      auto bits = to_bits(in.data[in.offset]);
      float v = static_cast<float>(decodeTensorValue<Fmt>(bits, inMeta));
      for (int64_t i = 0; i < outElems; ++i)
        out.data[out.offset + i] = v;
    }
    return;
  }
  int64_t n = std::min(inElems, outElems);
  // Method B: element-wise posit->f32 decode is independent per element (no shared
  // state). GPT-2 calls this very often on large tensors (f32<->posit bridge), so
  // it is a hot path. Bit-identical; threads from POSIT_OMP_THREADS.
#pragma omp parallel for num_threads(positOmpThreadCount()) schedule(static)
  for (int64_t i = 0; i < n; ++i) {
    auto bits = to_bits(in.data[in.offset + i]);
    out.data[out.offset + i] = static_cast<float>(decodeTensorValue<Fmt>(bits, inMeta));
  }
}

enum class ElemwiseOpKind {
  Add,
  Sub,
  Mul,
  Div,
};

template <typename Fmt>
static void elemwise_kernel(ElemwiseOpKind opKind,
                            UnrankedMemRefType<typename Fmt::MemT> *A,
                            UnrankedMemRefType<typename Fmt::MemT> *B,
                            UnrankedMemRefType<typename Fmt::MemT> *Out,
                            int64_t qalignKey) {
  DynamicMemRefType<typename Fmt::MemT> a(*A), b(*B), o(*Out);
  TensorGPMetadata aMeta = lookupTensorGPMetadata(tensorMetaPtr(a));
  TensorGPMetadata bMeta = lookupTensorGPMetadata(tensorMetaPtr(b));
  TensorGPMetadata outMeta =
      chooseOutputTensorGPMetadata<Fmt>(qalignKey, aMeta, bMeta);
  bool useMetaPath = tensorHasSpecialDecodeMetadata(outMeta) ||
                     tensorHasSpecialDecodeMetadata(aMeta) ||
                     tensorHasSpecialDecodeMetadata(bMeta);
  bool useMetaF32 = tensorMathUsesF32ForAlps(outMeta, aMeta, bMeta);
  const bool useRuntimeOutputAlps =
      useMetaPath &&
      positRuntimeOutputAlpsMode() != PositRuntimeOutputAlpsMode::Off &&
      fmtSupportsRuntimeOutputAlps<Fmt>();

  int64_t outElems = num_elems(o);
  int64_t outRank = o.rank;
  QAlignOutputSampleBuffer samples =
      makeQAlignOutputSampleBuffer(qalignKey, outElems);
  std::vector<int64_t> outIdx(static_cast<size_t>(std::max<int64_t>(outRank, 1)), 0);
  std::vector<double> deferredValues;
  if (useRuntimeOutputAlps)
    deferredValues.resize(static_cast<size_t>(std::max<int64_t>(outElems, 0)));

  auto offset_with_broadcast = [&](const DynamicMemRefType<typename Fmt::MemT> &m) {
    if (m.rank == 0)
      return m.offset;
    int64_t rankDelta = outRank - m.rank;
    int64_t off = m.offset;
    for (int64_t od = 0; od < outRank; ++od) {
      int64_t md = od - rankDelta;
      if (md < 0)
        continue;
      int64_t idx = outIdx[static_cast<size_t>(od)];
      if (m.sizes[md] == 1)
        idx = 0;
      off += idx * m.strides[md];
    }
    return off;
  };

  for (int64_t lin = 0; lin < outElems; ++lin) {
    if (outRank > 0) {
      int64_t rem = lin;
      for (int64_t d = outRank - 1; d >= 0; --d) {
        outIdx[static_cast<size_t>(d)] = rem % o.sizes[d];
        rem /= o.sizes[d];
      }
    }

    auto abit = to_bits(a.data[offset_with_broadcast(a)]);
    auto bbit = to_bits(b.data[offset_with_broadcast(b)]);
    if (useMetaPath) {
      if (useMetaF32) {
        float av = static_cast<float>(decodeTensorValue<Fmt>(abit, aMeta));
        float bv = static_cast<float>(decodeTensorValue<Fmt>(bbit, bMeta));
        float cv = 0.0f;
        switch (opKind) {
        case ElemwiseOpKind::Add:
          cv = av + bv;
          break;
        case ElemwiseOpKind::Sub:
          cv = av - bv;
          break;
        case ElemwiseOpKind::Mul:
          cv = av * bv;
          break;
        case ElemwiseOpKind::Div:
          cv = av / bv;
          break;
        }
        if (useRuntimeOutputAlps) {
          deferredValues[static_cast<size_t>(lin)] = static_cast<double>(cv);
        } else {
          auto outBits = encodeTensorValue<Fmt>(static_cast<double>(cv), outMeta);
          o.data[offset_of(o, lin)] = from_bits<typename Fmt::MemT>(outBits);
          samples.maybeAppend(lin, -1, static_cast<double>(cv),
                              decodeTensorValue<Fmt>(outBits, outMeta));
        }
      } else {
        double av = decodeTensorValue<Fmt>(abit, aMeta);
        double bv = decodeTensorValue<Fmt>(bbit, bMeta);
        double cv = 0.0;
        switch (opKind) {
        case ElemwiseOpKind::Add:
          cv = av + bv;
          break;
        case ElemwiseOpKind::Sub:
          cv = av - bv;
          break;
        case ElemwiseOpKind::Mul:
          cv = av * bv;
          break;
        case ElemwiseOpKind::Div:
          cv = av / bv;
          break;
        }
        if (useRuntimeOutputAlps) {
          deferredValues[static_cast<size_t>(lin)] = cv;
        } else {
          auto outBits = encodeTensorValue<Fmt>(cv, outMeta);
          o.data[offset_of(o, lin)] = from_bits<typename Fmt::MemT>(outBits);
          samples.maybeAppend(lin, -1, cv, decodeTensorValue<Fmt>(outBits, outMeta));
        }
      }
      continue;
    }
    double av = double_from_bits_fmt<Fmt>(abit);
    double bv = double_from_bits_fmt<Fmt>(bbit);
    auto pa = Fmt::fromRaw(abit);
    auto pb = Fmt::fromRaw(bbit);
    typename Fmt::PositT pc = pa;
    switch (opKind) {
    case ElemwiseOpKind::Add:
      pc = Fmt::add(pa, pb);
      break;
    case ElemwiseOpKind::Sub:
      pc = Fmt::sub(pa, pb);
      break;
    case ElemwiseOpKind::Mul:
      pc = Fmt::mul(pa, pb);
      break;
    case ElemwiseOpKind::Div:
      pc = Fmt::div(pa, pb);
      break;
    }
    auto outBits = Fmt::toRaw(pc);
    o.data[offset_of(o, lin)] = from_bits<typename Fmt::MemT>(outBits);
    double cv = 0.0;
    switch (opKind) {
    case ElemwiseOpKind::Add:
      cv = av + bv;
      break;
    case ElemwiseOpKind::Sub:
      cv = av - bv;
      break;
    case ElemwiseOpKind::Mul:
      cv = av * bv;
      break;
    case ElemwiseOpKind::Div:
      cv = av / bv;
      break;
    }
    samples.maybeAppend(lin, -1, cv, double_from_bits_fmt<Fmt>(outBits));
  }
  if (useRuntimeOutputAlps) {
    outMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, outMeta);
    for (int64_t lin = 0; lin < outElems; ++lin) {
      double cv = deferredValues[static_cast<size_t>(lin)];
      auto outBits = encodeTensorValue<Fmt>(cv, outMeta);
      o.data[offset_of(o, lin)] = from_bits<typename Fmt::MemT>(outBits);
      samples.maybeAppend(lin, -1, cv, decodeTensorValue<Fmt>(outBits, outMeta));
    }
  }
  samples.flush();
  registerTensorGPMetadata(tensorMetaPtr(o), outMeta);
}

template <typename Fmt>
static void relu_kernel(UnrankedMemRefType<typename Fmt::MemT> *In,
                        UnrankedMemRefType<typename Fmt::MemT> *Out,
                        int64_t qalignKey) {
  DynamicMemRefType<typename Fmt::MemT> in(*In), out(*Out);
  TensorGPMetadata inMeta = lookupTensorGPMetadata(tensorMetaPtr(in));
  TensorGPMetadata outMeta =
      chooseOutputTensorGPMetadata<Fmt>(qalignKey, inMeta);
  bool useMetaF32 = tensorMathUsesF32ForAlps(inMeta, outMeta);
  int64_t n = num_elems(in);
  QAlignOutputSampleBuffer samples = makeQAlignOutputSampleBuffer(qalignKey, n);
  const bool useRuntimeOutputAlps =
      positRuntimeOutputAlpsMode() != PositRuntimeOutputAlpsMode::Off &&
      fmtSupportsRuntimeOutputAlps<Fmt>();
  std::vector<double> deferredValues;
  if (useRuntimeOutputAlps)
    deferredValues.resize(static_cast<size_t>(std::max<int64_t>(n, 0)));
  for (int64_t i = 0; i < n; ++i) {
    auto bits = to_bits(in.data[offset_of(in, i)]);
    double v = 0.0;
    if (useMetaF32) {
      float orig = static_cast<float>(decodeTensorValue<Fmt>(bits, inMeta));
      float vf = (orig < 0.0f) ? 0.0f : orig;
      v = static_cast<double>(vf);
    } else {
      double orig = decodeTensorValue<Fmt>(bits, inMeta);
      v = (orig < 0.0) ? 0.0 : orig;
    }
    if (useRuntimeOutputAlps) {
      deferredValues[static_cast<size_t>(i)] = v;
    } else {
      auto outBits = encodeTensorValue<Fmt>(v, outMeta);
      out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(outBits);
      samples.maybeAppend(i, -1, v, decodeTensorValue<Fmt>(outBits, outMeta));
    }
  }
  if (useRuntimeOutputAlps) {
    outMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, outMeta);
    for (int64_t i = 0; i < n; ++i) {
      double v = deferredValues[static_cast<size_t>(i)];
      auto outBits = encodeTensorValue<Fmt>(v, outMeta);
      out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(outBits);
      samples.maybeAppend(i, -1, v, decodeTensorValue<Fmt>(outBits, outMeta));
    }
  }
  samples.flush();
  registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
}

// -------- decoded-weight f32 cache (optimization (c)) --------
// Decoding posit->f32 is the dominant cost of the useF32Dot (POSIT_QOP_F32_MATH)
// gemm path, and CONSTANT WEIGHTS get re-decoded every forward. Cache the decoded
// f32 buffer keyed by data pointer. Safety: a content hash of the raw posit words
// (cheap, ~2 instr/elem vs ~15 for decode) is verified on every lookup, so if a
// pointer is reused for a different tensor (e.g. an activation whose content
// changed) the hash mismatches and we re-decode -> never returns stale data.
// Constant weights always match -> cache hit -> skip the expensive decode.
// gemm_kernel calls are serial (the omp parallelism is inside a call), so the
// lookup runs outside the parallel region and a single mutex suffices.
struct DecodedF32Entry {
  uint64_t hash = 0;
  int64_t elems = -1;
  std::vector<float> f32;
};
static std::unordered_map<const void *, DecodedF32Entry> gDecodedF32Cache;
static std::mutex gDecodedF32CacheMutex;

static inline bool positWeightCacheEnabled() {
  static int e = -1;
  if (e >= 0)
    return e == 1;
  const char *v = std::getenv("POSIT_WEIGHT_CACHE");
  e = (v && (std::string(v) == "0" || std::string(v) == "off" ||
             std::string(v) == "false"))
          ? 0
          : 1;  // default ON
  return e == 1;
}

template <typename Fmt>
static const float *getDecodedF32(const typename Fmt::MemT *data, int64_t offset,
                                  int64_t elems) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a over raw posit words
  for (int64_t t = 0; t < elems; ++t) {
    h ^= static_cast<uint64_t>(to_bits(data[offset + t]));
    h *= 1099511628211ULL;
  }
  const void *key = static_cast<const void *>(data + offset);
  std::lock_guard<std::mutex> lock(gDecodedF32CacheMutex);
  DecodedF32Entry &e = gDecodedF32Cache[key];
  if (e.elems == elems && e.hash == h &&
      static_cast<int64_t>(e.f32.size()) == elems)
    return e.f32.data();  // hit
  e.hash = h;
  e.elems = elems;
  e.f32.resize(static_cast<size_t>(elems));
  for (int64_t t = 0; t < elems; ++t)
    e.f32[static_cast<size_t>(t)] =
        static_cast<float>(double_from_bits_fmt<Fmt>(data[offset + t]));
  return e.f32.data();
}

template <typename Fmt>
static void gemm_kernel(UnrankedMemRefType<typename Fmt::MemT> *A,
                        UnrankedMemRefType<typename Fmt::MemT> *B,
                        UnrankedMemRefType<typename Fmt::MemT> *C,
                        UnrankedMemRefType<typename Fmt::MemT> *Y, float alpha, float beta,
                        int64_t transA, int64_t transB, int64_t qalignKey) {
  struct GemmProfScope {
    long long t0;
    ~GemmProfScope() {
      if (positProfileEnabled())
        gProfGemmNs += positNowNs() - t0;
    }
  } _gemmProf{positProfileEnabled() ? positNowNs() : 0};
  DynamicMemRefType<typename Fmt::MemT> a(*A), b(*B), c(*C), y(*Y);
  const void *aMetaPtr = tensorMetaPtr(a);
  const void *bMetaPtr = tensorMetaPtr(b);
  const void *cMetaPtr = tensorMetaPtr(c);
  TensorGPMetadata aMeta = lookupTensorGPMetadata(aMetaPtr);
  TensorGPMetadata bMeta = lookupTensorGPMetadata(bMetaPtr);
  TensorGPMetadata cMeta = lookupTensorGPMetadata(cMetaPtr);
  TensorGPMetadata yMeta =
      chooseOutputTensorGPMetadata<Fmt>(qalignKey, aMeta, bMeta, cMeta);
  const bool aHasChannelMeta = tensorHasSpecialChannelGPMetadata(aMetaPtr);
  const bool bHasChannelMeta = tensorHasSpecialChannelGPMetadata(bMetaPtr);
  const bool cHasChannelMeta = tensorHasSpecialChannelGPMetadata(cMetaPtr);
  const bool hasChannelMeta = aHasChannelMeta || bHasChannelMeta || cHasChannelMeta;
  const bool hasAlpsChannelMeta = tensorHasAlpsChannelGPMetadata(aMetaPtr) ||
                                  tensorHasAlpsChannelGPMetadata(bMetaPtr) ||
                                  tensorHasAlpsChannelGPMetadata(cMetaPtr);
  bool useMetaF32 = tensorMathUsesF32ForAlps(yMeta, aMeta, bMeta, cMeta) ||
                    hasAlpsChannelMeta;
  const bool useMetaPath = hasChannelMeta ||
                           tensorHasSpecialDecodeMetadata(yMeta) ||
                           tensorHasSpecialDecodeMetadata(aMeta) ||
                           tensorHasSpecialDecodeMetadata(bMeta) ||
                           tensorHasSpecialDecodeMetadata(cMeta);
  const bool useRuntimeOutputAlps =
      useMetaPath &&
      positRuntimeOutputAlpsMode() != PositRuntimeOutputAlpsMode::Off &&
      fmtSupportsRuntimeOutputAlps<Fmt>();
  QAlignOutputSampleBuffer samples;
  auto zeroY = [&]() {
    int64_t yElems = 0;
    if (!num_elems_safe(y, yElems))
      return;
    auto z = from_bits<typename Fmt::MemT>(encodeTensorValue<Fmt>(0.0, yMeta));
    for (int64_t i = 0; i < yElems; ++i)
      y.data[offset_of(y, i)] = z;
    registerTensorGPMetadata(tensorMetaPtr(y), yMeta);
  };
  if (!validate_memref_desc(a) || !validate_memref_desc(b) ||
      !validate_memref_desc(c) || !validate_memref_desc(y)) {
    return;
  }
  if (!is_dense_contiguous_memref(a) || !is_dense_contiguous_memref(b) ||
      !is_dense_contiguous_memref(c) || !is_dense_contiguous_memref(y)) {
    return;
  }
  bool mapA = has_mapped_dense_storage(a);
  bool mapB = has_mapped_dense_storage(b);
  bool mapY = has_mapped_dense_storage(y);
  bool mapC = has_mapped_dense_storage(c);
  if (!mapA || !mapB || !mapY || !mapC) {
    if (!mapA)
      dumpMemRefDesc("gemm.A", a);
    if (!mapB)
      dumpMemRefDesc("gemm.B", b);
    if (!mapY)
      dumpMemRefDesc("gemm.Y", y);
    if (!mapC)
      dumpMemRefDesc("gemm.C", c);
    zeroY();
    return;
  }

  // Batched matmul: A[...,M,K] x B[...,K,N] -> Y[...,M,N] with matching leading
  // (batch) dims. Covers GPT-2 attention (rank-4 [B,H,M,K] x [B,H,K,N], alpha=1,
  // beta=0, no bias). The 2D / 3D-im2col paths below do NOT handle rank>=3 batched
  // matmul; without this it fell into the 2D path, mis-read dims and zeroY()'d ->
  // every attention output was 0 -> GPT-2 logits collapsed (uniform ppl, any
  // bit-width). Plain posit dot per (batch, i, j); no ALPS metadata on attention.
  if (a.rank >= 3 && y.rank == a.rank && transA == 0 && transB == 0 &&
      (b.rank == a.rank || b.rank == 2)) {
    int64_t R = a.rank;
    const bool sharedB = (b.rank == 2);  // B[K,N] shared over batch (e.g. lm_head)
    int64_t M = a.sizes[R - 2], K = a.sizes[R - 1];
    int64_t Kb = sharedB ? b.sizes[0] : b.sizes[R - 2];
    int64_t N = sharedB ? b.sizes[1] : b.sizes[R - 1];
    bool ok = (M >= 0 && K >= 0 && N >= 0 && K == Kb &&
               y.sizes[R - 2] == M && y.sizes[R - 1] == N);
    int64_t batch = 1;
    for (int64_t d = 0; d < R - 2 && ok; ++d) {
      if (a.sizes[d] != y.sizes[d])
        ok = false;
      else if (!sharedB && a.sizes[d] != b.sizes[d])
        ok = false;
      else
        batch *= a.sizes[d];
    }
    if (ok) {
      samples = makeQAlignOutputSampleBuffer(qalignKey, batch * M * N);
      const bool useF32Dot = positQopF32MathEnabled() &&
                             positQopF32MathOpEnabled(PositDotOpKind::Gemm);
      {
        bool quireForProbe = (!useMetaPath) && (!useF32Dot) &&
                             dotAccumulatorWouldUseQuire<Fmt>();
        recordDotProbe<Fmt>("matmul_batched", qalignKey,
            useF32Dot ? "f32" : (quireForProbe ? "quire" : "posit_acc"),
            quireForProbe, useF32Dot, false, useMetaPath, batch, M, N, 0, K,
            batch * M * N);
      }
      const double alphaD = static_cast<double>(alpha);
      const double betaD = static_cast<double>(beta);
      auto pAlpha = Fmt::fromDouble(alphaD);
      int64_t cElems = 0;
      double betaC = 0.0;  // beta * C, with C broadcast as a scalar (GPT-2: C=[1])
      if (betaD != 0.0 && num_elems_safe(c, cElems) && cElems > 0)
        betaC = betaD * Fmt::toDouble(Fmt::fromRaw(c.data[c.offset]));
      const int64_t aSlice = M * K, bSlice = sharedB ? 0 : K * N, ySlice = M * N;
      // Speedup (f32-math only): A is invariant over j, but the naive loop
      // re-decodes each A element once per j (e.g. lm_head M=1,N=50257 decodes
      // A[768] 50257x). Pre-decode A once (batch*M*K) into f32 and reuse -> ~2x
      // fewer posit decodes for skinny matmuls, and the inner loop vectorizes.
      // Bit-identical (same f32 fma order). B stays inline (no per-i redundancy
      // for M=1; caching decoded weights across calls is a separate change).
      std::vector<float> aF32;
      const float *aF32p = nullptr;
      if (useF32Dot) {
        // Serial on purpose: A is small (batch*M*K, e.g. 768) and this runs once
        // per gemm call; an omp region here would fork/join per call and dominate.
        aF32.resize(static_cast<size_t>(batch * M * K));
        for (int64_t t = 0; t < batch * M * K; ++t)
          aF32[static_cast<size_t>(t)] = static_cast<float>(
              double_from_bits_fmt<Fmt>(a.data[a.offset + t]));
        aF32p = aF32.data();
      }
      // (c) decoded-weight cache: only for a shared 2D weight B (e.g. lm_head);
      // attention's B is a per-slice activation (not cached). Decodes B once and
      // reuses across forwards via content-hash; nullptr -> inline decode below.
      const float *bF32c = (useF32Dot && sharedB && positWeightCacheEnabled())
                               ? getDecodedF32<Fmt>(b.data, b.offset, K * N)
                               : nullptr;
      // Method B: parallelize independent output elements (batch,i,j). Each output
      // is its own dot product -> bit-identical to serial; only when sample
      // collection is off. ao/bo/yo computed inside so the 3 loops are perfectly
      // nested for collapse(3). lm_head (batch=1,M=1,N=50257) parallelizes over N.
#pragma omp parallel for collapse(3) if(!samples.enabled) num_threads(positOmpThreadCount()) schedule(static)
      for (int64_t bi = 0; bi < batch; ++bi) {
        for (int64_t i = 0; i < M; ++i) {
          for (int64_t j = 0; j < N; ++j) {
            const int64_t ao = a.offset + bi * aSlice;
            const int64_t bo = b.offset + bi * bSlice;
            const int64_t yo = y.offset + bi * ySlice;
            typename Fmt::PositT acc;
            if (useF32Dot) {
              // POSIT_QOP_F32_MATH: decode posit->f32, accumulate the dot in f32,
              // re-encode. Storage stays posit; only the MAC arithmetic is f32
              // (mirrors the 2D/3D gemm f32 path so attention/lm_head are
              // consistent with the linear layers under POSIT_QOP_F32_MATH=on).
              float dotF = 0.0f;
              const int64_t arow = (bi * M + i) * K;
              for (int64_t k = 0; k < K; ++k) {
                float bv = bF32c ? bF32c[k * N + j]
                                 : static_cast<float>(Fmt::toDouble(
                                       Fmt::fromRaw(b.data[bo + k * N + j])));
                dotF = std::fma(aF32p[arow + k], bv, dotF);
              }
              acc = Fmt::fromDouble(static_cast<double>(dotF) * alphaD + betaC);
            } else {
              auto dot = dot_product_accumulate_safe<Fmt>(
                  K,
                  [&](int64_t k) { return Fmt::fromRaw(a.data[ao + i * K + k]); },
                  [&](int64_t k) { return Fmt::fromRaw(b.data[bo + k * N + j]); });
              acc = dot;
              if (alphaD != 1.0) {
                try { acc = Fmt::mul(acc, pAlpha); } catch (...) {}
              }
              if (betaC != 0.0)
                acc = Fmt::add(acc, Fmt::fromDouble(betaC));
            }
            auto outBits = Fmt::toRaw(acc);
            y.data[yo + i * N + j] = outBits;
            int64_t lin = (bi * M + i) * N + j;
            samples.maybeAppend(lin, -1, Fmt::toDouble(acc),
                                double_from_bits_fmt<Fmt>(outBits));
          }
        }
      }
      samples.flush();
      registerTensorGPMetadata(tensorMetaPtr(y), yMeta);
      return;
    }
  }

  // Conv-im2col style GEMM for ONNX MatMul lowering:
  // A[M,K], B[N,K,S], Y[N,M,S] where N is batch-like leading dim.
  if (a.rank == 2 && b.rank == 3 && y.rank == 3 && transA == 0 && transB == 0) {
    int64_t M = a.sizes[0];
    int64_t K = a.sizes[1];
    int64_t N = b.sizes[0];
    int64_t S = b.sizes[2];
    if (M < 0 || K < 0 || N < 0 || S < 0 || b.sizes[1] != K || y.sizes[0] != N ||
        y.sizes[1] != M || y.sizes[2] != S) {
      zeroY();
      return;
    }
    samples = makeQAlignOutputSampleBuffer(qalignKey, N * M * S);

    {
      bool metaPathForProbe = useMetaPath;
      bool f32ForProbe = (!metaPathForProbe) && positQopF32MathEnabled() && positQopF32MathOpEnabled(PositDotOpKind::Gemm);
      bool p16ForProbe = (!metaPathForProbe) && (!f32ForProbe) && shouldUseBuildMixedDot<Fmt>(PositDotOpKind::Gemm);
      bool quireForProbe = (!metaPathForProbe) && (!f32ForProbe) && (!p16ForProbe) && dotAccumulatorWouldUseQuire<Fmt>();
      std::string modeForProbe = metaPathForProbe ? (useMetaF32 ? "gp_metadata_f32" : "gp_metadata_f64") : (f32ForProbe ? "f32" : (p16ForProbe ? positBuildMixedModeName() : (quireForProbe ? "quire" : "posit_acc")));
      recordDotProbe<Fmt>("gemm3d", qalignKey, modeForProbe, quireForProbe, f32ForProbe, p16ForProbe, metaPathForProbe, N, M, S, 0, K, N * M * S);
      if (positProfileEnabled()) {
        static std::atomic<int> _gp{0};
        if (_gp.fetch_add(1) < 3)
          std::fprintf(stderr, "[GEMM3D-PATH] mode=%s M=%lld N=%lld K=%lld S=%lld\n",
                       modeForProbe.c_str(), (long long)M, (long long)N,
                       (long long)K, (long long)S);
      }
    }

    if (useMetaPath) {
      if (useMetaF32) {
        const float alphaF = alpha;
        const float betaF = beta;
        std::vector<double> deferredValues;
        if (useRuntimeOutputAlps)
          deferredValues.resize(static_cast<size_t>(N * M * S));
        // Speedup (the 1x1 pointwise-conv path — the bulk of MobileNetV2's MACs):
        // the per-channel metadata of A depends only on m and B only on n, and the
        // naive loop re-decoded a[m,k] once per output pixel s (S times) and
        // b[n,k,s] once per output channel m (M times), each with a mutex-locked
        // metadata lookup PER multiply-accumulate. Hoist the two lookups out and
        // pre-decode A (M*K) and B (N*K*S) ONCE; the inner loop becomes a plain
        // fma. Removes O(N*M*S*K) locks + redundant posit-decode/ALPS-decompand.
        // Bit-identical: same float decode values, same fma order over k.
        std::vector<float> aF32(static_cast<size_t>(M) * K);
        for (int64_t m = 0; m < M; ++m) {
          TensorGPMetadata aMetaCh =
              lookupTensorGPMetadataForChannel(aMetaPtr, m, aMeta);
          for (int64_t k = 0; k < K; ++k)
            aF32[static_cast<size_t>(m) * K + k] = static_cast<float>(
                decodeTensorValue<Fmt>(load2(a, m, k), aMetaCh));
        }
        std::vector<float> bF32(static_cast<size_t>(N) * K * S);
        for (int64_t n = 0; n < N; ++n) {
          TensorGPMetadata bMetaCh =
              lookupTensorGPMetadataForChannel(bMetaPtr, n, bMeta);
          for (int64_t k = 0; k < K; ++k)
            for (int64_t s = 0; s < S; ++s)
              bF32[(static_cast<size_t>(n) * K + k) * S + s] = static_cast<float>(
                  decodeTensorValue<Fmt>(load3(b, n, k, s), bMetaCh));
        }
#pragma omp parallel for collapse(2) if(!samples.enabled && N * M >= 4) num_threads(positOmpThreadCount()) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
          for (int64_t m = 0; m < M; ++m) {
            const float *aRow = &aF32[static_cast<size_t>(m) * K];
            for (int64_t s = 0; s < S; ++s) {
              int64_t idx3[3] = {n, m, s};
              TensorGPMetadata cMetaCh =
                  (c.rank == 1)
                      ? lookupTensorGPMetadataForChannel(tensorMetaPtr(c), n, cMeta)
                      : cMeta;
              float cF = static_cast<float>(
                  decodeTensorValue<Fmt>(load_broadcast_nd(c, idx3, 3), cMetaCh));
              float dot = 0.0f;
              for (int64_t k = 0; k < K; ++k)
                dot = std::fma(aRow[k],
                               bF32[(static_cast<size_t>(n) * K + k) * S + s], dot);
              float yF = std::fma(dot, alphaF, betaF * cF);
              int64_t lin = (n * M + m) * S + s;
              if (useRuntimeOutputAlps) {
                deferredValues[static_cast<size_t>(lin)] = static_cast<double>(yF);
              } else {
                auto outBits = encodeTensorValue<Fmt>(static_cast<double>(yF), yMeta);
                store3(y, n, m, s, outBits);
                samples.maybeAppend(lin, -1, static_cast<double>(yF),
                                    decodeTensorValue<Fmt>(outBits, yMeta));
              }
            }
          }
        }
        if (useRuntimeOutputAlps) {
          yMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, yMeta);
          for (int64_t n = 0; n < N; ++n) {
            for (int64_t m = 0; m < M; ++m) {
              for (int64_t s = 0; s < S; ++s) {
                int64_t lin = (n * M + m) * S + s;
                double yV = deferredValues[static_cast<size_t>(lin)];
                auto outBits = encodeTensorValue<Fmt>(yV, yMeta);
                store3(y, n, m, s, outBits);
                samples.maybeAppend(lin, -1, yV, decodeTensorValue<Fmt>(outBits, yMeta));
              }
            }
          }
        }
      } else {
        const double alphaD = static_cast<double>(alpha);
        const double betaD = static_cast<double>(beta);
        std::vector<double> deferredValues;
        if (useRuntimeOutputAlps)
          deferredValues.resize(static_cast<size_t>(N * M * S));
        // Speedup (gp_metadata_f64 path — the branch MobileNetV2's 1x1 pointwise
        // convs actually take): hoist the per-channel metadata lookups (A depends
        // only on m, B only on n) out of the per-MAC inner loop, and pre-decode A
        // (M*K) and B (N*K*S) ONCE instead of re-decoding a[m,k] for every s and
        // b[n,k,s] for every m with a mutex-locked lookup per multiply-accumulate.
        // Inner loop becomes a plain fma. Bit-identical (same double decode values,
        // same fma order over k).
        std::vector<double> aDec(static_cast<size_t>(M) * K);
        for (int64_t m = 0; m < M; ++m) {
          TensorGPMetadata aMetaCh =
              lookupTensorGPMetadataForChannel(aMetaPtr, m, aMeta);
          for (int64_t k = 0; k < K; ++k)
            aDec[static_cast<size_t>(m) * K + k] =
                decodeTensorValue<Fmt>(load2(a, m, k), aMetaCh);
        }
        std::vector<double> bDec(static_cast<size_t>(N) * K * S);
        for (int64_t n = 0; n < N; ++n) {
          TensorGPMetadata bMetaCh =
              lookupTensorGPMetadataForChannel(bMetaPtr, n, bMeta);
          for (int64_t k = 0; k < K; ++k)
            for (int64_t s = 0; s < S; ++s)
              bDec[(static_cast<size_t>(n) * K + k) * S + s] =
                  decodeTensorValue<Fmt>(load3(b, n, k, s), bMetaCh);
        }
#pragma omp parallel for collapse(2) if(!samples.enabled && N * M >= 4) num_threads(positOmpThreadCount()) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
          for (int64_t m = 0; m < M; ++m) {
            const double *aRow = &aDec[static_cast<size_t>(m) * K];
            for (int64_t s = 0; s < S; ++s) {
              int64_t idx3[3] = {n, m, s};
              TensorGPMetadata cMetaCh =
                  (c.rank == 1)
                      ? lookupTensorGPMetadataForChannel(tensorMetaPtr(c), n, cMeta)
                      : cMeta;
              double cD =
                  decodeTensorValue<Fmt>(load_broadcast_nd(c, idx3, 3), cMetaCh);
              double dot = 0.0;
              for (int64_t k = 0; k < K; ++k)
                dot = std::fma(aRow[k],
                               bDec[(static_cast<size_t>(n) * K + k) * S + s], dot);
              double yD = std::fma(dot, alphaD, betaD * cD);
              int64_t lin = (n * M + m) * S + s;
              if (useRuntimeOutputAlps) {
                deferredValues[static_cast<size_t>(lin)] = yD;
              } else {
                auto outBits = encodeTensorValue<Fmt>(yD, yMeta);
                store3(y, n, m, s, outBits);
                samples.maybeAppend(lin, -1, yD, decodeTensorValue<Fmt>(outBits, yMeta));
              }
            }
          }
        }
        if (useRuntimeOutputAlps) {
          yMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, yMeta);
          for (int64_t n = 0; n < N; ++n) {
            for (int64_t m = 0; m < M; ++m) {
              for (int64_t s = 0; s < S; ++s) {
                int64_t lin = (n * M + m) * S + s;
                double yV = deferredValues[static_cast<size_t>(lin)];
                auto outBits = encodeTensorValue<Fmt>(yV, yMeta);
                store3(y, n, m, s, outBits);
                samples.maybeAppend(lin, -1, yV, decodeTensorValue<Fmt>(outBits, yMeta));
              }
            }
          }
        }
      }
      samples.flush();
      registerTensorGPMetadata(tensorMetaPtr(y), yMeta);
      return;
    }

    const bool opAlign = positOpAlignEnabled();
    const bool useF32Dot =
        positQopF32MathEnabled() &&
        positQopF32MathOpEnabled(PositDotOpKind::Gemm);
    const double alphaD = static_cast<double>(alpha);
    const double betaD = static_cast<double>(beta);
    auto pAlpha = Fmt::fromDouble(alphaD);
    auto pBeta = Fmt::fromDouble(betaD);

    // Method B (intra-op parallelism): each output element (n,m,s) is an
    // independent dot product, so we parallelize the output loops. Results are
    // BIT-IDENTICAL to serial (we do not change the per-element accumulation
    // order). Only enabled when sample collection is off (samples.maybeAppend
    // is not thread-safe). Threads come from OMP_NUM_THREADS; if the .so was not
    // compiled with -fopenmp the pragma is ignored and this stays serial.
#pragma omp parallel for collapse(3) if(!samples.enabled) num_threads(positOmpThreadCount()) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t s = 0; s < S; ++s) {
          int64_t idx3[3] = {n, m, s};
          auto cbit = load_broadcast_nd(c, idx3, 3);

          if (useF32Dot) {
            try {
              float dotF = 0.0f;
              for (int64_t k = 0; k < K; ++k) {
                float av =
                    static_cast<float>(double_from_bits_fmt<Fmt>(load2(a, m, k)));
                float bv =
                    static_cast<float>(double_from_bits_fmt<Fmt>(load3(b, n, k, s)));
                dotF = std::fma(av, bv, dotF);
              }
              float cF = static_cast<float>(Fmt::toDouble(Fmt::fromRaw(cbit)));
              float accF = std::fma(dotF, alpha, beta * cF);
              auto outBits = Fmt::toRaw(Fmt::fromDouble(static_cast<double>(accF)));
              store3(y, n, m, s, outBits);
              int64_t lin = (n * M + m) * S + s;
              samples.maybeAppend(lin, -1, static_cast<double>(accF),
                                  double_from_bits_fmt<Fmt>(outBits));
              continue;
            } catch (...) {
              recordPositFallback(gPositOpAlignFallbackCount, "f32_dot_fallback",
                                  "gemm 3D f32 dot path threw, fallback to posit path");
            }
          }

          auto dot = dot_product_accumulate_safe<Fmt>(
              K,
              [&](int64_t k) { return Fmt::fromRaw(load2(a, m, k)); },
              [&](int64_t k) { return Fmt::fromRaw(load3(b, n, k, s)); });
          if (opAlign) {
            try {
              double dotD = Fmt::toDouble(dot);
              double cD = Fmt::toDouble(Fmt::fromRaw(cbit));
              double accD = std::fma(dotD, alphaD, betaD * cD);
              auto outBits = Fmt::toRaw(Fmt::fromDouble(accD));
              store3(y, n, m, s, outBits);
              int64_t lin = (n * M + m) * S + s;
              samples.maybeAppend(lin, -1, accD,
                                  double_from_bits_fmt<Fmt>(outBits));
              continue;
            } catch (...) {
              recordPositFallback(gPositOpAlignFallbackCount, "op_align",
                                  "gemm 3D epilogue align threw, fallback to posit epilogue");
            }
          }

          auto acc = dot;
          try {
            acc = Fmt::mul(acc, pAlpha);
          } catch (...) {
            recordPositFallback(gPositQuireFallbackCount, "quire_fallback",
                                "gemm alpha multiply threw");
            acc = Fmt::fromDouble(0.0);
          }
          auto addc = Fmt::fromDouble(0.0);
          try {
            addc = Fmt::mul(Fmt::fromRaw(cbit), pBeta);
            acc = Fmt::add(acc, addc);
          } catch (...) {
            recordPositFallback(gPositQuireFallbackCount, "quire_fallback",
                                "gemm beta add threw");
          }
          auto outBits = Fmt::toRaw(acc);
          store3(y, n, m, s, outBits);
          double orig = Fmt::toDouble(acc);
          int64_t lin = (n * M + m) * S + s;
          samples.maybeAppend(lin, -1, orig, double_from_bits_fmt<Fmt>(outBits));
        }
      }
    }
    samples.flush();
    registerTensorGPMetadata(tensorMetaPtr(y), yMeta);
    return;
  }

  // Standard 2D GEMM.
  if (a.rank < 2 || b.rank < 2 || y.rank < 2)
    return;
  int64_t a0 = a.sizes[0], a1 = a.sizes[1];
  int64_t b0 = b.sizes[0], b1 = b.sizes[1];
  int64_t M = transA ? a1 : a0;
  int64_t K = transA ? a0 : a1;
  int64_t N = transB ? b0 : b1;
  int64_t kFromB = transB ? b1 : b0;
  if (K != kFromB || y.sizes[0] != M || y.sizes[1] != N) {
    zeroY();
    return;
  }
  samples = makeQAlignOutputSampleBuffer(qalignKey, M * N);

  {
    bool metaPathForProbe = useMetaPath;
    bool f32ForProbe = (!metaPathForProbe) && positQopF32MathEnabled() && positQopF32MathOpEnabled(PositDotOpKind::Gemm);
    bool p16ForProbe = (!metaPathForProbe) && (!f32ForProbe) && shouldUseBuildMixedDot<Fmt>(PositDotOpKind::Gemm);
    bool quireForProbe = (!metaPathForProbe) && (!f32ForProbe) && (!p16ForProbe) && dotAccumulatorWouldUseQuire<Fmt>();
    std::string modeForProbe = metaPathForProbe ? (useMetaF32 ? "gp_metadata_f32" : "gp_metadata_f64") : (f32ForProbe ? "f32" : (p16ForProbe ? positBuildMixedModeName() : (quireForProbe ? "quire" : "posit_acc")));
    recordDotProbe<Fmt>("gemm2d", qalignKey, modeForProbe, quireForProbe, f32ForProbe, p16ForProbe, metaPathForProbe, M, N, 0, 0, K, M * N);
  }

  if (useMetaPath) {
    if (useMetaF32) {
      const float alphaF = alpha;
      const float betaF = beta;
      std::vector<double> deferredValues;
      if (useRuntimeOutputAlps)
        deferredValues.resize(static_cast<size_t>(M * N));
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          int64_t idx2[2] = {i, j};
          TensorGPMetadata cMetaCh =
              (c.rank == 1)
                  ? lookupTensorGPMetadataForChannel(tensorMetaPtr(c), j, cMeta)
                  : cMeta;
          float cF = static_cast<float>(
              decodeTensorValue<Fmt>(load_broadcast_nd(c, idx2, 2), cMetaCh));
          float dot = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            TensorGPMetadata aMetaCh = lookupTensorGPMetadataForChannel(
                aMetaPtr, transA ? k : i, aMeta);
            TensorGPMetadata bMetaCh = lookupTensorGPMetadataForChannel(
                bMetaPtr, transB ? j : k, bMeta);
            float av = static_cast<float>(decodeTensorValue<Fmt>(
                transA ? load2(a, k, i) : load2(a, i, k), aMetaCh));
            float bv = static_cast<float>(decodeTensorValue<Fmt>(
                transB ? load2(b, j, k) : load2(b, k, j), bMetaCh));
            dot = std::fma(av, bv, dot);
          }
          float yF = std::fma(dot, alphaF, betaF * cF);
          int64_t lin = i * N + j;
          if (useRuntimeOutputAlps) {
            deferredValues[static_cast<size_t>(lin)] = static_cast<double>(yF);
          } else {
            auto outBits = encodeTensorValue<Fmt>(static_cast<double>(yF), yMeta);
            store2(y, i, j, outBits);
            samples.maybeAppend(lin, -1, static_cast<double>(yF),
                                decodeTensorValue<Fmt>(outBits, yMeta));
          }
        }
      }
      if (useRuntimeOutputAlps) {
        yMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, yMeta);
        for (int64_t i = 0; i < M; ++i) {
          for (int64_t j = 0; j < N; ++j) {
            int64_t lin = i * N + j;
            double yV = deferredValues[static_cast<size_t>(lin)];
            auto outBits = encodeTensorValue<Fmt>(yV, yMeta);
            store2(y, i, j, outBits);
            samples.maybeAppend(lin, -1, yV, decodeTensorValue<Fmt>(outBits, yMeta));
          }
        }
      }
    } else {
      const double alphaD = static_cast<double>(alpha);
      const double betaD = static_cast<double>(beta);
      std::vector<double> deferredValues;
      if (useRuntimeOutputAlps)
        deferredValues.resize(static_cast<size_t>(M * N));
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          int64_t idx2[2] = {i, j};
          TensorGPMetadata cMetaCh =
              (c.rank == 1)
                  ? lookupTensorGPMetadataForChannel(tensorMetaPtr(c), j, cMeta)
                  : cMeta;
          double cD =
              decodeTensorValue<Fmt>(load_broadcast_nd(c, idx2, 2), cMetaCh);
          double dot = 0.0;
          for (int64_t k = 0; k < K; ++k) {
            TensorGPMetadata aMetaCh = lookupTensorGPMetadataForChannel(
                aMetaPtr, transA ? k : i, aMeta);
            TensorGPMetadata bMetaCh = lookupTensorGPMetadataForChannel(
                bMetaPtr, transB ? j : k, bMeta);
            double av = decodeTensorValue<Fmt>(
                transA ? load2(a, k, i) : load2(a, i, k), aMetaCh);
            double bv = decodeTensorValue<Fmt>(
                transB ? load2(b, j, k) : load2(b, k, j), bMetaCh);
            dot = std::fma(av, bv, dot);
          }
          double yD = std::fma(dot, alphaD, betaD * cD);
          int64_t lin = i * N + j;
          if (useRuntimeOutputAlps) {
            deferredValues[static_cast<size_t>(lin)] = yD;
          } else {
            auto outBits = encodeTensorValue<Fmt>(yD, yMeta);
            store2(y, i, j, outBits);
            samples.maybeAppend(lin, -1, yD, decodeTensorValue<Fmt>(outBits, yMeta));
          }
        }
      }
      if (useRuntimeOutputAlps) {
        yMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, yMeta);
        for (int64_t i = 0; i < M; ++i) {
          for (int64_t j = 0; j < N; ++j) {
            int64_t lin = i * N + j;
            double yV = deferredValues[static_cast<size_t>(lin)];
            auto outBits = encodeTensorValue<Fmt>(yV, yMeta);
            store2(y, i, j, outBits);
            samples.maybeAppend(lin, -1, yV, decodeTensorValue<Fmt>(outBits, yMeta));
          }
        }
      }
    }
    samples.flush();
    registerTensorGPMetadata(tensorMetaPtr(y), yMeta);
    return;
  }

  const bool opAlign = positOpAlignEnabled();
  const bool useF32Dot =
      positQopF32MathEnabled() &&
      positQopF32MathOpEnabled(PositDotOpKind::Gemm);
  const double alphaD = static_cast<double>(alpha);
  const double betaD = static_cast<double>(beta);
  auto pAlpha = Fmt::fromDouble(alphaD);
  auto pBeta = Fmt::fromDouble(betaD);

  // Speedup (f32-math only): pre-decode A once (M*K) instead of re-decoding each
  // A element per j. Uses load2 so strides/transA are respected. Serial (small,
  // once per call). Bit-identical.
  std::vector<float> aF32g;
  const float *aF32gp = nullptr;
  if (useF32Dot) {
    aF32g.resize(static_cast<size_t>(M) * static_cast<size_t>(K));
    for (int64_t i = 0; i < M; ++i)
      for (int64_t k = 0; k < K; ++k)
        aF32g[static_cast<size_t>(i) * K + k] = static_cast<float>(Fmt::toDouble(
            Fmt::fromRaw(transA ? load2(a, k, i) : load2(a, i, k))));
    aF32gp = aF32g.data();
  }
  // (c) decoded-weight cache: B is the linear-layer weight (constant). Decode
  // once, reuse across forwards (content-hash verified). nullptr -> inline decode.
  const float *bF32cg = (useF32Dot && positWeightCacheEnabled())
                            ? getDecodedF32<Fmt>(b.data, b.offset, K * N)
                            : nullptr;
  // Method B: parallelize independent output elements (i,j); bit-identical to
  // serial; only when sample collection is off; needs -fopenmp + OMP_NUM_THREADS.
#pragma omp parallel for collapse(2) if(!samples.enabled) num_threads(positOmpThreadCount()) schedule(static)
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      int64_t idx2[2] = {i, j};
      auto cbit = load_broadcast_nd(c, idx2, 2);
      if (useF32Dot) {
        try {
          float dotF = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            float bv = bF32cg
                           ? (transB ? bF32cg[static_cast<size_t>(j) * K + k]
                                     : bF32cg[static_cast<size_t>(k) * N + j])
                           : static_cast<float>(double_from_bits_fmt<Fmt>(
                                 transB ? load2(b, j, k) : load2(b, k, j)));
            dotF = std::fma(aF32gp[static_cast<size_t>(i) * K + k], bv, dotF);
          }
          float cF = static_cast<float>(Fmt::toDouble(Fmt::fromRaw(cbit)));
          float accF = std::fma(dotF, alpha, beta * cF);
          auto outBits = Fmt::toRaw(Fmt::fromDouble(static_cast<double>(accF)));
          store2(y, i, j, outBits);
          int64_t lin = i * N + j;
          samples.maybeAppend(lin, -1, static_cast<double>(accF),
                              double_from_bits_fmt<Fmt>(outBits));
          continue;
        } catch (...) {
          recordPositFallback(gPositOpAlignFallbackCount, "f32_dot_fallback",
                              "gemm 2D f32 dot path threw, fallback to posit path");
        }
      }

      auto dot = dot_product_accumulate_safe<Fmt>(
          K,
          [&](int64_t k) {
            return Fmt::fromRaw(transA ? load2(a, k, i) : load2(a, i, k));
          },
          [&](int64_t k) {
            return Fmt::fromRaw(transB ? load2(b, j, k) : load2(b, k, j));
          });
      if (opAlign) {
        try {
          double dotD = Fmt::toDouble(dot);
          double cD = Fmt::toDouble(Fmt::fromRaw(cbit));
          double accD = std::fma(dotD, alphaD, betaD * cD);
          auto outBits = Fmt::toRaw(Fmt::fromDouble(accD));
          store2(y, i, j, outBits);
          int64_t lin = i * N + j;
          samples.maybeAppend(lin, -1, accD, double_from_bits_fmt<Fmt>(outBits));
          continue;
        } catch (...) {
          recordPositFallback(gPositOpAlignFallbackCount, "op_align",
                              "gemm 2D epilogue align threw, fallback to posit epilogue");
        }
      }

      auto acc = dot;
      try {
        acc = Fmt::mul(acc, pAlpha);
      } catch (...) {
        recordPositFallback(gPositQuireFallbackCount, "quire_fallback",
                            "gemm alpha multiply threw");
        acc = Fmt::fromDouble(0.0);
      }
      auto addc = Fmt::fromDouble(0.0);
      try {
        addc = Fmt::mul(Fmt::fromRaw(cbit), pBeta);
        acc = Fmt::add(acc, addc);
      } catch (...) {
        recordPositFallback(gPositQuireFallbackCount, "quire_fallback",
                            "gemm beta add threw");
      }
      auto outBits = Fmt::toRaw(acc);
      store2(y, i, j, outBits);
      int64_t lin = i * N + j;
      samples.maybeAppend(lin, -1, Fmt::toDouble(acc),
                          double_from_bits_fmt<Fmt>(outBits));
    }
  }
  samples.flush();
  registerTensorGPMetadata(tensorMetaPtr(y), yMeta);
}

template <typename Fmt>
static void conv2d_nchw_kernel(UnrankedMemRefType<typename Fmt::MemT> *X,
                               UnrankedMemRefType<typename Fmt::MemT> *W,
                               UnrankedMemRefType<typename Fmt::MemT> *B,
                               UnrankedMemRefType<typename Fmt::MemT> *Out,
                               int64_t stride_h, int64_t stride_w, int64_t dilation_h,
                               int64_t dilation_w, int64_t pad_top, int64_t pad_left,
                               int64_t pad_bottom, int64_t pad_right,
                               int64_t groups, int64_t qalignKey) {
  (void)pad_bottom;
  (void)pad_right;

  DynamicMemRefType<typename Fmt::MemT> x(*X), w(*W), b(*B), out(*Out);
  const void *xMetaPtr = tensorMetaPtr(x);
  const void *wMetaPtr = tensorMetaPtr(w);
  const void *bMetaPtr = tensorMetaPtr(b);
  TensorGPMetadata xMeta = lookupTensorGPMetadata(xMetaPtr);
  TensorGPMetadata wMeta = lookupTensorGPMetadata(wMetaPtr);
  TensorGPMetadata bMeta = lookupTensorGPMetadata(bMetaPtr);
  TensorGPMetadata outMeta =
      chooseOutputTensorGPMetadata<Fmt>(qalignKey, xMeta, wMeta, bMeta);

  int64_t N = x.sizes[0];
  int64_t H = x.sizes[2];
  int64_t Wd = x.sizes[3];
  int64_t M = w.sizes[0];
  int64_t Cpg = w.sizes[1];
  int64_t kH = w.sizes[2];
  int64_t kW = w.sizes[3];
  int64_t outH = out.sizes[2];
  int64_t outW = out.sizes[3];

  int64_t g = (groups <= 0) ? 1 : groups;
  int64_t Mpg = M / g;
  using DotAcc = DotAccumulator<Fmt>;
  const bool opAlign = positOpAlignEnabled();
  const bool useF32Dot =
      positQopF32MathEnabled() &&
      positQopF32MathOpEnabled(PositDotOpKind::Conv2d);
  const bool useP16MixedDot = shouldUseBuildMixedDot<Fmt>(PositDotOpKind::Conv2d);
  const bool useMetaPath = tensorHasSpecialChannelGPMetadata(xMetaPtr) ||
                           tensorHasSpecialChannelGPMetadata(wMetaPtr) ||
                           tensorHasSpecialChannelGPMetadata(bMetaPtr) ||
                           tensorHasSpecialDecodeMetadata(outMeta) ||
                           tensorHasSpecialDecodeMetadata(xMeta) ||
                           tensorHasSpecialDecodeMetadata(wMeta) ||
                           tensorHasSpecialDecodeMetadata(bMeta);
  const bool useMetaF32 = tensorHasAlpsChannelGPMetadata(xMetaPtr) ||
                          tensorHasAlpsChannelGPMetadata(wMetaPtr) ||
                          tensorHasAlpsChannelGPMetadata(bMetaPtr) ||
                          tensorMathUsesF32ForAlps(outMeta, xMeta, wMeta, bMeta);
  const bool useRuntimeOutputAlps =
      useMetaPath &&
      positRuntimeOutputAlpsMode() != PositRuntimeOutputAlpsMode::Off &&
      fmtSupportsRuntimeOutputAlps<Fmt>();
  {
    bool quireForProbe = (!useMetaPath) && (!useF32Dot) && (!useP16MixedDot) && dotAccumulatorWouldUseQuire<Fmt>();
    std::string modeForProbe = useMetaPath ? (useMetaF32 ? "gp_metadata_f32" : "gp_metadata_f64") : (useF32Dot ? "f32" : (useP16MixedDot ? positBuildMixedModeName() : (quireForProbe ? "quire" : "posit_acc")));
    recordDotProbe<Fmt>("conv2d", qalignKey, modeForProbe, quireForProbe, useF32Dot, useP16MixedDot, useMetaPath, N, M, outH, outW, Cpg * kH * kW, N * M * outH * outW);
  }
  QAlignOutputSampleBuffer samples =
      makeQAlignOutputSampleBuffer(qalignKey, N * M * outH * outW);
  int64_t sampleLin = 0;

  // Speedup (metadata/ALPS path): the input decode uses the per-tensor xMeta and
  // the SAME input pixel x[n,ic,ih,iw] is otherwise re-decoded once per output
  // channel (Mpg times; large for pointwise 1x1 convs). Pre-decode the whole
  // input tensor ONCE here so each pixel is decoded a single time. Bit-identical
  // to the per-pixel decode (same xMeta). Only for the metadata path.
  const bool prof = positProfileEnabled();
  const int64_t Cin = x.sizes[1];
  std::vector<double> xDec;
  const double *xDecp = nullptr;
  if (useMetaPath) {
    long long _td = prof ? positNowNs() : 0;
    xDec.resize(static_cast<size_t>(N) * Cin * H * Wd);
    for (int64_t n = 0; n < N; ++n)
      for (int64_t ic = 0; ic < Cin; ++ic)
        for (int64_t ih = 0; ih < H; ++ih)
          for (int64_t iw = 0; iw < Wd; ++iw)
            xDec[(((static_cast<size_t>(n) * Cin + ic) * H + ih) * Wd) + iw] =
                decodeTensorValue<Fmt>(load4(x, n, ic, ih, iw), xMeta);
    xDecp = xDec.data();
    if (prof)
      gProfConvDecodeNs += positNowNs() - _td;
  }

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t gg = 0; gg < g; ++gg) {
      for (int64_t mm = 0; mm < Mpg; ++mm) {
        int64_t oc = gg * Mpg + mm;
        auto bias = Fmt::fromRaw(load1(b, oc));
        // Speedup (metadata/ALPS path): this output channel's weight metadata
        // (wMetaCh) depends only on oc, so pre-decode the whole Cpg*kH*kW kernel
        // ONCE per oc instead of re-decoding every weight for every output pixel
        // (the old inner loop re-decoded each weight outH*outW times; e.g. an
        // early 112x112 conv decoded each weight ~12544x). Values are identical
        // to the per-pixel decode. Only for the metadata path.
        std::vector<double> wDoc;
        const double *wDocp = nullptr;
        if (useMetaPath) {
          long long _tw = prof ? positNowNs() : 0;
          TensorGPMetadata wMetaChOc =
              lookupTensorGPMetadataForChannel(wMetaPtr, oc, wMeta);
          wDoc.resize(static_cast<size_t>(Cpg) * kH * kW);
          for (int64_t cc = 0; cc < Cpg; ++cc)
            for (int64_t kh = 0; kh < kH; ++kh)
              for (int64_t kw = 0; kw < kW; ++kw)
                wDoc[(static_cast<size_t>(cc) * kH + kh) * kW + kw] =
                    decodeTensorValue<Fmt>(load4(w, oc, cc, kh, kw), wMetaChOc);
          wDocp = wDoc.data();
          if (prof)
            gProfConvDecodeNs += positNowNs() - _tw;
        }
        long long _tmac = prof ? positNowNs() : 0;
        for (int64_t oh = 0; oh < outH; ++oh) {
          for (int64_t ow = 0; ow < outW; ++ow) {
            if (useMetaPath) {
              if (useMetaF32) {
                TensorGPMetadata bMetaCh =
                    lookupTensorGPMetadataForChannel(tensorMetaPtr(b), oc, bMeta);
                double accV =
                    static_cast<double>(static_cast<float>(
                        decodeTensorValue<Fmt>(load1(b, oc), bMetaCh)));
                for (int64_t cc = 0; cc < Cpg; ++cc) {
                  int64_t ic = gg * Cpg + cc;
                  for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t ih = oh * stride_h - pad_top + kh * dilation_h;
                    if (ih < 0 || ih >= H)
                      continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                      int64_t iw = ow * stride_w - pad_left + kw * dilation_w;
                      if (iw < 0 || iw >= Wd)
                        continue;
                      float xv = static_cast<float>(
                          xDecp[(((static_cast<size_t>(n) * Cin + ic) * H + ih) *
                                 Wd) +
                                iw]);
                      float wv = static_cast<float>(
                          wDocp[(static_cast<size_t>(cc) * kH + kh) * kW + kw]);
                      accV = static_cast<double>(std::fma(xv, wv, static_cast<float>(accV)));
                    }
                  }
                }
                if (useRuntimeOutputAlps) {
                  ++sampleLin;
                } else {
                  auto outBits = encodeTensorValue<Fmt>(accV, outMeta);
                  store4(out, n, oc, oh, ow, outBits);
                  samples.maybeAppend(sampleLin++, -1, accV,
                                      decodeTensorValue<Fmt>(outBits, outMeta));
                }
              } else {
                TensorGPMetadata bMetaCh =
                    lookupTensorGPMetadataForChannel(tensorMetaPtr(b), oc, bMeta);
                double accD = decodeTensorValue<Fmt>(load1(b, oc), bMetaCh);
                for (int64_t cc = 0; cc < Cpg; ++cc) {
                  int64_t ic = gg * Cpg + cc;
                  for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t ih = oh * stride_h - pad_top + kh * dilation_h;
                    if (ih < 0 || ih >= H)
                      continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                      int64_t iw = ow * stride_w - pad_left + kw * dilation_w;
                      if (iw < 0 || iw >= Wd)
                        continue;
                      double xv =
                          xDecp[(((static_cast<size_t>(n) * Cin + ic) * H + ih) *
                                 Wd) +
                                iw];
                      double wv =
                          wDocp[(static_cast<size_t>(cc) * kH + kh) * kW + kw];
                      accD = std::fma(xv, wv, accD);
                    }
                  }
                }
                if (useRuntimeOutputAlps) {
                  ++sampleLin;
                } else {
                  auto outBits = encodeTensorValue<Fmt>(accD, outMeta);
                  store4(out, n, oc, oh, ow, outBits);
                  samples.maybeAppend(sampleLin++, -1, accD,
                                      decodeTensorValue<Fmt>(outBits, outMeta));
                }
              }
              continue;
            }
            auto st = DotAcc::init();
#if defined(POSIT_BUILD_MIXED_P16E2)
            using ActiveMixedFmt = FmtP16E2;
#elif defined(POSIT_BUILD_MIXED_P32E2)
            using ActiveMixedFmt = FmtP32E2;
#else
            using ActiveMixedFmt = FmtP16E1;
#endif
            auto mixedAcc = ActiveMixedFmt::fromDouble(0.0);
            float dotF = 0.0f;
            bool dotFailed = false;
            for (int64_t cc = 0; cc < Cpg; ++cc) {
              int64_t ic = gg * Cpg + cc;
              for (int64_t kh = 0; kh < kH; ++kh) {
                int64_t ih = oh * stride_h - pad_top + kh * dilation_h;
                if (ih < 0 || ih >= H)
                  continue;
                for (int64_t kw = 0; kw < kW; ++kw) {
                  int64_t iw = ow * stride_w - pad_left + kw * dilation_w;
                  if (iw < 0 || iw >= Wd)
                    continue;
                  auto px = Fmt::fromRaw(load4(x, n, ic, ih, iw));
                  auto pw = Fmt::fromRaw(load4(w, oc, cc, kh, kw));
                  try {
                    if (useF32Dot) {
                      float xF = static_cast<float>(Fmt::toDouble(px));
                      float wF = static_cast<float>(Fmt::toDouble(pw));
                      dotF = std::fma(xF, wF, dotF);
                    } else if (useP16MixedDot) {
                      auto x16 = ActiveMixedFmt::fromDouble(Fmt::toDouble(px));
                      auto w16 = ActiveMixedFmt::fromDouble(Fmt::toDouble(pw));
                      mixedAcc = ActiveMixedFmt::add(
                          mixedAcc, ActiveMixedFmt::mul(x16, w16));
                    } else {
                      DotAcc::fdp(st, px, pw);
                    }
                  } catch (...) {
                    if (!dotFailed) {
                      recordPositFallback(
                          useF32Dot
                              ? gPositOpAlignFallbackCount
                              : (useP16MixedDot ? gPositMixedP16FallbackCount
                                                : gPositQuireFallbackCount),
                          useF32Dot ? "f32_dot_fallback"
                                    : (useP16MixedDot ? "mixed_p16_fallback"
                                                      : "quire_fallback"),
                          useF32Dot ? "conv f32 dot threw"
                                    : (useP16MixedDot
                                           ? "conv build/runtime mixed dot threw"
                                           : "conv dot fdp threw"));
                    }
                    dotFailed = true;
                  }
                }
              }
            }
            auto acc = bias;
            auto dotPos = Fmt::fromDouble(0.0);
            if (!dotFailed) {
              if (useF32Dot) {
                try {
                  float biasF = static_cast<float>(Fmt::toDouble(bias));
                  float accF = std::fma(1.0f, dotF, biasF);
                  acc = Fmt::fromDouble(static_cast<double>(accF));
                } catch (...) {
                  recordPositFallback(gPositOpAlignFallbackCount, "f32_dot_fallback",
                                      "conv f32 epilogue threw");
                  dotFailed = true;
                }
              } else {
                try {
                  dotPos = useP16MixedDot
                               ? Fmt::fromDouble(ActiveMixedFmt::toDouble(mixedAcc))
                               : DotAcc::finish(st);
                } catch (...) {
                  recordPositFallback(
                      useP16MixedDot ? gPositMixedP16FallbackCount
                                     : gPositQuireFallbackCount,
                      useP16MixedDot ? "mixed_p16_fallback" : "quire_fallback",
                      useP16MixedDot ? "conv build/runtime mixed finish threw"
                                     : "conv dot finish threw");
                  dotFailed = true;
                }
              }
            }
            if (!dotFailed && !useF32Dot) {
              bool aligned = false;
              if (opAlign) {
                try {
                  double biasD = Fmt::toDouble(bias);
                  double dotD = Fmt::toDouble(dotPos);
                  double accD = std::fma(1.0, dotD, biasD);
                  acc = Fmt::fromDouble(accD);
                  aligned = true;
                } catch (...) {
                  recordPositFallback(gPositOpAlignFallbackCount, "op_align",
                                      "conv epilogue align threw, fallback to posit epilogue");
                }
              }
              if (!aligned) {
                try {
                  acc = Fmt::add(bias, dotPos);
                } catch (...) {
                  recordPositFallback(
                      useP16MixedDot ? gPositMixedP16FallbackCount
                                     : gPositQuireFallbackCount,
                      useP16MixedDot ? "mixed_p16_fallback" : "quire_fallback",
                      useP16MixedDot ? "conv build/runtime mixed add-bias threw"
                                     : "conv dot finish threw");
                  acc = bias;
                }
              }
            }
            auto outBits = Fmt::toRaw(acc);
            store4(out, n, oc, oh, ow, outBits);
            samples.maybeAppend(sampleLin++, -1, Fmt::toDouble(acc),
                                double_from_bits_fmt<Fmt>(outBits));
          }
        }
        if (prof)
          gProfConvMacNs += positNowNs() - _tmac;
      }
    }
  }
  if (useRuntimeOutputAlps) {
    std::vector<double> deferredValues;
    deferredValues.reserve(static_cast<size_t>(N * M * outH * outW));
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t gg = 0; gg < g; ++gg) {
        for (int64_t mm = 0; mm < Mpg; ++mm) {
          int64_t oc = gg * Mpg + mm;
          for (int64_t oh = 0; oh < outH; ++oh) {
            for (int64_t ow = 0; ow < outW; ++ow) {
              double accD = decodeTensorValue<Fmt>(
                  load1(b, oc),
                  lookupTensorGPMetadataForChannel(tensorMetaPtr(b), oc, bMeta));
              for (int64_t cc = 0; cc < Cpg; ++cc) {
                int64_t ic = gg * Cpg + cc;
                for (int64_t kh = 0; kh < kH; ++kh) {
                  int64_t ih = oh * stride_h - pad_top + kh * dilation_h;
                  if (ih < 0 || ih >= H)
                    continue;
                  for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t iw = ow * stride_w - pad_left + kw * dilation_w;
                    if (iw < 0 || iw >= Wd)
                      continue;
                    double xv =
                        decodeTensorValue<Fmt>(load4(x, n, ic, ih, iw), xMeta);
                    TensorGPMetadata wMetaCh =
                        lookupTensorGPMetadataForChannel(wMetaPtr, oc, wMeta);
                    double wv =
                        decodeTensorValue<Fmt>(load4(w, oc, cc, kh, kw), wMetaCh);
                    if (useMetaF32) {
                      float accF = static_cast<float>(accD);
                      accF = std::fma(static_cast<float>(xv), static_cast<float>(wv), accF);
                      accD = static_cast<double>(accF);
                    } else {
                      accD = std::fma(xv, wv, accD);
                    }
                  }
                }
              }
              deferredValues.push_back(accD);
            }
          }
        }
      }
    }
    outMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, outMeta);
    sampleLin = 0;
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t gg = 0; gg < g; ++gg) {
        for (int64_t mm = 0; mm < Mpg; ++mm) {
          int64_t oc = gg * Mpg + mm;
          for (int64_t oh = 0; oh < outH; ++oh) {
            for (int64_t ow = 0; ow < outW; ++ow) {
              double accV = deferredValues[static_cast<size_t>(sampleLin)];
              auto outBits = encodeTensorValue<Fmt>(accV, outMeta);
              store4(out, n, oc, oh, ow, outBits);
              samples.maybeAppend(sampleLin++, -1, accV,
                                  decodeTensorValue<Fmt>(outBits, outMeta));
            }
          }
        }
      }
    }
  }
  samples.flush();
  registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
}

template <typename Fmt>
static void maxpool2d_nchw_kernel(UnrankedMemRefType<typename Fmt::MemT> *X,
                                  UnrankedMemRefType<typename Fmt::MemT> *Out,
                                  int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                  int64_t stride_w, int64_t pad_top, int64_t pad_left,
                                  int64_t pad_bottom, int64_t pad_right,
                                  int64_t ceil_mode, int64_t qalignKey) {
  (void)pad_bottom;
  (void)pad_right;
  (void)ceil_mode;

  DynamicMemRefType<typename Fmt::MemT> x(*X), out(*Out);
  TensorGPMetadata xMeta = lookupTensorGPMetadata(tensorMetaPtr(x));
  TensorGPMetadata outMeta =
      chooseOutputTensorGPMetadata<Fmt>(qalignKey, xMeta);
  bool useMetaF32 = tensorMathUsesF32ForAlps(xMeta, outMeta);
  const bool useRuntimeOutputAlps =
      useMetaF32 && positRuntimeOutputAlpsMode() != PositRuntimeOutputAlpsMode::Off &&
      fmtSupportsRuntimeOutputAlps<Fmt>();
  int64_t N = x.sizes[0];
  int64_t C = x.sizes[1];
  int64_t H = x.sizes[2];
  int64_t Wd = x.sizes[3];
  int64_t outH = out.sizes[2];
  int64_t outW = out.sizes[3];
  auto zeroBits = encodeTensorValue<Fmt>(0.0, outMeta);
  QAlignOutputSampleBuffer samples =
      makeQAlignOutputSampleBuffer(qalignKey, N * C * outH * outW);
  int64_t sampleLin = 0;
  std::vector<double> deferredValues;
  if (useRuntimeOutputAlps)
    deferredValues.resize(static_cast<size_t>(std::max<int64_t>(N * C * outH * outW, int64_t{0})));

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t oh = 0; oh < outH; ++oh) {
        for (int64_t ow = 0; ow < outW; ++ow) {
          bool any = false;
          typename Fmt::UIntT best = zeroBits;
          double bestv = 0.0;

          int64_t ih0 = oh * stride_h - pad_top;
          int64_t iw0 = ow * stride_w - pad_left;
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            int64_t ih = ih0 + kh;
            if (ih < 0 || ih >= H)
              continue;
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              int64_t iw = iw0 + kw;
              if (iw < 0 || iw >= Wd)
                continue;
              auto bits = load4(x, n, c, ih, iw);
              double v = useMetaF32
                             ? static_cast<double>(static_cast<float>(
                                   decodeTensorValue<Fmt>(bits, xMeta)))
                             : decodeTensorValue<Fmt>(bits, xMeta);
              if (!any || v > bestv) {
                any = true;
                best = encodeTensorValue<Fmt>(v, outMeta);
                bestv = v;
              }
            }
          }
          if (useRuntimeOutputAlps) {
            deferredValues[static_cast<size_t>(sampleLin)] = any ? bestv : 0.0;
            ++sampleLin;
          } else {
            auto outBits = any ? best : zeroBits;
            store4(out, n, c, oh, ow, qalignBitsFromRaw<Fmt>(outBits, qalignKey, c));
            samples.maybeAppend(sampleLin++, -1, any ? bestv : 0.0,
                                any ? decodeTensorValue<Fmt>(best, outMeta) : 0.0);
          }
        }
      }
    }
  }
  if (useRuntimeOutputAlps) {
    outMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, outMeta);
    sampleLin = 0;
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t oh = 0; oh < outH; ++oh) {
          for (int64_t ow = 0; ow < outW; ++ow) {
            double v = deferredValues[static_cast<size_t>(sampleLin)];
            auto outBits = encodeTensorValue<Fmt>(v, outMeta);
            store4(out, n, c, oh, ow, qalignBitsFromRaw<Fmt>(outBits, qalignKey, c));
            samples.maybeAppend(sampleLin++, -1, v, decodeTensorValue<Fmt>(outBits, outMeta));
          }
        }
      }
    }
  }
  samples.flush();
  registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
}

template <typename Fmt>
static void clip_kernel(UnrankedMemRefType<typename Fmt::MemT> *In,
                        UnrankedMemRefType<typename Fmt::MemT> *Out, float minVal,
                        float maxVal, int64_t hasMin, int64_t hasMax,
                        int64_t qalignKey) {
  DynamicMemRefType<typename Fmt::MemT> in(*In), out(*Out);
  TensorGPMetadata inMeta = lookupTensorGPMetadata(tensorMetaPtr(in));
  TensorGPMetadata outMeta =
      chooseOutputTensorGPMetadata<Fmt>(qalignKey, inMeta);
  bool useMetaF32 = tensorMathUsesF32ForAlps(inMeta, outMeta);
  int64_t n = num_elems(in);
  QAlignOutputSampleBuffer samples = makeQAlignOutputSampleBuffer(qalignKey, n);
  const bool useRuntimeOutputAlps =
      useMetaF32 && positRuntimeOutputAlpsMode() != PositRuntimeOutputAlpsMode::Off &&
      fmtSupportsRuntimeOutputAlps<Fmt>();
  std::vector<double> deferredValues;
  if (useRuntimeOutputAlps)
    deferredValues.resize(static_cast<size_t>(std::max<int64_t>(n, 0)));
  for (int64_t i = 0; i < n; ++i) {
    double v = 0.0;
    if (useMetaF32) {
      float vf = static_cast<float>(
          decodeTensorValue<Fmt>(to_bits(in.data[offset_of(in, i)]), inMeta));
      if (hasMin)
        vf = std::max(vf, minVal);
      if (hasMax)
        vf = std::min(vf, maxVal);
      v = static_cast<double>(vf);
    } else {
      v = decodeTensorValue<Fmt>(to_bits(in.data[offset_of(in, i)]), inMeta);
      if (hasMin)
        v = std::max(v, static_cast<double>(minVal));
      if (hasMax)
        v = std::min(v, static_cast<double>(maxVal));
    }
    if (useRuntimeOutputAlps) {
      deferredValues[static_cast<size_t>(i)] = v;
    } else {
      auto outBits = encodeTensorValue<Fmt>(v, outMeta);
      out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(outBits);
      samples.maybeAppend(i, -1, v, decodeTensorValue<Fmt>(outBits, outMeta));
    }
  }
  if (useRuntimeOutputAlps) {
    outMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, outMeta);
    for (int64_t i = 0; i < n; ++i) {
      double v = deferredValues[static_cast<size_t>(i)];
      auto outBits = encodeTensorValue<Fmt>(v, outMeta);
      out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(outBits);
      samples.maybeAppend(i, -1, v, decodeTensorValue<Fmt>(outBits, outMeta));
    }
  }
  samples.flush();
  registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
}

template <typename Fmt>
static void reduce_mean_kernel(UnrankedMemRefType<typename Fmt::MemT> *In,
                               UnrankedMemRefType<typename Fmt::MemT> *Out,
                               int64_t axis0, int64_t axis1, int64_t keepdims,
                               int64_t qalignKey) {
  DynamicMemRefType<typename Fmt::MemT> in(*In), out(*Out);
  TensorGPMetadata inMeta = lookupTensorGPMetadata(tensorMetaPtr(in));
  TensorGPMetadata outMeta =
      chooseOutputTensorGPMetadata<Fmt>(qalignKey, inMeta);
  bool useMetaF32 = tensorMathUsesF32ForAlps(inMeta, outMeta);
  const bool useRuntimeOutputAlps =
      useMetaF32 && positRuntimeOutputAlpsMode() != PositRuntimeOutputAlpsMode::Off &&
      fmtSupportsRuntimeOutputAlps<Fmt>();
  QAlignOutputSampleBuffer samples =
      makeQAlignOutputSampleBuffer(qalignKey, num_elems(out));
  std::vector<double> deferredValues;
  if (!keepdims || in.rank != out.rank || in.rank <= 0) {
    // Conservative fallback.
    int64_t n = num_elems(in);
    double mean = 0.0;
    if (useMetaF32) {
      float acc = 0.0f;
      for (int64_t i = 0; i < n; ++i)
        acc += static_cast<float>(
            decodeTensorValue<Fmt>(to_bits(in.data[offset_of(in, i)]), inMeta));
      mean = (n > 0) ? static_cast<double>(acc / static_cast<float>(n)) : 0.0;
    } else {
      double acc = 0.0;
      for (int64_t i = 0; i < n; ++i)
        acc += decodeTensorValue<Fmt>(to_bits(in.data[offset_of(in, i)]), inMeta);
      mean = (n > 0) ? (acc / static_cast<double>(n)) : 0.0;
    }
    int64_t m = num_elems(out);
    if (useRuntimeOutputAlps) {
      deferredValues.assign(static_cast<size_t>(std::max<int64_t>(m, 0)), mean);
      outMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, outMeta);
      auto bits = encodeTensorValue<Fmt>(mean, outMeta);
      for (int64_t i = 0; i < m; ++i)
        out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(bits);
      if (m > 0)
        samples.maybeAppend(0, -1, mean, decodeTensorValue<Fmt>(bits, outMeta));
    } else {
      auto bits = encodeTensorValue<Fmt>(mean, outMeta);
      for (int64_t i = 0; i < m; ++i)
        out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(bits);
      if (m > 0)
        samples.maybeAppend(0, -1, mean, decodeTensorValue<Fmt>(bits, outMeta));
    }
    samples.flush();
    registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
    return;
  }

  int64_t r = in.rank;
  if (axis0 < 0)
    axis0 += r;
  if (axis1 < 0)
    axis1 += r;
  if (axis0 < 0 || axis0 >= r || axis1 < 0 || axis1 >= r) {
    int64_t n = num_elems(in);
    double mean = 0.0;
    if (useMetaF32) {
      float acc = 0.0f;
      for (int64_t i = 0; i < n; ++i)
        acc += static_cast<float>(
            decodeTensorValue<Fmt>(to_bits(in.data[offset_of(in, i)]), inMeta));
      mean = (n > 0) ? static_cast<double>(acc / static_cast<float>(n)) : 0.0;
    } else {
      double acc = 0.0;
      for (int64_t i = 0; i < n; ++i)
        acc += decodeTensorValue<Fmt>(to_bits(in.data[offset_of(in, i)]), inMeta);
      mean = (n > 0) ? (acc / static_cast<double>(n)) : 0.0;
    }
    int64_t m = num_elems(out);
    if (useRuntimeOutputAlps) {
      deferredValues.assign(static_cast<size_t>(std::max<int64_t>(m, 0)), mean);
      outMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, outMeta);
      auto bits = encodeTensorValue<Fmt>(mean, outMeta);
      for (int64_t i = 0; i < m; ++i)
        out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(bits);
      if (m > 0)
        samples.maybeAppend(0, -1, mean, decodeTensorValue<Fmt>(bits, outMeta));
    } else {
      auto bits = encodeTensorValue<Fmt>(mean, outMeta);
      for (int64_t i = 0; i < m; ++i)
        out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(bits);
      if (m > 0)
        samples.maybeAppend(0, -1, mean, decodeTensorValue<Fmt>(bits, outMeta));
    }
    samples.flush();
    registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
    return;
  }

  int64_t outElems = num_elems(out);
  if (useRuntimeOutputAlps)
    deferredValues.resize(static_cast<size_t>(std::max<int64_t>(outElems, 0)));
  std::vector<int64_t> outIdx(static_cast<size_t>(r), 0);
  for (int64_t lin = 0; lin < outElems; ++lin) {
    int64_t rem = lin;
    for (int64_t d = r - 1; d >= 0; --d) {
      outIdx[static_cast<size_t>(d)] = rem % out.sizes[d];
      rem /= out.sizes[d];
    }

    double mean = 0.0;
    int64_t cnt = 0;
    if (useMetaF32) {
      float acc = 0.0f;
      for (int64_t i0 = 0; i0 < in.sizes[axis0]; ++i0) {
        for (int64_t i1 = 0; i1 < in.sizes[axis1]; ++i1) {
          int64_t off = in.offset;
          for (int64_t d = 0; d < r; ++d) {
            int64_t idx = outIdx[static_cast<size_t>(d)];
            if (d == axis0)
              idx = i0;
            else if (d == axis1)
              idx = i1;
            off += idx * in.strides[d];
          }
          acc += static_cast<float>(decodeTensorValue<Fmt>(to_bits(in.data[off]), inMeta));
          ++cnt;
        }
      }
      mean = (cnt > 0) ? static_cast<double>(acc / static_cast<float>(cnt)) : 0.0;
    } else {
      double acc = 0.0;
      for (int64_t i0 = 0; i0 < in.sizes[axis0]; ++i0) {
        for (int64_t i1 = 0; i1 < in.sizes[axis1]; ++i1) {
          int64_t off = in.offset;
          for (int64_t d = 0; d < r; ++d) {
            int64_t idx = outIdx[static_cast<size_t>(d)];
            if (d == axis0)
              idx = i0;
            else if (d == axis1)
              idx = i1;
            off += idx * in.strides[d];
          }
          acc += decodeTensorValue<Fmt>(to_bits(in.data[off]), inMeta);
          ++cnt;
        }
      }
      mean = (cnt > 0) ? (acc / static_cast<double>(cnt)) : 0.0;
    }
    if (useRuntimeOutputAlps) {
      deferredValues[static_cast<size_t>(lin)] = mean;
    } else {
      auto outBits = encodeTensorValue<Fmt>(mean, outMeta);
      out.data[offset_of(out, lin)] = from_bits<typename Fmt::MemT>(outBits);
      samples.maybeAppend(lin, -1, mean, decodeTensorValue<Fmt>(outBits, outMeta));
    }
  }
  if (useRuntimeOutputAlps) {
    outMeta = chooseRuntimeOutputAlpsMetadata<Fmt>(qalignKey, deferredValues, outMeta);
    for (int64_t lin = 0; lin < outElems; ++lin) {
      double mean = deferredValues[static_cast<size_t>(lin)];
      auto outBits = encodeTensorValue<Fmt>(mean, outMeta);
      out.data[offset_of(out, lin)] = from_bits<typename Fmt::MemT>(outBits);
      samples.maybeAppend(lin, -1, mean, decodeTensorValue<Fmt>(outBits, outMeta));
    }
  }
  samples.flush();
  registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
}

template <typename Fmt>
static void dequantize_linear_kernel(UnrankedMemRefType<int8_t> *In,
                                     UnrankedMemRefType<typename Fmt::MemT> *Out,
                                     float scale, int64_t zeroPoint,
                                     int64_t hasZeroPoint, int64_t axis,
                                     int64_t inputSigned,
                                     int64_t qalignKey,
                                     UnrankedMemRefType<float> *OrigRef = nullptr) {
  DynamicMemRefType<int8_t> in(*In);
  DynamicMemRefType<typename Fmt::MemT> out(*Out);
  std::unique_ptr<DynamicMemRefType<float>> origRefMem;
  if (OrigRef)
    origRefMem = std::make_unique<DynamicMemRefType<float>>(*OrigRef);
  auto loadOrigRef = [&](int64_t lin) -> double {
    if (!origRefMem)
      return std::numeric_limits<double>::quiet_NaN();
    int64_t m = num_elems(*origRefMem);
    if (m <= 0)
      return std::numeric_limits<double>::quiet_NaN();
    int64_t idx = std::min<int64_t>(std::max<int64_t>(lin, 0), m - 1);
    return static_cast<double>(origRefMem->data[offset_of(*origRefMem, idx)]);
  };
  TensorGPMetadata outMeta = chooseOutputTensorGPMetadata<Fmt>(qalignKey);

  // ONNX DQ uses per-axis broadcasting semantics for scale/zero_point.
  // This runtime ABI currently passes scalar scale/zero_point, so this kernel
  // applies scalar values but keeps axis-normalization/indexing logic explicit.
  int64_t inRank = in.rank;
  if (inRank <= 0) {
    int64_t inOff = in.offset;
    int64_t outOff = out.offset;
    double qRaw = inputSigned ? static_cast<double>(in.data[inOff])
                              : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    double q = qRaw;
    if (hasZeroPoint)
      q -= static_cast<double>(zeroPoint);
    double rawV = q * static_cast<double>(scale);
    double origV = loadOrigRef(0);
    if (positQAlignCollectEnabled()) {
      if (std::isfinite(origV))
        collectQAlignBatch(
            kQAlignCollectOrig,
            qalignKey, std::vector<int64_t>{-1}, std::vector<double>{origV});
      collectQAlignBatch(
          kQAlignCollectDQ,
          qalignKey, std::vector<int64_t>{-1}, std::vector<double>{rawV});
    }
    double v = applyQAlignScaleSnap<Fmt>(rawV, qalignKey, -1,
        kQAlignProbeDQ, qRaw, static_cast<double>(scale),
        hasZeroPoint ? static_cast<double>(zeroPoint) : 0.0, "dq_posit_storage",
        origV);
    out.data[outOff] = from_bits<typename Fmt::MemT>(encodeTensorValue<Fmt>(v, outMeta));
    registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
    return;
  }

  int64_t normAxis = axis;
  if (normAxis < 0)
    normAxis += inRank;
  if (normAxis < 0)
    normAxis = 0;
  if (normAxis >= inRank)
    normAxis = inRank - 1;

  int64_t inElems = num_elems(in);
  int64_t outElems = num_elems(out);
  int64_t n = std::min(inElems, outElems);
  bool collect = positQAlignCollectEnabled();
  uint64_t perCallBudget = collect ? positQAlignCollectPerCall() : 0;
  int64_t sampleStride = 1;
  if (collect && perCallBudget > 0 && n > static_cast<int64_t>(perCallBudget))
    sampleStride = std::max<int64_t>(1, n / static_cast<int64_t>(perCallBudget));
  std::vector<int64_t> sampleChannels;
  std::vector<double> sampleValues;
  std::vector<double> sampleOrigValues;
  if (collect && perCallBudget > 0) {
    sampleChannels.reserve(static_cast<size_t>(perCallBudget));
    sampleValues.reserve(static_cast<size_t>(perCallBudget));
    sampleOrigValues.reserve(static_cast<size_t>(perCallBudget));
  }

  auto axis_index_from_linear = [&](int64_t lin) -> int64_t {
    int64_t rem = lin;
    int64_t axisIdx = 0;
    for (int64_t d = inRank - 1; d >= 0; --d) {
      int64_t dim = in.sizes[d];
      int64_t idx = (dim > 0) ? (rem % dim) : 0;
      rem = (dim > 0) ? (rem / dim) : 0;
      if (d == normAxis)
        axisIdx = idx;
    }
    return axisIdx;
  };

  for (int64_t i = 0; i < n; ++i) {
    int64_t inOff = offset_of(in, i);
    int64_t outOff = offset_of(out, i);
    int64_t axisIdx = axis_index_from_linear(i);
    (void)axisIdx; // Placeholder for future per-axis scale/zp buffers in ABI.

    double qRaw = inputSigned ? static_cast<double>(in.data[inOff])
                              : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    double q = qRaw;
    if (hasZeroPoint)
      q -= static_cast<double>(zeroPoint);
    double v = q * static_cast<double>(scale);
    double origV = loadOrigRef(i);
    if (collect && perCallBudget > 0 &&
        sampleValues.size() < static_cast<size_t>(perCallBudget) &&
        (i % sampleStride == 0)) {
      sampleChannels.push_back(-1);
      sampleValues.push_back(v);
      if (std::isfinite(origV))
        sampleOrigValues.push_back(origV);
    }
    v = applyQAlignScaleSnap<Fmt>(v, qalignKey, -1, kQAlignProbeDQ, qRaw,
        static_cast<double>(scale),
        hasZeroPoint ? static_cast<double>(zeroPoint) : 0.0, "dq_posit_storage",
        origV);
    out.data[outOff] = from_bits<typename Fmt::MemT>(encodeTensorValue<Fmt>(v, outMeta));
  }
  if (collect && !sampleValues.empty()) {
    if (sampleOrigValues.size() == sampleValues.size())
      collectQAlignBatch(kQAlignCollectOrig, qalignKey, sampleChannels, sampleOrigValues);
    collectQAlignBatch(kQAlignCollectDQ, qalignKey, sampleChannels, sampleValues);
  }
  registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
}

template <typename Fmt>
static void dequantize_linear_axis_kernel(
    UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<typename Fmt::MemT> *Out,
    UnrankedMemRefType<float> *Scale, UnrankedMemRefType<int64_t> *ZeroPoint,
    int64_t hasZeroPoint, int64_t axis, int64_t inputSigned,
    int64_t qalignKey,
    UnrankedMemRefType<float> *OrigRef = nullptr) {
  DynamicMemRefType<int8_t> in(*In);
  DynamicMemRefType<typename Fmt::MemT> out(*Out);
  DynamicMemRefType<float> scale(*Scale);
  DynamicMemRefType<int64_t> zp(*ZeroPoint);
  std::unique_ptr<DynamicMemRefType<float>> origRefMem;
  if (OrigRef)
    origRefMem = std::make_unique<DynamicMemRefType<float>>(*OrigRef);
  auto loadOrigRef = [&](int64_t lin) -> double {
    if (!origRefMem)
      return std::numeric_limits<double>::quiet_NaN();
    int64_t m = num_elems(*origRefMem);
    if (m <= 0)
      return std::numeric_limits<double>::quiet_NaN();
    int64_t idx = std::min<int64_t>(std::max<int64_t>(lin, 0), m - 1);
    return static_cast<double>(origRefMem->data[offset_of(*origRefMem, idx)]);
  };
  TensorGPMetadata outMeta = chooseOutputTensorGPMetadata<Fmt>(qalignKey);

  int64_t inRank = in.rank;
  if (inRank <= 0) {
    int64_t inOff = in.offset;
    int64_t outOff = out.offset;
    int64_t sOff = scale.offset;
    int64_t zOff = zp.offset;
    double s = static_cast<double>(scale.data[sOff]);
    double z = hasZeroPoint ? static_cast<double>(zp.data[zOff]) : 0.0;
    double qRaw = inputSigned ? static_cast<double>(in.data[inOff])
                              : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    double q = qRaw - z;
    double rawV = q * s;
    double origV = loadOrigRef(0);
    if (positQAlignCollectEnabled()) {
      if (std::isfinite(origV))
        collectQAlignBatch(
            kQAlignCollectOrig,
            qalignKey, std::vector<int64_t>{-1}, std::vector<double>{origV});
      collectQAlignBatch(
          kQAlignCollectDQ,
          qalignKey, std::vector<int64_t>{-1}, std::vector<double>{rawV});
    }
    double v = applyQAlignScaleSnap<Fmt>(
        rawV, qalignKey, -1, kQAlignProbeDQ, qRaw, s, z, "dq_posit_storage",
        origV);
    out.data[outOff] = from_bits<typename Fmt::MemT>(encodeTensorValue<Fmt>(v, outMeta));
    registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
    return;
  }

  int64_t normAxis = axis;
  if (normAxis < 0)
    normAxis += inRank;
  if (normAxis < 0)
    normAxis = 0;
  if (normAxis >= inRank)
    normAxis = inRank - 1;

  int64_t scaleElems = std::max<int64_t>(1, num_elems(scale));
  int64_t zpElems = std::max<int64_t>(1, num_elems(zp));
  int64_t inElems = num_elems(in);
  int64_t outElems = num_elems(out);
  int64_t n = std::min(inElems, outElems);
  bool collect = positQAlignCollectEnabled();
  uint64_t perCallBudget = collect ? positQAlignCollectPerCall() : 0;
  int64_t sampleStride = 1;
  if (collect && perCallBudget > 0 && n > static_cast<int64_t>(perCallBudget))
    sampleStride = std::max<int64_t>(1, n / static_cast<int64_t>(perCallBudget));
  std::vector<int64_t> sampleChannels;
  std::vector<double> sampleValues;
  std::vector<double> sampleOrigValues;
  if (collect && perCallBudget > 0) {
    sampleChannels.reserve(static_cast<size_t>(perCallBudget));
    sampleValues.reserve(static_cast<size_t>(perCallBudget));
    sampleOrigValues.reserve(static_cast<size_t>(perCallBudget));
  }

  auto axis_index_from_linear = [&](int64_t lin) -> int64_t {
    int64_t rem = lin;
    int64_t axisIdx = 0;
    for (int64_t d = inRank - 1; d >= 0; --d) {
      int64_t dim = in.sizes[d];
      int64_t idx = (dim > 0) ? (rem % dim) : 0;
      rem = (dim > 0) ? (rem / dim) : 0;
      if (d == normAxis)
        axisIdx = idx;
    }
    return axisIdx;
  };

  auto load_scale = [&](int64_t axisIdx) -> double {
    int64_t i = (scaleElems <= 1) ? 0 : std::min<int64_t>(axisIdx, scaleElems - 1);
    return static_cast<double>(scale.data[offset_of(scale, i)]);
  };

  auto load_zp = [&](int64_t axisIdx) -> double {
    if (!hasZeroPoint)
      return 0.0;
    int64_t i = (zpElems <= 1) ? 0 : std::min<int64_t>(axisIdx, zpElems - 1);
    return static_cast<double>(zp.data[offset_of(zp, i)]);
  };

  for (int64_t i = 0; i < n; ++i) {
    int64_t inOff = offset_of(in, i);
    int64_t outOff = offset_of(out, i);
    int64_t axisIdx = axis_index_from_linear(i);
    double s = load_scale(axisIdx);
    double z = load_zp(axisIdx);

    double q = inputSigned ? static_cast<double>(in.data[inOff])
                           : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    double v = (q - z) * s;
    double origV = loadOrigRef(i);
    if (collect && perCallBudget > 0 &&
        sampleValues.size() < static_cast<size_t>(perCallBudget) &&
        (i % sampleStride == 0)) {
      sampleChannels.push_back(axisIdx);
      sampleValues.push_back(v);
      if (std::isfinite(origV))
        sampleOrigValues.push_back(origV);
    }
    v = applyQAlignScaleSnap<Fmt>(
        v, qalignKey, axisIdx, kQAlignProbeDQ, q, s, z, "dq_posit_storage",
        origV);
    out.data[outOff] = from_bits<typename Fmt::MemT>(encodeTensorValue<Fmt>(v, outMeta));
  }
  if (collect && !sampleValues.empty()) {
    if (sampleOrigValues.size() == sampleValues.size())
      collectQAlignBatch(kQAlignCollectOrig, qalignKey, sampleChannels, sampleOrigValues);
    collectQAlignBatch(kQAlignCollectDQ, qalignKey, sampleChannels, sampleValues);
  }
  registerTensorGPMetadata(tensorMetaPtr(out), outMeta);
}

// QDQ-compatible f32-output DequantizeLinear kernels.
// These are used when posit.dequantize_linear has an f32 result type. They
// intentionally store runtime_x_dq into f32 output storage, mirroring ONNX INT8
// QDQ semantics (Q stores the low-precision code; DQ materializes f32; later
// Conv/Gemm/Add/Relu consume f32). The probe still records runtime_final so we
// can compare against the legacy posit-storage path.
template <typename Fmt>
static void dequantize_linear_f32_kernel(UnrankedMemRefType<int8_t> *In,
                                         UnrankedMemRefType<float> *Out,
                                         float scale, int64_t zeroPoint,
                                         int64_t hasZeroPoint, int64_t axis,
                                         int64_t inputSigned,
                                         int64_t qalignKey,
                                         UnrankedMemRefType<float> *OrigRef = nullptr) {
  DynamicMemRefType<int8_t> in(*In);
  DynamicMemRefType<float> out(*Out);
  std::unique_ptr<DynamicMemRefType<float>> origRefMem;
  if (OrigRef)
    origRefMem = std::make_unique<DynamicMemRefType<float>>(*OrigRef);
  auto loadOrigRef = [&](int64_t lin) -> double {
    if (!origRefMem)
      return std::numeric_limits<double>::quiet_NaN();
    int64_t m = num_elems(*origRefMem);
    if (m <= 0)
      return std::numeric_limits<double>::quiet_NaN();
    int64_t idx = std::min<int64_t>(std::max<int64_t>(lin, 0), m - 1);
    return static_cast<double>(origRefMem->data[offset_of(*origRefMem, idx)]);
  };

  int64_t inRank = in.rank;
  if (inRank <= 0) {
    int64_t inOff = in.offset;
    int64_t outOff = out.offset;
    double qRaw = inputSigned ? static_cast<double>(in.data[inOff])
                              : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    double q = qRaw;
    if (hasZeroPoint)
      q -= static_cast<double>(zeroPoint);
    double rawV = q * static_cast<double>(scale);
    double origV = loadOrigRef(0);
    if (positQAlignCollectEnabled()) {
      if (std::isfinite(origV))
        collectQAlignBatch(
            kQAlignCollectOrig, qalignKey, std::vector<int64_t>{-1},
            std::vector<double>{origV});
      collectQAlignBatch(
          kQAlignCollectDQ, qalignKey, std::vector<int64_t>{-1},
          std::vector<double>{rawV});
    }
    double v = applyQAlignScaleSnap<Fmt>(
        rawV, qalignKey, -1, kQAlignProbeDQ, qRaw, static_cast<double>(scale),
        hasZeroPoint ? static_cast<double>(zeroPoint) : 0.0, "dq_f32",
        origV);
    out.data[outOff] = static_cast<float>(v);
    // f32 outputs must not inherit stale GP metadata for the same data pointer.
    registerTensorGPMetadata(tensorMetaPtr(out), TensorGPMetadata{});
    return;
  }

  int64_t normAxis = axis;
  if (normAxis < 0)
    normAxis += inRank;
  if (normAxis < 0)
    normAxis = 0;
  if (normAxis >= inRank)
    normAxis = inRank - 1;

  int64_t inElems = num_elems(in);
  int64_t outElems = num_elems(out);
  int64_t n = std::min(inElems, outElems);
  bool collect = positQAlignCollectEnabled();
  uint64_t perCallBudget = collect ? positQAlignCollectPerCall() : 0;
  int64_t sampleStride = 1;
  if (collect && perCallBudget > 0 && n > static_cast<int64_t>(perCallBudget))
    sampleStride = std::max<int64_t>(1, n / static_cast<int64_t>(perCallBudget));
  std::vector<int64_t> sampleChannels;
  std::vector<double> sampleValues;
  std::vector<double> sampleOrigValues;
  if (collect && perCallBudget > 0) {
    sampleChannels.reserve(static_cast<size_t>(perCallBudget));
    sampleValues.reserve(static_cast<size_t>(perCallBudget));
    sampleOrigValues.reserve(static_cast<size_t>(perCallBudget));
  }

  (void)normAxis; // scalar ABI; kept for parity with posit-output kernel.
  for (int64_t i = 0; i < n; ++i) {
    int64_t inOff = offset_of(in, i);
    int64_t outOff = offset_of(out, i);
    double qRaw = inputSigned ? static_cast<double>(in.data[inOff])
                              : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    double q = qRaw;
    if (hasZeroPoint)
      q -= static_cast<double>(zeroPoint);
    double rawV = q * static_cast<double>(scale);
    double origV = loadOrigRef(i);
    if (collect && perCallBudget > 0 &&
        sampleValues.size() < static_cast<size_t>(perCallBudget) &&
        (i % sampleStride == 0)) {
      sampleChannels.push_back(-1);
      sampleValues.push_back(rawV);
      if (std::isfinite(origV))
        sampleOrigValues.push_back(origV);
    }
    double v = applyQAlignScaleSnap<Fmt>(
        rawV, qalignKey, -1, kQAlignProbeDQ, qRaw, static_cast<double>(scale),
        hasZeroPoint ? static_cast<double>(zeroPoint) : 0.0, "dq_f32",
        origV);
    out.data[outOff] = static_cast<float>(v);
  }
  if (collect && !sampleValues.empty()) {
    if (sampleOrigValues.size() == sampleValues.size())
      collectQAlignBatch(kQAlignCollectOrig, qalignKey, sampleChannels, sampleOrigValues);
    collectQAlignBatch(kQAlignCollectDQ, qalignKey, sampleChannels, sampleValues);
  }
  registerTensorGPMetadata(tensorMetaPtr(out), TensorGPMetadata{});
}

template <typename Fmt>
static void dequantize_linear_axis_f32_kernel(
    UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<float> *Out,
    UnrankedMemRefType<float> *Scale, UnrankedMemRefType<int64_t> *ZeroPoint,
    int64_t hasZeroPoint, int64_t axis, int64_t inputSigned,
    int64_t qalignKey,
    UnrankedMemRefType<float> *OrigRef = nullptr) {
  DynamicMemRefType<int8_t> in(*In);
  DynamicMemRefType<float> out(*Out);
  DynamicMemRefType<float> scale(*Scale);
  DynamicMemRefType<int64_t> zp(*ZeroPoint);
  std::unique_ptr<DynamicMemRefType<float>> origRefMem;
  if (OrigRef)
    origRefMem = std::make_unique<DynamicMemRefType<float>>(*OrigRef);
  auto loadOrigRef = [&](int64_t lin) -> double {
    if (!origRefMem)
      return std::numeric_limits<double>::quiet_NaN();
    int64_t m = num_elems(*origRefMem);
    if (m <= 0)
      return std::numeric_limits<double>::quiet_NaN();
    int64_t idx = std::min<int64_t>(std::max<int64_t>(lin, 0), m - 1);
    return static_cast<double>(origRefMem->data[offset_of(*origRefMem, idx)]);
  };

  int64_t inRank = in.rank;
  if (inRank <= 0) {
    int64_t inOff = in.offset;
    int64_t outOff = out.offset;
    int64_t sOff = scale.offset;
    int64_t zOff = zp.offset;
    double s = static_cast<double>(scale.data[sOff]);
    double z = hasZeroPoint ? static_cast<double>(zp.data[zOff]) : 0.0;
    double qRaw = inputSigned ? static_cast<double>(in.data[inOff])
                              : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    double rawV = (qRaw - z) * s;
    double origV = loadOrigRef(0);
    if (positQAlignCollectEnabled()) {
      if (std::isfinite(origV))
        collectQAlignBatch(
            kQAlignCollectOrig, qalignKey, std::vector<int64_t>{-1},
            std::vector<double>{origV});
      collectQAlignBatch(
          kQAlignCollectDQ, qalignKey, std::vector<int64_t>{-1},
          std::vector<double>{rawV});
    }
    double v = applyQAlignScaleSnap<Fmt>(
        rawV, qalignKey, -1, kQAlignProbeDQ, qRaw, s, z, "dq_f32",
        origV);
    out.data[outOff] = static_cast<float>(v);
    registerTensorGPMetadata(tensorMetaPtr(out), TensorGPMetadata{});
    return;
  }

  int64_t normAxis = axis;
  if (normAxis < 0)
    normAxis += inRank;
  if (normAxis < 0)
    normAxis = 0;
  if (normAxis >= inRank)
    normAxis = inRank - 1;

  int64_t scaleElems = std::max<int64_t>(1, num_elems(scale));
  int64_t zpElems = std::max<int64_t>(1, num_elems(zp));
  int64_t inElems = num_elems(in);
  int64_t outElems = num_elems(out);
  int64_t n = std::min(inElems, outElems);
  bool collect = positQAlignCollectEnabled();
  uint64_t perCallBudget = collect ? positQAlignCollectPerCall() : 0;
  int64_t sampleStride = 1;
  if (collect && perCallBudget > 0 && n > static_cast<int64_t>(perCallBudget))
    sampleStride = std::max<int64_t>(1, n / static_cast<int64_t>(perCallBudget));
  std::vector<int64_t> sampleChannels;
  std::vector<double> sampleValues;
  std::vector<double> sampleOrigValues;
  if (collect && perCallBudget > 0) {
    sampleChannels.reserve(static_cast<size_t>(perCallBudget));
    sampleValues.reserve(static_cast<size_t>(perCallBudget));
    sampleOrigValues.reserve(static_cast<size_t>(perCallBudget));
  }

  auto axis_index_from_linear = [&](int64_t lin) -> int64_t {
    int64_t rem = lin;
    int64_t axisIdx = 0;
    for (int64_t d = inRank - 1; d >= 0; --d) {
      int64_t dim = in.sizes[d];
      int64_t idx = (dim > 0) ? (rem % dim) : 0;
      rem = (dim > 0) ? (rem / dim) : 0;
      if (d == normAxis)
        axisIdx = idx;
    }
    return axisIdx;
  };

  auto load_scale = [&](int64_t axisIdx) -> double {
    int64_t i = (scaleElems <= 1) ? 0 : std::min<int64_t>(axisIdx, scaleElems - 1);
    return static_cast<double>(scale.data[offset_of(scale, i)]);
  };

  auto load_zp = [&](int64_t axisIdx) -> double {
    if (!hasZeroPoint)
      return 0.0;
    int64_t i = (zpElems <= 1) ? 0 : std::min<int64_t>(axisIdx, zpElems - 1);
    return static_cast<double>(zp.data[offset_of(zp, i)]);
  };

  for (int64_t i = 0; i < n; ++i) {
    int64_t inOff = offset_of(in, i);
    int64_t outOff = offset_of(out, i);
    int64_t axisIdx = axis_index_from_linear(i);
    double s = load_scale(axisIdx);
    double z = load_zp(axisIdx);
    double qRaw = inputSigned ? static_cast<double>(in.data[inOff])
                              : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    double rawV = (qRaw - z) * s;
    double origV = loadOrigRef(i);
    if (collect && perCallBudget > 0 &&
        sampleValues.size() < static_cast<size_t>(perCallBudget) &&
        (i % sampleStride == 0)) {
      sampleChannels.push_back(axisIdx);
      sampleValues.push_back(rawV);
      if (std::isfinite(origV))
        sampleOrigValues.push_back(origV);
    }
    double v = applyQAlignScaleSnap<Fmt>(
        rawV, qalignKey, axisIdx, kQAlignProbeDQ, qRaw, s, z, "dq_f32",
        origV);
    out.data[outOff] = static_cast<float>(v);
  }
  if (collect && !sampleValues.empty()) {
    if (sampleOrigValues.size() == sampleValues.size())
      collectQAlignBatch(kQAlignCollectOrig, qalignKey, sampleChannels, sampleOrigValues);
    collectQAlignBatch(kQAlignCollectDQ, qalignKey, sampleChannels, sampleValues);
  }
  registerTensorGPMetadata(tensorMetaPtr(out), TensorGPMetadata{});
}

//===----------------------------------------------------------------------===//
// Export wrappers
//===----------------------------------------------------------------------===//

#define DEFINE_POSIT_RUNTIME_EXPORTS(TAG, FMT, MEMTY)                                             \
  extern "C" void _mlir_ciface_posit_from_f32_##TAG(                                              \
      UnrankedMemRefType<float> *In, UnrankedMemRefType<MEMTY> *Out, int64_t qalignKey) {         \
    posit_from_f32_kernel<FMT>(In, Out, qalignKey);                                                \
    traceMemrefSummary<FMT>("from_f32_" #TAG, DynamicMemRefType<MEMTY>(*Out));                    \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_to_f32_##TAG(                                                \
      UnrankedMemRefType<MEMTY> *In, UnrankedMemRefType<float> *Out) {                            \
    posit_to_f32_kernel<FMT>(In, Out);                                                             \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_add_##TAG(                                                   \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *Out, int64_t qalignKey) {                                        \
    elemwise_kernel<FMT>(ElemwiseOpKind::Add, A, B, Out, qalignKey);                              \
    traceMemrefSummary<FMT>("add_" #TAG, DynamicMemRefType<MEMTY>(*Out));                         \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_sub_##TAG(                                                   \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *Out, int64_t qalignKey) {                                        \
    elemwise_kernel<FMT>(ElemwiseOpKind::Sub, A, B, Out, qalignKey);                              \
    traceMemrefSummary<FMT>("sub_" #TAG, DynamicMemRefType<MEMTY>(*Out));                         \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_mul_##TAG(                                                   \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *Out, int64_t qalignKey) {                                        \
    elemwise_kernel<FMT>(ElemwiseOpKind::Mul, A, B, Out, qalignKey);                              \
    traceMemrefSummary<FMT>("mul_" #TAG, DynamicMemRefType<MEMTY>(*Out));                         \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_div_##TAG(                                                   \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *Out, int64_t qalignKey) {                                        \
    elemwise_kernel<FMT>(ElemwiseOpKind::Div, A, B, Out, qalignKey);                              \
    traceMemrefSummary<FMT>("div_" #TAG, DynamicMemRefType<MEMTY>(*Out));                         \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_relu_##TAG(                                                  \
      UnrankedMemRefType<MEMTY> *In, UnrankedMemRefType<MEMTY> *Out,                              \
      int64_t qalignKey) {                                                                         \
    relu_kernel<FMT>(In, Out, qalignKey);                                                          \
    traceMemrefSummary<FMT>("relu_" #TAG, DynamicMemRefType<MEMTY>(*Out));                        \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_gemm_##TAG(                                                  \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *C, UnrankedMemRefType<MEMTY> *Y,                                 \
      float alpha, float beta, int64_t transA, int64_t transB, int64_t qalignKey) {               \
    if (!A || !B || !Y || !A->descriptor || !B->descriptor || !Y->descriptor)                     \
      return;                                                                                      \
    MEMTY zeroStorage = from_bits<MEMTY>(bits_from_double_fmt<FMT>(0.0));                         \
    StridedMemRefType<MEMTY, 0> zeroDesc;                                                          \
    zeroDesc.basePtr = &zeroStorage;                                                               \
    zeroDesc.data = &zeroStorage;                                                                  \
    zeroDesc.offset = 0;                                                                           \
    UnrankedMemRefType<MEMTY> zeroUnranked;                                                        \
    zeroUnranked.rank = 0;                                                                         \
    zeroUnranked.descriptor = &zeroDesc;                                                           \
    UnrankedMemRefType<MEMTY> *effectiveC =                                                        \
        (C && C->descriptor) ? C : &zeroUnranked;                                                  \
    gemm_kernel<FMT>(A, B, effectiveC, Y, alpha, beta, transA, transB, qalignKey);                \
    { DynamicMemRefType<MEMTY> _gcoll(*Y);                                                        \
      collectLayerRangeValues<FMT>(qalignKey, "gemm", _gcoll); }                                  \
    if (auto *_poclamp = lookupPositOutputClamp(qalignKey)) {                                     \
      DynamicMemRefType<MEMTY> _pocout(*Y);                                                       \
      applyPositOutputClamp<FMT>(_pocout, *_poclamp);                                             \
    }                                                                                             \
    traceMemrefSummary<FMT>("gemm_" #TAG, DynamicMemRefType<MEMTY>(*Y));                          \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_conv2d_nchw_##TAG(                                           \
      UnrankedMemRefType<MEMTY> *X, UnrankedMemRefType<MEMTY> *W,                                 \
      UnrankedMemRefType<MEMTY> *B, UnrankedMemRefType<MEMTY> *Out,                               \
      int64_t stride_h, int64_t stride_w, int64_t dilation_h, int64_t dilation_w,                 \
      int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,                   \
      int64_t groups, int64_t qalignKey) {                                                         \
    conv2d_nchw_kernel<FMT>(X, W, B, Out, stride_h, stride_w, dilation_h, dilation_w,             \
                            pad_top, pad_left, pad_bottom, pad_right, groups, qalignKey);          \
    { DynamicMemRefType<MEMTY> _ccoll(*Out);                                                      \
      collectLayerRangeValues<FMT>(qalignKey, "conv2d", _ccoll); }                                \
    if (auto *_poclamp = lookupPositOutputClamp(qalignKey)) {                                     \
      DynamicMemRefType<MEMTY> _pocout(*Out);                                                     \
      applyPositOutputClamp<FMT>(_pocout, *_poclamp);                                             \
    }                                                                                             \
    traceMemrefSummary<FMT>("conv2d_nchw_" #TAG, DynamicMemRefType<MEMTY>(*Out));                 \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_maxpool2d_nchw_##TAG(                                        \
      UnrankedMemRefType<MEMTY> *X, UnrankedMemRefType<MEMTY> *Out,                               \
      int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,                     \
      int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,                   \
      int64_t ceil_mode, int64_t qalignKey) {                                                      \
    maxpool2d_nchw_kernel<FMT>(X, Out, kernel_h, kernel_w, stride_h, stride_w,                    \
                               pad_top, pad_left, pad_bottom, pad_right, ceil_mode, qalignKey);   \
    traceMemrefSummary<FMT>("maxpool2d_nchw_" #TAG, DynamicMemRefType<MEMTY>(*Out));              \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_clip_##TAG(                                                  \
      UnrankedMemRefType<MEMTY> *In, UnrankedMemRefType<MEMTY> *Out, float minVal,                \
      float maxVal, int64_t hasMin, int64_t hasMax, int64_t qalignKey) {                          \
    clip_kernel<FMT>(In, Out, minVal, maxVal, hasMin, hasMax, qalignKey);                         \
    if (positDotProbeEnabled() && qalignKey != 0) {                                               \
      DynamicMemRefType<MEMTY> _clipOut(*Out);                                                    \
      int64_t _n=num_elems(_clipOut), _r=_clipOut.rank;                                           \
      int64_t _d0=_r>0?_clipOut.sizes[0]:0, _d1=_r>1?_clipOut.sizes[1]:0;                        \
      int64_t _d2=_r>2?_clipOut.sizes[2]:0, _d3=_r>3?_clipOut.sizes[3]:0;                        \
      recordDotProbe<FMT>("clip",qalignKey,"clip",false,false,false,false,_d0,_d1,_d2,_d3,0,_n); \
    }                                                                                             \
    if (auto *_poclamp = lookupPositOutputClamp(qalignKey)) {                                     \
      DynamicMemRefType<MEMTY> _pocout(*Out);                                                     \
      applyPositOutputClamp<FMT>(_pocout, *_poclamp);                                             \
    }                                                                                             \
    traceMemrefSummary<FMT>("clip_" #TAG, DynamicMemRefType<MEMTY>(*Out));                        \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_reduce_mean_##TAG(                                            \
      UnrankedMemRefType<MEMTY> *In, UnrankedMemRefType<MEMTY> *Out, int64_t axis0,               \
      int64_t axis1, int64_t keepdims, int64_t qalignKey) {                                        \
    reduce_mean_kernel<FMT>(In, Out, axis0, axis1, keepdims, qalignKey);                          \
    traceMemrefSummary<FMT>("reduce_mean_" #TAG, DynamicMemRefType<MEMTY>(*Out));                 \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_dequantize_linear_##TAG(                                      \
      UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<MEMTY> *Out, float scale,                \
      int64_t zeroPoint, int64_t hasZeroPoint, int64_t axis, int64_t inputSigned,                  \
      int64_t qalignKey) {                                                                          \
    dequantize_linear_kernel<FMT>(In, Out, scale, zeroPoint, hasZeroPoint, axis, inputSigned,      \
                                  qalignKey);                                                       \
    traceMemrefSummary<FMT>("dequantize_linear_" #TAG, DynamicMemRefType<MEMTY>(*Out));           \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_dequantize_linear_ref_##TAG(                                  \
      UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<MEMTY> *Out,                              \
      UnrankedMemRefType<float> *OrigRef, float scale, int64_t zeroPoint,                          \
      int64_t hasZeroPoint, int64_t axis, int64_t inputSigned, int64_t qalignKey) {                \
    dequantize_linear_kernel<FMT>(In, Out, scale, zeroPoint, hasZeroPoint, axis, inputSigned,      \
                                  qalignKey, OrigRef);                                             \
    traceMemrefSummary<FMT>("dequantize_linear_ref_" #TAG, DynamicMemRefType<MEMTY>(*Out));       \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_dequantize_linear_axis_##TAG(                                 \
      UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<MEMTY> *Out,                              \
      UnrankedMemRefType<float> *Scale, UnrankedMemRefType<int64_t> *ZeroPoint,                    \
      int64_t hasZeroPoint, int64_t axis, int64_t inputSigned, int64_t qalignKey) {                \
    dequantize_linear_axis_kernel<FMT>(In, Out, Scale, ZeroPoint, hasZeroPoint, axis, inputSigned, \
                                       qalignKey);                                                  \
    traceMemrefSummary<FMT>("dequantize_linear_axis_" #TAG, DynamicMemRefType<MEMTY>(*Out));      \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_dequantize_linear_axis_ref_##TAG(                             \
      UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<MEMTY> *Out,                              \
      UnrankedMemRefType<float> *OrigRef, UnrankedMemRefType<float> *Scale,                        \
      UnrankedMemRefType<int64_t> *ZeroPoint, int64_t hasZeroPoint, int64_t axis,                  \
      int64_t inputSigned, int64_t qalignKey) {                                                    \
    dequantize_linear_axis_kernel<FMT>(In, Out, Scale, ZeroPoint, hasZeroPoint, axis, inputSigned, \
                                       qalignKey, OrigRef);                                        \
    traceMemrefSummary<FMT>("dequantize_linear_axis_ref_" #TAG, DynamicMemRefType<MEMTY>(*Out));  \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_dequantize_linear_f32_##TAG(                                  \
      UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<float> *Out, float scale,                 \
      int64_t zeroPoint, int64_t hasZeroPoint, int64_t axis, int64_t inputSigned,                  \
      int64_t qalignKey) {                                                                          \
    dequantize_linear_f32_kernel<FMT>(In, Out, scale, zeroPoint, hasZeroPoint, axis, inputSigned,  \
                                      qalignKey);                                                   \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_dequantize_linear_f32_ref_##TAG(                              \
      UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<float> *Out,                              \
      UnrankedMemRefType<float> *OrigRef, float scale, int64_t zeroPoint,                          \
      int64_t hasZeroPoint, int64_t axis, int64_t inputSigned, int64_t qalignKey) {                \
    dequantize_linear_f32_kernel<FMT>(In, Out, scale, zeroPoint, hasZeroPoint, axis, inputSigned,  \
                                      qalignKey, OrigRef);                                         \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_dequantize_linear_axis_f32_##TAG(                             \
      UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<float> *Out,                              \
      UnrankedMemRefType<float> *Scale, UnrankedMemRefType<int64_t> *ZeroPoint,                    \
      int64_t hasZeroPoint, int64_t axis, int64_t inputSigned, int64_t qalignKey) {                \
    dequantize_linear_axis_f32_kernel<FMT>(In, Out, Scale, ZeroPoint, hasZeroPoint, axis,          \
                                           inputSigned, qalignKey);                                 \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_dequantize_linear_axis_f32_ref_##TAG(                         \
      UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<float> *Out,                              \
      UnrankedMemRefType<float> *OrigRef, UnrankedMemRefType<float> *Scale,                        \
      UnrankedMemRefType<int64_t> *ZeroPoint, int64_t hasZeroPoint, int64_t axis,                  \
      int64_t inputSigned, int64_t qalignKey) {                                                    \
    dequantize_linear_axis_f32_kernel<FMT>(In, Out, Scale, ZeroPoint, hasZeroPoint, axis,          \
                                           inputSigned, qalignKey, OrigRef);                       \
  }

#if defined(POSIT_RUNTIME_SINGLE_FORMAT)
#define POSIT_RUNTIME_ENABLE_P4E0 0
#define POSIT_RUNTIME_ENABLE_P4E1 0
#define POSIT_RUNTIME_ENABLE_P4E2 0
#define POSIT_RUNTIME_ENABLE_P4E3 0
#define POSIT_RUNTIME_ENABLE_P5E0 0
#define POSIT_RUNTIME_ENABLE_P5E1 0
#define POSIT_RUNTIME_ENABLE_P5E2 0
#define POSIT_RUNTIME_ENABLE_P5E3 0
#define POSIT_RUNTIME_ENABLE_P6E0 0
#define POSIT_RUNTIME_ENABLE_P6E1 0
#define POSIT_RUNTIME_ENABLE_P6E2 0
#define POSIT_RUNTIME_ENABLE_P6E3 0
#define POSIT_RUNTIME_ENABLE_P7E0 0
#define POSIT_RUNTIME_ENABLE_P7E1 0
#define POSIT_RUNTIME_ENABLE_P7E2 0
#define POSIT_RUNTIME_ENABLE_P7E3 0
#define POSIT_RUNTIME_ENABLE_P8E0 0
#define POSIT_RUNTIME_ENABLE_P8E1 0
#define POSIT_RUNTIME_ENABLE_P8E2 0
#define POSIT_RUNTIME_ENABLE_P9E0 0
#define POSIT_RUNTIME_ENABLE_P9E1 0
#define POSIT_RUNTIME_ENABLE_P9E2 0
#define POSIT_RUNTIME_ENABLE_P9E3 0
#define POSIT_RUNTIME_ENABLE_P10E0 0
#define POSIT_RUNTIME_ENABLE_P10E1 0
#define POSIT_RUNTIME_ENABLE_P10E2 0
#define POSIT_RUNTIME_ENABLE_P11E0 0
#define POSIT_RUNTIME_ENABLE_P11E1 0
#define POSIT_RUNTIME_ENABLE_P11E2 0
#define POSIT_RUNTIME_ENABLE_P12E0 0
#define POSIT_RUNTIME_ENABLE_P12E1 0
#define POSIT_RUNTIME_ENABLE_P12E2 0
#define POSIT_RUNTIME_ENABLE_P13E0 0
#define POSIT_RUNTIME_ENABLE_P13E1 0
#define POSIT_RUNTIME_ENABLE_P13E2 0
#define POSIT_RUNTIME_ENABLE_P14E0 0
#define POSIT_RUNTIME_ENABLE_P14E1 0
#define POSIT_RUNTIME_ENABLE_P14E2 0
#define POSIT_RUNTIME_ENABLE_P15E0 0
#define POSIT_RUNTIME_ENABLE_P15E1 0
#define POSIT_RUNTIME_ENABLE_P15E2 0
#define POSIT_RUNTIME_ENABLE_P16E0 0
#define POSIT_RUNTIME_ENABLE_P16E1 0
#define POSIT_RUNTIME_ENABLE_P16E2 0
#define POSIT_RUNTIME_ENABLE_P32E0 0
#define POSIT_RUNTIME_ENABLE_P32E1 0
#define POSIT_RUNTIME_ENABLE_P32E2 0
#if defined(POSIT_RUNTIME_FMT_P4E0)
#undef POSIT_RUNTIME_ENABLE_P4E0
#define POSIT_RUNTIME_ENABLE_P4E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P4E1)
#undef POSIT_RUNTIME_ENABLE_P4E1
#define POSIT_RUNTIME_ENABLE_P4E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P4E2)
#undef POSIT_RUNTIME_ENABLE_P4E2
#define POSIT_RUNTIME_ENABLE_P4E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P4E3)
#undef POSIT_RUNTIME_ENABLE_P4E3
#define POSIT_RUNTIME_ENABLE_P4E3 1
#endif
#if defined(POSIT_RUNTIME_FMT_P5E0)
#undef POSIT_RUNTIME_ENABLE_P5E0
#define POSIT_RUNTIME_ENABLE_P5E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P5E1)
#undef POSIT_RUNTIME_ENABLE_P5E1
#define POSIT_RUNTIME_ENABLE_P5E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P5E2)
#undef POSIT_RUNTIME_ENABLE_P5E2
#define POSIT_RUNTIME_ENABLE_P5E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P5E3)
#undef POSIT_RUNTIME_ENABLE_P5E3
#define POSIT_RUNTIME_ENABLE_P5E3 1
#endif
#if defined(POSIT_RUNTIME_FMT_P6E0)
#undef POSIT_RUNTIME_ENABLE_P6E0
#define POSIT_RUNTIME_ENABLE_P6E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P6E1)
#undef POSIT_RUNTIME_ENABLE_P6E1
#define POSIT_RUNTIME_ENABLE_P6E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P6E2)
#undef POSIT_RUNTIME_ENABLE_P6E2
#define POSIT_RUNTIME_ENABLE_P6E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P6E3)
#undef POSIT_RUNTIME_ENABLE_P6E3
#define POSIT_RUNTIME_ENABLE_P6E3 1
#endif
#if defined(POSIT_RUNTIME_FMT_P7E0)
#undef POSIT_RUNTIME_ENABLE_P7E0
#define POSIT_RUNTIME_ENABLE_P7E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P7E1)
#undef POSIT_RUNTIME_ENABLE_P7E1
#define POSIT_RUNTIME_ENABLE_P7E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P7E2)
#undef POSIT_RUNTIME_ENABLE_P7E2
#define POSIT_RUNTIME_ENABLE_P7E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P7E3)
#undef POSIT_RUNTIME_ENABLE_P7E3
#define POSIT_RUNTIME_ENABLE_P7E3 1
#endif
#if defined(POSIT_RUNTIME_FMT_P8E0)
#undef POSIT_RUNTIME_ENABLE_P8E0
#define POSIT_RUNTIME_ENABLE_P8E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P8E1)
#undef POSIT_RUNTIME_ENABLE_P8E1
#define POSIT_RUNTIME_ENABLE_P8E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P8E2)
#undef POSIT_RUNTIME_ENABLE_P8E2
#define POSIT_RUNTIME_ENABLE_P8E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P9E0)
#undef POSIT_RUNTIME_ENABLE_P9E0
#define POSIT_RUNTIME_ENABLE_P9E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P9E1)
#undef POSIT_RUNTIME_ENABLE_P9E1
#define POSIT_RUNTIME_ENABLE_P9E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P9E2)
#undef POSIT_RUNTIME_ENABLE_P9E2
#define POSIT_RUNTIME_ENABLE_P9E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P9E3)
#undef POSIT_RUNTIME_ENABLE_P9E3
#define POSIT_RUNTIME_ENABLE_P9E3 1
#endif
#if defined(POSIT_RUNTIME_FMT_P10E0)
#undef POSIT_RUNTIME_ENABLE_P10E0
#define POSIT_RUNTIME_ENABLE_P10E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P10E1)
#undef POSIT_RUNTIME_ENABLE_P10E1
#define POSIT_RUNTIME_ENABLE_P10E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P10E2)
#undef POSIT_RUNTIME_ENABLE_P10E2
#define POSIT_RUNTIME_ENABLE_P10E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P11E0)
#undef POSIT_RUNTIME_ENABLE_P11E0
#define POSIT_RUNTIME_ENABLE_P11E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P11E1)
#undef POSIT_RUNTIME_ENABLE_P11E1
#define POSIT_RUNTIME_ENABLE_P11E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P11E2)
#undef POSIT_RUNTIME_ENABLE_P11E2
#define POSIT_RUNTIME_ENABLE_P11E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P12E0)
#undef POSIT_RUNTIME_ENABLE_P12E0
#define POSIT_RUNTIME_ENABLE_P12E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P12E1)
#undef POSIT_RUNTIME_ENABLE_P12E1
#define POSIT_RUNTIME_ENABLE_P12E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P12E2)
#undef POSIT_RUNTIME_ENABLE_P12E2
#define POSIT_RUNTIME_ENABLE_P12E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P13E0)
#undef POSIT_RUNTIME_ENABLE_P13E0
#define POSIT_RUNTIME_ENABLE_P13E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P13E1)
#undef POSIT_RUNTIME_ENABLE_P13E1
#define POSIT_RUNTIME_ENABLE_P13E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P13E2)
#undef POSIT_RUNTIME_ENABLE_P13E2
#define POSIT_RUNTIME_ENABLE_P13E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P14E0)
#undef POSIT_RUNTIME_ENABLE_P14E0
#define POSIT_RUNTIME_ENABLE_P14E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P14E1)
#undef POSIT_RUNTIME_ENABLE_P14E1
#define POSIT_RUNTIME_ENABLE_P14E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P14E2)
#undef POSIT_RUNTIME_ENABLE_P14E2
#define POSIT_RUNTIME_ENABLE_P14E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P15E0)
#undef POSIT_RUNTIME_ENABLE_P15E0
#define POSIT_RUNTIME_ENABLE_P15E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P15E1)
#undef POSIT_RUNTIME_ENABLE_P15E1
#define POSIT_RUNTIME_ENABLE_P15E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P15E2)
#undef POSIT_RUNTIME_ENABLE_P15E2
#define POSIT_RUNTIME_ENABLE_P15E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P16E0)
#undef POSIT_RUNTIME_ENABLE_P16E0
#define POSIT_RUNTIME_ENABLE_P16E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P16E1)
#undef POSIT_RUNTIME_ENABLE_P16E1
#define POSIT_RUNTIME_ENABLE_P16E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P16E2)
#undef POSIT_RUNTIME_ENABLE_P16E2
#define POSIT_RUNTIME_ENABLE_P16E2 1
#endif
#if defined(POSIT_RUNTIME_FMT_P32E0)
#undef POSIT_RUNTIME_ENABLE_P32E0
#define POSIT_RUNTIME_ENABLE_P32E0 1
#endif
#if defined(POSIT_RUNTIME_FMT_P32E1)
#undef POSIT_RUNTIME_ENABLE_P32E1
#define POSIT_RUNTIME_ENABLE_P32E1 1
#endif
#if defined(POSIT_RUNTIME_FMT_P32E2)
#undef POSIT_RUNTIME_ENABLE_P32E2
#define POSIT_RUNTIME_ENABLE_P32E2 1
#endif
#else
#define POSIT_RUNTIME_ENABLE_P4E0 1
#define POSIT_RUNTIME_ENABLE_P4E1 1
#define POSIT_RUNTIME_ENABLE_P4E2 1
#define POSIT_RUNTIME_ENABLE_P4E3 1
#define POSIT_RUNTIME_ENABLE_P5E0 1
#define POSIT_RUNTIME_ENABLE_P5E1 1
#define POSIT_RUNTIME_ENABLE_P5E2 1
#define POSIT_RUNTIME_ENABLE_P5E3 1
#define POSIT_RUNTIME_ENABLE_P6E0 1
#define POSIT_RUNTIME_ENABLE_P6E1 1
#define POSIT_RUNTIME_ENABLE_P6E2 1
#define POSIT_RUNTIME_ENABLE_P6E3 1
#define POSIT_RUNTIME_ENABLE_P7E0 1
#define POSIT_RUNTIME_ENABLE_P7E1 1
#define POSIT_RUNTIME_ENABLE_P7E2 1
#define POSIT_RUNTIME_ENABLE_P7E3 1
#define POSIT_RUNTIME_ENABLE_P8E0 1
#define POSIT_RUNTIME_ENABLE_P8E1 1
#define POSIT_RUNTIME_ENABLE_P8E2 1
#define POSIT_RUNTIME_ENABLE_P9E0 1
#define POSIT_RUNTIME_ENABLE_P9E1 1
#define POSIT_RUNTIME_ENABLE_P9E2 1
#define POSIT_RUNTIME_ENABLE_P9E3 1
#define POSIT_RUNTIME_ENABLE_P10E0 1
#define POSIT_RUNTIME_ENABLE_P10E1 1
#define POSIT_RUNTIME_ENABLE_P10E2 1
#define POSIT_RUNTIME_ENABLE_P11E0 1
#define POSIT_RUNTIME_ENABLE_P11E1 1
#define POSIT_RUNTIME_ENABLE_P11E2 1
#define POSIT_RUNTIME_ENABLE_P12E0 1
#define POSIT_RUNTIME_ENABLE_P12E1 1
#define POSIT_RUNTIME_ENABLE_P12E2 1
#define POSIT_RUNTIME_ENABLE_P13E0 1
#define POSIT_RUNTIME_ENABLE_P13E1 1
#define POSIT_RUNTIME_ENABLE_P13E2 1
#define POSIT_RUNTIME_ENABLE_P14E0 1
#define POSIT_RUNTIME_ENABLE_P14E1 1
#define POSIT_RUNTIME_ENABLE_P14E2 1
#define POSIT_RUNTIME_ENABLE_P15E0 1
#define POSIT_RUNTIME_ENABLE_P15E1 1
#define POSIT_RUNTIME_ENABLE_P15E2 1
#define POSIT_RUNTIME_ENABLE_P16E0 1
#define POSIT_RUNTIME_ENABLE_P16E1 1
#define POSIT_RUNTIME_ENABLE_P16E2 1
#define POSIT_RUNTIME_ENABLE_P32E0 1
#define POSIT_RUNTIME_ENABLE_P32E1 1
#define POSIT_RUNTIME_ENABLE_P32E2 1
#endif

#if POSIT_RUNTIME_ENABLE_P8E0
DEFINE_POSIT_RUNTIME_EXPORTS(p8e0, FmtP8E0, int8_t)
#endif
#if defined(POSIT_USE_UNIVERSAL)
#if POSIT_RUNTIME_ENABLE_P4E0
DEFINE_POSIT_RUNTIME_EXPORTS(p4e0, FmtP4E0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P4E1
DEFINE_POSIT_RUNTIME_EXPORTS(p4e1, FmtP4E1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P4E2
DEFINE_POSIT_RUNTIME_EXPORTS(p4e2, FmtP4E2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P4E3
DEFINE_POSIT_RUNTIME_EXPORTS(p4e3, FmtP4E3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E0
DEFINE_POSIT_RUNTIME_EXPORTS(p5e0, FmtP5E0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E1
DEFINE_POSIT_RUNTIME_EXPORTS(p5e1, FmtP5E1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E2
DEFINE_POSIT_RUNTIME_EXPORTS(p5e2, FmtP5E2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E3
DEFINE_POSIT_RUNTIME_EXPORTS(p5e3, FmtP5E3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E0
DEFINE_POSIT_RUNTIME_EXPORTS(p6e0, FmtP6E0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E1
DEFINE_POSIT_RUNTIME_EXPORTS(p6e1, FmtP6E1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E2
DEFINE_POSIT_RUNTIME_EXPORTS(p6e2, FmtP6E2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E3
DEFINE_POSIT_RUNTIME_EXPORTS(p6e3, FmtP6E3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E0
DEFINE_POSIT_RUNTIME_EXPORTS(p7e0, FmtP7E0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E1
DEFINE_POSIT_RUNTIME_EXPORTS(p7e1, FmtP7E1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E2
DEFINE_POSIT_RUNTIME_EXPORTS(p7e2, FmtP7E2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E3
DEFINE_POSIT_RUNTIME_EXPORTS(p7e3, FmtP7E3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P8E1
DEFINE_POSIT_RUNTIME_EXPORTS(p8e1, FmtP8E1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P8E2
DEFINE_POSIT_RUNTIME_EXPORTS(p8e2, FmtP8E2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E0
DEFINE_POSIT_RUNTIME_EXPORTS(p9e0, FmtP9E0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E1
DEFINE_POSIT_RUNTIME_EXPORTS(p9e1, FmtP9E1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E2
DEFINE_POSIT_RUNTIME_EXPORTS(p9e2, FmtP9E2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E3
DEFINE_POSIT_RUNTIME_EXPORTS(p9e3, FmtP9E3, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P10E0
DEFINE_POSIT_RUNTIME_EXPORTS(p10e0, FmtP10E0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P10E1
DEFINE_POSIT_RUNTIME_EXPORTS(p10e1, FmtP10E1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P10E2
DEFINE_POSIT_RUNTIME_EXPORTS(p10e2, FmtP10E2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P11E0
DEFINE_POSIT_RUNTIME_EXPORTS(p11e0, FmtP11E0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P11E1
DEFINE_POSIT_RUNTIME_EXPORTS(p11e1, FmtP11E1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P11E2
DEFINE_POSIT_RUNTIME_EXPORTS(p11e2, FmtP11E2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P12E0
DEFINE_POSIT_RUNTIME_EXPORTS(p12e0, FmtP12E0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P12E1
DEFINE_POSIT_RUNTIME_EXPORTS(p12e1, FmtP12E1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P12E2
DEFINE_POSIT_RUNTIME_EXPORTS(p12e2, FmtP12E2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P13E0
DEFINE_POSIT_RUNTIME_EXPORTS(p13e0, FmtP13E0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P13E1
DEFINE_POSIT_RUNTIME_EXPORTS(p13e1, FmtP13E1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P13E2
DEFINE_POSIT_RUNTIME_EXPORTS(p13e2, FmtP13E2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P14E0
DEFINE_POSIT_RUNTIME_EXPORTS(p14e0, FmtP14E0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P14E1
DEFINE_POSIT_RUNTIME_EXPORTS(p14e1, FmtP14E1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P14E2
DEFINE_POSIT_RUNTIME_EXPORTS(p14e2, FmtP14E2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P15E0
DEFINE_POSIT_RUNTIME_EXPORTS(p15e0, FmtP15E0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P15E1
DEFINE_POSIT_RUNTIME_EXPORTS(p15e1, FmtP15E1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P15E2
DEFINE_POSIT_RUNTIME_EXPORTS(p15e2, FmtP15E2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E0
DEFINE_POSIT_RUNTIME_EXPORTS(p16e0, FmtP16E0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E1
DEFINE_POSIT_RUNTIME_EXPORTS(p16e1, FmtP16E1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E2
DEFINE_POSIT_RUNTIME_EXPORTS(p16e2, FmtP16E2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E0
DEFINE_POSIT_RUNTIME_EXPORTS(p32e0, FmtP32E0, int32_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E1
DEFINE_POSIT_RUNTIME_EXPORTS(p32e1, FmtP32E1, int32_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E2
DEFINE_POSIT_RUNTIME_EXPORTS(p32e2, FmtP32E2, int32_t)
#endif
#elif defined(POSIT_USE_SOFTPOSIT_PX1) || defined(POSIT_USE_SOFTPOSIT_PX2)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
#if POSIT_RUNTIME_ENABLE_P8E1
DEFINE_POSIT_RUNTIME_EXPORTS(p8e1, FmtP8E1ViaPX1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E1
DEFINE_POSIT_RUNTIME_EXPORTS(p32e1, FmtP32E1ViaPX1, int32_t)
#endif
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
#if POSIT_RUNTIME_ENABLE_P8E2
DEFINE_POSIT_RUNTIME_EXPORTS(p8e2, FmtP8E2ViaPX2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E2
DEFINE_POSIT_RUNTIME_EXPORTS(p16e2, FmtP16E2ViaPX2, int16_t)
#endif
#endif
#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
#if POSIT_RUNTIME_ENABLE_P16E0
DEFINE_POSIT_RUNTIME_EXPORTS(p16e0, FmtP16E0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E0
DEFINE_POSIT_RUNTIME_EXPORTS(p32e0, FmtP32E0, int32_t)
#endif
#endif
#if POSIT_RUNTIME_ENABLE_P16E1
DEFINE_POSIT_RUNTIME_EXPORTS(p16e1, FmtP16E1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E2
DEFINE_POSIT_RUNTIME_EXPORTS(p32e2, FmtP32E2, int32_t)
#endif
#else
#if POSIT_RUNTIME_ENABLE_P16E1
DEFINE_POSIT_RUNTIME_EXPORTS(p16e1, FmtP16E1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E2
DEFINE_POSIT_RUNTIME_EXPORTS(p32e2, FmtP32E2, int32_t)
#endif
#endif

#define DEFINE_POSIT_COMPACT_COMPAND_EXPORT(TAG, MEMTY)                                           \
  extern "C" void _mlir_ciface_posit_register_tensor_compand_##TAG(                              \
      UnrankedMemRefType<MEMTY> *tensor, int64_t mode, double theta, double gamma) {             \
    DynamicMemRefType<MEMTY> t(*tensor);                                                         \
    registerTensorCompandMetadata(t, mode, theta, gamma);                                        \
  }

#define DEFINE_POSIT_CONST_META_EXPORT(TAG, MEMTY)                                                 \
  extern "C" void _mlir_ciface_posit_register_tensor_constmeta_##TAG(                             \
      UnrankedMemRefType<MEMTY> *tensor, int64_t mode, double theta, double gamma,               \
      int64_t gpEnabled, int64_t gpRs, int64_t gpSc) {                                           \
    DynamicMemRefType<MEMTY> t(*tensor);                                                          \
    registerTensorConstMetadata(t, mode, theta, gamma, gpEnabled, gpRs, gpSc);                   \
  }

#define DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(TAG, MEMTY)                                         \
  extern "C" void _mlir_ciface_posit_register_tensor_constmeta_channel_##TAG(                     \
      UnrankedMemRefType<MEMTY> *tensor, int64_t channel, int64_t mode,                           \
      double theta, double gamma, int64_t gpEnabled, int64_t gpRs, int64_t gpSc) {              \
    DynamicMemRefType<MEMTY> t(*tensor);                                                          \
    registerTensorConstMetadataForChannel(                                                        \
        t, channel, mode, theta, gamma, gpEnabled, gpRs, gpSc);                                  \
  }

#if POSIT_RUNTIME_ENABLE_P8E0
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p8e0, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p8e0, int8_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p8e0, int8_t)
#endif
#if defined(POSIT_USE_UNIVERSAL)
#if POSIT_RUNTIME_ENABLE_P4E0
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p4e0, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p4e0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P4E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p4e1, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p4e1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P4E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p4e2, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p4e2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P4E3
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p4e3, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p4e3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E0
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p5e0, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p5e0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p5e1, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p5e1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p5e2, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p5e2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E3
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p5e3, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p5e3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E0
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p6e0, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p6e0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p6e1, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p6e1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p6e2, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p6e2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E3
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p6e3, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p6e3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E0
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p7e0, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p7e0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p7e1, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p7e1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p7e2, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p7e2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E3
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p7e3, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p7e3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P8E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p8e1, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p8e1, int8_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p8e1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P8E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p8e2, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p8e2, int8_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p8e2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E0
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p9e0, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p9e0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p9e1, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p9e1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p9e2, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p9e2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E3
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p9e3, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p9e3, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E0
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p16e0, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p16e0, int16_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p16e0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p16e1, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p16e1, int16_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p16e1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p16e2, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p16e2, int16_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p16e2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E0
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p32e0, int32_t)
DEFINE_POSIT_CONST_META_EXPORT(p32e0, int32_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p32e0, int32_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p32e1, int32_t)
DEFINE_POSIT_CONST_META_EXPORT(p32e1, int32_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p32e1, int32_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p32e2, int32_t)
DEFINE_POSIT_CONST_META_EXPORT(p32e2, int32_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p32e2, int32_t)
#endif
#elif defined(POSIT_USE_SOFTPOSIT_PX1) || defined(POSIT_USE_SOFTPOSIT_PX2)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
#if POSIT_RUNTIME_ENABLE_P8E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p8e1, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p8e1, int8_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p8e1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p32e1, int32_t)
DEFINE_POSIT_CONST_META_EXPORT(p32e1, int32_t)
#endif
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
#if POSIT_RUNTIME_ENABLE_P8E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p8e2, int8_t)
DEFINE_POSIT_CONST_META_EXPORT(p8e2, int8_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p8e2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p16e2, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p16e2, int16_t)
#endif
#endif
#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
#if POSIT_RUNTIME_ENABLE_P16E0
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p16e0, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p16e0, int16_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p16e0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E0
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p32e0, int32_t)
DEFINE_POSIT_CONST_META_EXPORT(p32e0, int32_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p32e0, int32_t)
#endif
#endif
#if POSIT_RUNTIME_ENABLE_P16E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p16e1, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p16e1, int16_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p16e1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p32e2, int32_t)
DEFINE_POSIT_CONST_META_EXPORT(p32e2, int32_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p32e2, int32_t)
#endif
#else
#if POSIT_RUNTIME_ENABLE_P16E1
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p16e1, int16_t)
DEFINE_POSIT_CONST_META_EXPORT(p16e1, int16_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p16e1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E2
DEFINE_POSIT_COMPACT_COMPAND_EXPORT(p32e2, int32_t)
DEFINE_POSIT_CONST_META_EXPORT(p32e2, int32_t)
DEFINE_POSIT_CONST_META_CHANNEL_EXPORT(p32e2, int32_t)
#endif
#endif

#undef DEFINE_POSIT_COMPACT_COMPAND_EXPORT
#undef DEFINE_POSIT_CONST_META_EXPORT
#undef DEFINE_POSIT_CONST_META_CHANNEL_EXPORT

#define DEFINE_POSIT_CIFACE_FORWARDERS(TAG, MEMTY)                                                 \
  extern "C" {                                                                                     \
  void _mlir_ciface__mlir_ciface_posit_from_f32_##TAG(UnrankedMemRefType<float> *in,              \
                                                      UnrankedMemRefType<MEMTY> *out,              \
                                                      int64_t qalignKey) {                          \
    _mlir_ciface_posit_from_f32_##TAG(in, out, qalignKey);                                         \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_to_f32_##TAG(UnrankedMemRefType<MEMTY> *in,                \
                                                    UnrankedMemRefType<float> *out) {              \
    _mlir_ciface_posit_to_f32_##TAG(in, out);                                                      \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_add_##TAG(UnrankedMemRefType<MEMTY> *a,                    \
                                                 UnrankedMemRefType<MEMTY> *b,                     \
                                                 UnrankedMemRefType<MEMTY> *out) {                 \
    _mlir_ciface_posit_add_##TAG(a, b, out, 0);                                                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_sub_##TAG(UnrankedMemRefType<MEMTY> *a,                    \
                                                 UnrankedMemRefType<MEMTY> *b,                     \
                                                 UnrankedMemRefType<MEMTY> *out) {                 \
    _mlir_ciface_posit_sub_##TAG(a, b, out, 0);                                                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_mul_##TAG(UnrankedMemRefType<MEMTY> *a,                    \
                                                 UnrankedMemRefType<MEMTY> *b,                     \
                                                 UnrankedMemRefType<MEMTY> *out) {                 \
    _mlir_ciface_posit_mul_##TAG(a, b, out, 0);                                                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_div_##TAG(UnrankedMemRefType<MEMTY> *a,                    \
                                                 UnrankedMemRefType<MEMTY> *b,                     \
                                                 UnrankedMemRefType<MEMTY> *out) {                 \
    _mlir_ciface_posit_div_##TAG(a, b, out, 0);                                                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_relu_##TAG(UnrankedMemRefType<MEMTY> *in,                  \
                                                  UnrankedMemRefType<MEMTY> *out) {                \
    _mlir_ciface_posit_relu_##TAG(in, out, 0);                                                     \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_conv2d_nchw_##TAG(                                          \
      UnrankedMemRefType<MEMTY> *X, UnrankedMemRefType<MEMTY> *W,                                 \
      UnrankedMemRefType<MEMTY> *B, UnrankedMemRefType<MEMTY> *Out,                               \
      int64_t stride_h, int64_t stride_w, int64_t dilat_h, int64_t dilat_w,                       \
      int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right, int64_t group) { \
    _mlir_ciface_posit_conv2d_nchw_##TAG(X, W, B, Out, stride_h, stride_w, dilat_h, dilat_w,      \
                                         pad_top, pad_left, pad_bottom, pad_right, group, 0);      \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_maxpool2d_nchw_##TAG(                                       \
      UnrankedMemRefType<MEMTY> *X, UnrankedMemRefType<MEMTY> *Out,                               \
      int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,                     \
      int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,                   \
      int64_t ceil_mode) {                                                                         \
    _mlir_ciface_posit_maxpool2d_nchw_##TAG(X, Out, kernel_h, kernel_w, stride_h, stride_w,       \
                                            pad_top, pad_left, pad_bottom, pad_right, ceil_mode,   \
                                            0);                                                     \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_gemm_##TAG(                                                 \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *C, UnrankedMemRefType<MEMTY> *Y,                                 \
      float alpha, float beta, int64_t transA, int64_t transB) {                                  \
    _mlir_ciface_posit_gemm_##TAG(A, B, C, Y, alpha, beta, transA, transB, 0);                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_clip_##TAG(                                                 \
      UnrankedMemRefType<MEMTY> *in, UnrankedMemRefType<MEMTY> *out,                              \
      float minVal, float maxVal, int64_t hasMin, int64_t hasMax) {                               \
    _mlir_ciface_posit_clip_##TAG(in, out, minVal, maxVal, hasMin, hasMax, 0);                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_reduce_mean_##TAG(                                          \
      UnrankedMemRefType<MEMTY> *in, UnrankedMemRefType<MEMTY> *out,                              \
      int64_t axis0, int64_t axis1, int64_t keepdims) {                                            \
    _mlir_ciface_posit_reduce_mean_##TAG(in, out, axis0, axis1, keepdims, 0);                     \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_dequantize_linear_##TAG(                                    \
      UnrankedMemRefType<int8_t> *in, UnrankedMemRefType<MEMTY> *out, float scale,                \
      int64_t zeroPoint, int64_t hasZeroPoint, int64_t axis, int64_t inputSigned,                  \
      int64_t qalignKey) {                                                                          \
    _mlir_ciface_posit_dequantize_linear_##TAG(                                                    \
        in, out, scale, zeroPoint, hasZeroPoint, axis, inputSigned, qalignKey);                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_dequantize_linear_ref_##TAG(                                \
      UnrankedMemRefType<int8_t> *in, UnrankedMemRefType<MEMTY> *out,                              \
      UnrankedMemRefType<float> *origRef, float scale, int64_t zeroPoint,                          \
      int64_t hasZeroPoint, int64_t axis, int64_t inputSigned, int64_t qalignKey) {                \
    _mlir_ciface_posit_dequantize_linear_ref_##TAG(                                                \
        in, out, origRef, scale, zeroPoint, hasZeroPoint, axis, inputSigned, qalignKey);           \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_dequantize_linear_axis_##TAG(                               \
      UnrankedMemRefType<int8_t> *in, UnrankedMemRefType<MEMTY> *out,                              \
      UnrankedMemRefType<float> *scale, UnrankedMemRefType<int64_t> *zeroPoint,                    \
      int64_t hasZeroPoint, int64_t axis, int64_t inputSigned, int64_t qalignKey) {                \
    _mlir_ciface_posit_dequantize_linear_axis_##TAG(                                               \
        in, out, scale, zeroPoint, hasZeroPoint, axis, inputSigned, qalignKey);                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_dequantize_linear_axis_ref_##TAG(                           \
      UnrankedMemRefType<int8_t> *in, UnrankedMemRefType<MEMTY> *out,                              \
      UnrankedMemRefType<float> *origRef, UnrankedMemRefType<float> *scale,                        \
      UnrankedMemRefType<int64_t> *zeroPoint, int64_t hasZeroPoint, int64_t axis,                  \
      int64_t inputSigned, int64_t qalignKey) {                                                    \
    _mlir_ciface_posit_dequantize_linear_axis_ref_##TAG(                                           \
        in, out, origRef, scale, zeroPoint, hasZeroPoint, axis, inputSigned, qalignKey);           \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_dequantize_linear_f32_##TAG(                                \
      UnrankedMemRefType<int8_t> *in, UnrankedMemRefType<float> *out, float scale,                 \
      int64_t zeroPoint, int64_t hasZeroPoint, int64_t axis, int64_t inputSigned,                  \
      int64_t qalignKey) {                                                                         \
    _mlir_ciface_posit_dequantize_linear_f32_##TAG(                                                \
        in, out, scale, zeroPoint, hasZeroPoint, axis, inputSigned, qalignKey);                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_dequantize_linear_f32_ref_##TAG(                            \
      UnrankedMemRefType<int8_t> *in, UnrankedMemRefType<float> *out,                              \
      UnrankedMemRefType<float> *origRef, float scale, int64_t zeroPoint,                          \
      int64_t hasZeroPoint, int64_t axis, int64_t inputSigned, int64_t qalignKey) {                \
    _mlir_ciface_posit_dequantize_linear_f32_ref_##TAG(                                            \
        in, out, origRef, scale, zeroPoint, hasZeroPoint, axis, inputSigned, qalignKey);           \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_dequantize_linear_axis_f32_##TAG(                           \
      UnrankedMemRefType<int8_t> *in, UnrankedMemRefType<float> *out,                              \
      UnrankedMemRefType<float> *scale, UnrankedMemRefType<int64_t> *zeroPoint,                    \
      int64_t hasZeroPoint, int64_t axis, int64_t inputSigned, int64_t qalignKey) {                \
    _mlir_ciface_posit_dequantize_linear_axis_f32_##TAG(                                           \
        in, out, scale, zeroPoint, hasZeroPoint, axis, inputSigned, qalignKey);                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_dequantize_linear_axis_f32_ref_##TAG(                       \
      UnrankedMemRefType<int8_t> *in, UnrankedMemRefType<float> *out,                              \
      UnrankedMemRefType<float> *origRef, UnrankedMemRefType<float> *scale,                        \
      UnrankedMemRefType<int64_t> *zeroPoint, int64_t hasZeroPoint, int64_t axis,                  \
      int64_t inputSigned, int64_t qalignKey) {                                                    \
    _mlir_ciface_posit_dequantize_linear_axis_f32_ref_##TAG(                                       \
        in, out, origRef, scale, zeroPoint, hasZeroPoint, axis, inputSigned, qalignKey);           \
  }                                                                                                \
  }

#if POSIT_RUNTIME_ENABLE_P8E0
DEFINE_POSIT_CIFACE_FORWARDERS(p8e0, int8_t)
#endif
#if defined(POSIT_USE_UNIVERSAL) || defined(POSIT_USE_SOFTPOSIT_PX1)
#if POSIT_RUNTIME_ENABLE_P8E1
DEFINE_POSIT_CIFACE_FORWARDERS(p8e1, int8_t)
#endif
#endif
#if defined(POSIT_USE_UNIVERSAL) || defined(POSIT_USE_SOFTPOSIT_PX2)
#if POSIT_RUNTIME_ENABLE_P8E2
DEFINE_POSIT_CIFACE_FORWARDERS(p8e2, int8_t)
#endif
#endif
#if defined(POSIT_USE_UNIVERSAL)
#if POSIT_RUNTIME_ENABLE_P4E0
DEFINE_POSIT_CIFACE_FORWARDERS(p4e0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P4E1
DEFINE_POSIT_CIFACE_FORWARDERS(p4e1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P4E2
DEFINE_POSIT_CIFACE_FORWARDERS(p4e2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P4E3
DEFINE_POSIT_CIFACE_FORWARDERS(p4e3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E0
DEFINE_POSIT_CIFACE_FORWARDERS(p5e0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E1
DEFINE_POSIT_CIFACE_FORWARDERS(p5e1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E2
DEFINE_POSIT_CIFACE_FORWARDERS(p5e2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P5E3
DEFINE_POSIT_CIFACE_FORWARDERS(p5e3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E0
DEFINE_POSIT_CIFACE_FORWARDERS(p6e0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E1
DEFINE_POSIT_CIFACE_FORWARDERS(p6e1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E2
DEFINE_POSIT_CIFACE_FORWARDERS(p6e2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P6E3
DEFINE_POSIT_CIFACE_FORWARDERS(p6e3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E0
DEFINE_POSIT_CIFACE_FORWARDERS(p7e0, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E1
DEFINE_POSIT_CIFACE_FORWARDERS(p7e1, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E2
DEFINE_POSIT_CIFACE_FORWARDERS(p7e2, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P7E3
DEFINE_POSIT_CIFACE_FORWARDERS(p7e3, int8_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E0
DEFINE_POSIT_CIFACE_FORWARDERS(p9e0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E1
DEFINE_POSIT_CIFACE_FORWARDERS(p9e1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E2
DEFINE_POSIT_CIFACE_FORWARDERS(p9e2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P9E3
DEFINE_POSIT_CIFACE_FORWARDERS(p9e3, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E0
DEFINE_POSIT_CIFACE_FORWARDERS(p16e0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E1
DEFINE_POSIT_CIFACE_FORWARDERS(p16e1, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P16E2
DEFINE_POSIT_CIFACE_FORWARDERS(p16e2, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E0
DEFINE_POSIT_CIFACE_FORWARDERS(p32e0, int32_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E1
DEFINE_POSIT_CIFACE_FORWARDERS(p32e1, int32_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E2
DEFINE_POSIT_CIFACE_FORWARDERS(p32e2, int32_t)
#endif
#else
#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
#if POSIT_RUNTIME_ENABLE_P16E0
DEFINE_POSIT_CIFACE_FORWARDERS(p16e0, int16_t)
#endif
#if POSIT_RUNTIME_ENABLE_P32E0
DEFINE_POSIT_CIFACE_FORWARDERS(p32e0, int32_t)
#endif
#endif
#if POSIT_RUNTIME_ENABLE_P16E1
DEFINE_POSIT_CIFACE_FORWARDERS(p16e1, int16_t)
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
#if POSIT_RUNTIME_ENABLE_P16E2
DEFINE_POSIT_CIFACE_FORWARDERS(p16e2, int16_t)
#endif
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX1)
#if POSIT_RUNTIME_ENABLE_P32E1
DEFINE_POSIT_CIFACE_FORWARDERS(p32e1, int32_t)
#endif
#endif
#if POSIT_RUNTIME_ENABLE_P32E2
DEFINE_POSIT_CIFACE_FORWARDERS(p32e2, int32_t)
#endif
#endif

#if !defined(POSIT_USE_UNIVERSAL) && !defined(POSIT_USE_SOFTPOSIT_PX1) && !defined(POSIT_USE_SOFTPOSIT_PX2)
// [EXTEND] SoftPosit default API does not provide p8e1/p8e2 directly. To satisfy .ll
// files that declare _mlir_ciface__mlir_ciface_posit_*_p8e1/p8e2, rebuild with one of:
//   -DPOSIT_USE_UNIVERSAL
//   -DPOSIT_USE_SOFTPOSIT_PX1  (requires libsoftposit with pX1_* symbols)
//   -DPOSIT_USE_SOFTPOSIT_PX2  (requires libsoftposit with pX2_* / qX2_* symbols)
#endif

#if defined(POSIT_RUNTIME_OUTPUT_ALPS_CALIB_TOOL)
int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr,
                 "Usage: %s <runtime_output_alps_collect.csv> <params.csv>\n",
                 argv[0]);
    return 2;
  }

  std::ifstream fin(argv[1]);
  if (!fin.is_open()) {
    std::fprintf(stderr,
                 "[runtime-output-alps] cannot open collect csv: %s\n",
                 argv[1]);
    return 2;
  }

  {
    std::lock_guard<std::mutex> lock(gRuntimeOutputAlpsCollectMutex);
    gRuntimeOutputAlpsCollectBuckets.clear();
  }

  std::string line;
  bool first = true;
  while (std::getline(fin, line)) {
    if (line.empty())
      continue;
    if (first) {
      first = false;
      if (line.rfind("format,key,channel,seen,sample_count,samples", 0) == 0)
        continue;
    }
    auto cols = splitCsvLineSimple(line);
    if (cols.size() < 6)
      continue;

    RuntimeOutputAlpsBucketKey key;
    key.format = cols[0];
    key.key = std::strtoll(cols[1].c_str(), nullptr, 10);
    key.channel = std::strtoll(cols[2].c_str(), nullptr, 10);

    RuntimeOutputAlpsCollectBucket bucket;
    bucket.seen = static_cast<uint64_t>(
        std::strtoull(cols[3].c_str(), nullptr, 10));

    std::stringstream sampleSs(cols[5]);
    std::string tok;
    while (std::getline(sampleSs, tok, ';')) {
      tok = trimAscii(tok);
      if (tok.empty())
        continue;
      char *end = nullptr;
      double v = std::strtod(tok.c_str(), &end);
      if (!end || *end != '\0' || !std::isfinite(v))
        continue;
      bucket.samples.push_back(static_cast<float>(v));
    }

    std::lock_guard<std::mutex> lock(gRuntimeOutputAlpsCollectMutex);
    auto &dst = gRuntimeOutputAlpsCollectBuckets[key];
    dst.seen += bucket.seen;
    dst.samples.insert(dst.samples.end(), bucket.samples.begin(),
                       bucket.samples.end());
  }

  dumpRuntimeOutputAlpsCalibToPath(argv[2]);
  return 0;
}
#endif
