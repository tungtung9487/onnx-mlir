# ImageNet100 ONNX models (reproduction evidence)

Exact ONNX models used by the posit/ALPS experiments driven by
`experiments/imagenet100/reproduce_alps.sh`. Provided so the results can be
reproduced from the identical weights rather than a fresh (variance-prone)
training run.

| File | What it is |
|------|------------|
| `imagenet100_mobilenetv2.onnx` | MobileNetV2, f32 (non-QDQ) export |
| `imagenet100_mobilenetv2-int8-qdq.onnx` | MobileNetV2, static QDQ int8 (per-channel, MinMax, QUInt8 act / QInt8 weight) |
| `imagenet100_resnet18.onnx` | ResNet18, f32 (non-QDQ) export |
| `imagenet100_resnet18-int8-qdq.onnx` | ResNet18, static QDQ int8 (same recipe) |

Integrity: verify with `sha256sum -c SHA256SUMS.txt` from this directory.

The QDQ files were produced from the f32 ONNX by `quantize_imagenet100_qdq.py`
using the first N sorted validation samples as the MinMax calibration set (see
that script and the repo `CLAUDE.md` for the exact recipe).
