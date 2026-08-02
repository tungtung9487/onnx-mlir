# 從零重現 MobileNetV2 + ALPS 實驗(可攜、小參數快速版)

在一台**全新環境**從頭跑到「帶 ALPS 參數的結果」。所有指令用一個 `$WS`(workspace 根目錄)
變數,**沒有寫死絕對路徑**,你只要改第一行的 `WS`。

> 佈局假設:`$WS/onnx-mlir`(本 repo)和 `$WS/ImageNet100`(工作目錄)為**同層 sibling**。
> 若不同層,build wrapper 可用 `ONNX_MLIR_ROOT=/path/to/onnx-mlir` 覆寫。

小參數(為了快):θ 範圍 **0.01~10**、steps 60、GP 縮小、評估 **500 張**、offline 收集 **50 張**。

---

## 0. 設定 workspace 根目錄(每個新 shell 都先跑這行)

```bash
export WS="$HOME/onnx_mlir"     # ← 改成你要放的路徑
mkdir -p "$WS" && cd "$WS"
```

## 1. 系統依賴(Ubuntu 例)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build clang lld \
                        python3 python3-venv python3-pip git
```

## 2. 取得程式碼

```bash
cd "$WS"
# 本 posit 分支
git clone -b tungtung9487/posit-work-20260329 \
  https://github.com/tungtung9487/onnx-mlir.git onnx-mlir
# LLVM/MLIR(onnx-mlir 釘選的 commit)
git clone https://github.com/llvm/llvm-project.git
git -C llvm-project checkout 113f01aa82d055410f22a9d03b3468fa68600589
```

## 3. Build LLVM/MLIR(最耗時,約數十分鐘~數小時)

細節見 `onnx-mlir/docs/BuildOnLinuxOSX.md`;關鍵指令:

```bash
cd "$WS/llvm-project"
mkdir -p build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON
cmake --build . -- ${MAKEFLAGS:--j$(nproc)}
```

## 4. Build onnx-mlir + posit 依賴

```bash
# 4a. posit 數值庫(Universal / SoftPosit → 放進 src/.deps,已 gitignore)
bash "$WS/onnx-mlir/src/bash/install_posit_deps.sh"

# 4b. 編 onnx-mlir-opt + onnx-mlir
cd "$WS/onnx-mlir"
mkdir -p build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR="$WS/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$WS/llvm-project/build/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --target onnx-mlir-opt onnx-mlir -- -j4
```
產出 `build/Release/bin/onnx-mlir-opt`、`onnx-mlir`。

## 5. Python 環境 + 資料集 + 模型

在 `$WS/ImageNet100` 準備工作目錄(腳本在 repo 的 `experiments/imagenet100/`,複製過去用):

```bash
mkdir -p "$WS/ImageNet100" && cd "$WS/ImageNet100"
cp "$WS/onnx-mlir/experiments/imagenet100/"*.py "$WS/ImageNet100/"
cp "$WS/onnx-mlir/experiments/imagenet100/build_imagenet100_mobilenetv2_11_sos.sh" "$WS/ImageNet100/"

# venv
python3 -m venv gpt2 && ./gpt2/bin/pip install -U pip
./gpt2/bin/pip install torch torchvision timm datasets onnx onnxruntime numpy pillow

# 資料集(HuggingFace clane9/imagenet-100 → imagenet100_hf/{train,validation})
./gpt2/bin/python download_dataset.py

# f32 訓練(快測可用少 epoch;完整見 experiments/imagenet100/README.md)
./gpt2/bin/python train_imagenet100_mobilenetv2.py \
  --train-dir imagenet100_hf/train --val-dir imagenet100_hf/validation \
  --epochs 5 --output-dir imagenet100_mobilenetv2_out

# 匯出 f32 ONNX
./gpt2/bin/python export_imagenet100_mobilenetv2_onnx.py \
  --ckpt imagenet100_mobilenetv2_out/best.pth \
  --class-map imagenet100_mobilenetv2_out/class_map.json \
  --onnx-out model/imagenet100_mobilenetv2.onnx
```

> int8 QDQ ONNX 由第 6 步的 wrapper 自動產生(需要 `val_224_txt/` 校準;若沒有,先用
> `preprocess_imagenet100_tensor.py` 產生前處理 tensor,或走 nqdq 路徑跳過 int8)。
> 本 ALPS 快測走 **nqdq** 路徑,不需要 int8。

## 6. Build MobileNetV2 posit `.so` + ALPS(小參數)

```bash
cd "$WS/ImageNet100"
env ONNX_MLIR_POSIT_CONST_ALPS=1 POSIT_CONST_ALPS_JOBS="$(nproc)" \
  ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.01 ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=10 \
  ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=60 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 \
  ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.99 ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001 \
  ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=1024 \
  POSIT_GP_EXPERIMENTAL_FORMATS=p8e2 \
  POSIT_GP_RS_VALUES_P8=7,5,3 POSIT_GP_SC_VALUES_P8=3,0,-3 \
  POSIT_FORMATS=p8e2 INCLUDE_F32_BASELINES=1 \
  bash build_imagenet100_mobilenetv2_11_sos.sh build_mbv2_alps_quick \
    --posit-source nqdq --runtime-format-scope single --runtime-qalign-mode alps-only \
    --runtime-mixed-accum off --runtime-output-alps offline
```
產出 `build_mbv2_alps_quick/imagenet100_mobilenetv2-nqdq-p8e2.so`(權重 ALPS 已 baked)。

## 7. 評估 + offline 輸出/激活 ALPS(500 張、收集 50 張)

```bash
cd "$WS/ImageNet100"
bash "$WS/onnx-mlir/src/bash/time_model11_dataset_parallel.sh" \
  --model-name imagenet100_mobilenetv2 \
  --out-dir "$WS/ImageNet100/build_mbv2_alps_quick" \
  --image-dir "$WS/ImageNet100/imagenet100_hf/validation" \
  --image-preprocess-script "$WS/ImageNet100/preprocess_imagenet100_tensor.py" \
  --shape 1x3x224x224 --suffixes nqdq-p8e2 \
  --baseline none --qalign-auto off --qalign-mode off \
  --jobs "$(nproc)" --limit 500 --warmup 0 --iters 1 --no-benchmark --quire off \
  --output-alps-auto on --output-alps-formats p8e2 \
  --output-alps-collect-limit 50 --output-alps-collect-jobs "$(nproc)" \
  --record-preds on --progress 100 --task-progress on
```

## 8. 結果在哪

- **Top1 / 逐檔預測**:`build_mbv2_alps_quick/` 下的 `*.format_summary.*.log` / `*.predictions.*.log`
- **收集到的激活樣本 + 校準出的 output-ALPS 參數**:
  `build_mbv2_alps_quick/runtime_output_alps_auto/imagenet100_mobilenetv2/nqdq-p8e2/`
  裡的 `collect_upto_50.csv` 與 `params_upto_50.csv`(← **這就是帶 ALPS 參數的結果**)
- build 設定(含當時 ALPS 參數):`build_mbv2_alps_quick/build_posit_config.log`

---

## 只想「純推論(不收集 offline)」看 build-time 權重 ALPS 的話
把第 7 步的 `--output-alps-auto on ...` 那三行改成 `--output-alps-auto off`,就只跑權重 ALPS
的 p8e2,不做 offline 收集(更快)。

## 想加大(正式跑)
- `--limit 5000`(全驗證集)、`--output-alps-collect-limit 500`;
- θ 範圍改用 `suggest_alps_theta_range.py model/imagenet100_mobilenetv2.onnx` 建議值;
- GP 用完整 `RS=7,6,5,4,3 SC=3,2,1,0,-1,-2,-3`、`STEPS=150`、`MAX_SAMPLES=0`。
