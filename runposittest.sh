SOFTPOSIT_DIR=/home/lai/mlir_toy/SoftPosit/SoftPosit
SOFTPOSIT_BUILD=/home/lai/mlir_toy/SoftPosit/SoftPosit/build/Linux_x86_64_GCC
SOFTPOSIT_INC=$SOFTPOSIT_DIR/source/include

LLVM_SRC=/home/lai/mlir_toy/llvm-project
LLVM_BUILD=/home/lai/mlir_toy/llvm-project/build


../../../../llvm-project/build/bin/mlir-translate --mlir-to-llvmir ../../../positllvm2.mlir  -o test2.ll

clang++ -O2 -fPIC -shared \
  test.ll posit_runtime.cpp \
  -I$SOFTPOSIT_INC \
  -I/home/lai/mlir_toy/llvm-project/mlir/include \
  -I$LLVM_BUILD/include \
  -L$SOFTPOSIT_BUILD -lsoftposit \
  -Wl,-rpath,$SOFTPOSIT_BUILD \
  -o libtest_posit.so
  
// run ll .so檔
clang++ -O2 run.fixed.cpp \
  -I$SOFTPOSIT_INC \
  -I$LLVM_SRC/mlir/include \
  -I$LLVM_BUILD/tools/mlir/include \
  -I$LLVM_BUILD/include \
  -L$SOFTPOSIT_BUILD -lsoftposit \
  -Wl,-rpath,$SOFTPOSIT_BUILD \
  -ldl \
  -o run
// clang++ -O2 run.cpp -ldl -o run
./run

// run model
./run ./mnist-12-posit.so(so) mnist.txt(測資) --warmup 0 --iters 200

// 多個比較
hyperfine --warmup 5 --runs 30 "指令一" "指令二"
// 多比較 自行寫的sh
./scripts/build_softposit_px1_lib.sh
./scripts/build_px1_targets.sh mnist-12-qdq-p8e0.ll(檔名要換)



//rebuild onnx-mlir-opt
cmake -G Ninja .. -DMLIR_DIR=/home/lai/mlir_toy/llvm-project/build/lib/cmake/mlir -DLLVM_DIR=/home/lai/mlir_toy/llvm-project/build/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release

ninja onnx-mlir-opt

cmake --build /home/lai/onnx_mlir/onnx-mlir/build --target onnx-mlir-opt -- -j4
cmake --build /home/lai/onnx_mlir/onnx-mlir/build --target onnx-mlir-opt onnx-mlir -- -j4

//build model to onnx
./onnx-mlir --EmitONNXIR mnist-12.onnx(檔名) 
不省略：
./onnx-mlir --EmitONNXIR --mlir-elide-resource-strings-if-larger=1000000000 --mlir-elide-elementsattrs-if-larger=1000000000 

//每一階段單獨看
./onnx-mlir-opt --convert-onnx-to-posit ../../../testonnxtoposit2.mlir -o -
./onnx-mlir-opt --convert-posit-to-krnl ../../../posittokrnl4.mlir -o -
./onnx-mlir-opt --convert-krnl-to-llvm --canonicalize --cse  ../../../testkrnl4.mlir -o ../../../test3.ll

//一次生成11.so
bash scripts/build_mobilenet11_sos.sh 可加生成檔案位置
./bash/build_mobilenet11_sos.sh ./temp/mobilenet11_temp


bash/time_resnet50_11_dataset_parallel.sh \
  --out-dir ./temp/resnet50-11_temp \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --jobs 4 \
  --limit 1 \
  --warmup 0 \
  --iters 1 \
  --no-benchmark \
  --progress 1

最新版build 依照這板  可最低
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0078125 ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=16 ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=25 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.95 POSIT_GP_RS_VALUES_P8=7,6,5 POSIT_GP_SC_VALUES_P8=0,-3
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_mobilenet11_sos.sh \
  /tmp/mobilenet11_singlefmt \
  --posit-source nqdq \
  --posit-formats p8e0 \
  --runtime-format-scope single \
  --runtime-qalign-mode alps-only \
  --runtime-mixed-accum off 

RESNET50
ONNX_MLIR_POSIT_CONST_ALPS=1 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0078125 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=16 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=25 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.95 \
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1,p8e2 \
POSIT_GP_RS_VALUES_P8=7,6,5 \
POSIT_GP_SC_VALUES_P8=0,-3 \
POSIT_RUNTIME_CPP_PATH=/home/lai/onnx_mlir/onnx-mlir/src/posit_runtime.cpp \
RUN_TIME_CPP_PATH=/home/lai/onnx_mlir/onnx-mlir/src/run_time.cpp \
bash /home/lai/onnx_mlir/ImageNet100/build_imagenet100_11_sos.sh \
  /home/lai/onnx_mlir/ImageNet100/build_posit11_nqdq_gp_alps \
  --posit-source nqdq \
  --posit-formats p8e0,p8e1,p8e2,p16e0,p16e1,p16e2,p32e0,p32e1,p32e2 \
  --runtime-format-scope single \
  --runtime-qalign-mode alps-only \
  --runtime-mixed-accum off

10跟100的資料夾： 
10：
--txt-dir ./temp/imagenette_val_224
--label-map ./temp/imagenette_val_224_labels.txt 
100：
--txt-dir /home/lai/onnx_mlir/ImageNet100/val_224_txt
--label-map /home/lai/onnx_mlir/ImageNet100/val_224_labels_imagenet1k.txt

mobilenetv2最終版build (nqdq)  要記得改輸出資料夾 runtime也要用同樣的  正常格式p8/p16/p32 （es=0,1,2）用法
POSIT_CONST_ALPS_JOBS=25 \
POSIT_CONST_DEBUG=1 \
ONNX_MLIR_POSIT_CONST_ALPS=1 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0078125 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=16 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=25 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.95 \
ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001 \
ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=0 \
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1,p8e2 \
POSIT_GP_RS_VALUES_P8=7,6,5 \
POSIT_GP_SC_VALUES_P8=3,-3 \
POSIT_FORMATS=p8e0,p8e1,p8e2 \
bash /home/lai/onnx_mlir/ImageNet100/build_imagenet100_mobilenetv2_11_sos.sh \
  /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps \
  --posit-source nqdq \
  --runtime-format-scope single \
  --runtime-qalign-mode alps-only \
  --runtime-mixed-accum off \
  --runtime-output-alps offline

0525:
POSIT_FORMATS=p8e0,p8e1,p8e2 \
INCLUDE_F32_BASELINES=1 \
POSIT_CONST_ALPS_JOBS=25 \
POSIT_CONST_DEBUG=1 \
ONNX_MLIR_POSIT_CONST_ALPS=1 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0078125 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=2048 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=300 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.95 \
ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001 \
ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=0 \
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1,p8e2 \
POSIT_GP_RS_VALUES_P8=7,6,5,4 \
POSIT_GP_SC_VALUES_P8=5,3,1,0,-1,-3,-5 \
bash /home/lai/onnx_mlir/ImageNet100/build_imagenet100_mobilenetv2_11_sos.sh \
  /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps_v2 \
  --posit-source nqdq \
  --runtime-format-scope single \
  --runtime-qalign-mode alps-only \
  --runtime-mixed-accum off \
  --runtime-output-alps offline



mobilenetv2最終版runtime  (nqdq)  要記得改要測的資料夾

env \
  -u POSIT_MIXED_PRECISION_P16 \
  -u POSIT_MIXED_PRECISION_P16_OPS \
  -u POSIT_MIXED_PRECISION_P32 \
  -u POSIT_MIXED_PRECISION_P32_OPS \
  -u POSIT_QOP_F32_MATH \
  -u POSIT_QOP_F32_MATH_OPS \
  -u POSIT_DOT_F32_MATH \
  -u POSIT_DOT_F32_MATH_OPS \
  -u POSIT_DOT_MIXED_P16 \
  -u POSIT_DOT_MIXED_P16_OPS \
  -u POSIT_QUIRE_P8 \
  

env -u POSIT_MIXED_PRECISION_P16     -u POSIT_MIXED_PRECISION_P16_OPS     -u POSIT_MIXED_PRECISION_P32     -u POSIT_MIXED_PRECISION_P32_OPS     -u POSIT_QOP_F32_MATH     -u POSIT_QOP_F32_MATH_OPS     -u POSIT_DOT_F32_MATH     -u POSIT_DOT_F32_MATH_OPS     -u POSIT_DOT_MIXED_P16     -u POSIT_DOT_MIXED_P16_OPS     -u POSIT_QUIRE_P8  POSIT_RUNTIME_OUTPUT_ALPS_DEBUG=1 POSIT_RUNTIME_OUTPUT_ALPS_DEBUG_LIMIT=64   bash ./bash/time_mobilenet11_dataset_parallel.sh   --out-dir ./temp/mobilenet11_alps_strict   --txt-dir /home/lai/onnx_mlir/ImageNet100/val_224_txt   --label-map /home/lai/onnx_mlir/ImageNet100/val_224_labels_imagenet1k.txt   --suffixes qdq-f32,nqdq-f32,nqdq-p8e0,nqdq-p8e1,nqdq-p8e2   --qalign-auto off   --qalign-mode off   --jobs 25   --limit 5000   --warmup 0   --iters 1   --no-benchmark   --quire off   --progress 1   --task-progress on 2>&1 | tee ./temp/mobilenet11_alps_offline_0511/mobilenet11_runtime_output_alps_debug.log 
// test big range
POSIT_CONST_ALPS_JOBS=25 POSIT_CONST_DEBUG=1 ONNX_MLIR_POSIT_CONST_ALPS=1 ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0009765625 ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=1024 ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=769 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.95 ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.005 ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=0 POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1,p8e2 POSIT_GP_RS_VALUES_P8=7,6,5,4,3 POSIT_GP_SC_VALUES_P8=5,-5 bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_mobilenet11_sos.sh   ./temp/mobilenet11_alps_offline_ir   --posit-source nqdq   --posit-formats p8e0,p8e1,p8e2   --runtime-format-scope single   --runtime-qalign-mode alps-only   --runtime-mixed-accum off   --runtime-output-alps offline

0525：
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_model11_dataset_parallel.sh \
  --model-name imagenet100_mobilenetv2 \
  --out-dir /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps_v2 \
  --image-dir /home/lai/onnx_mlir/ImageNet100/imagenet100_hf/validation \
  --image-preprocess-script /home/lai/onnx_mlir/ImageNet100/preprocess_imagenet100_tensor.py \
  --shape 1x3x224x224 \
  --suffixes nqdq-f32,qdq-f32,nqdq-p8e0,nqdq-p8e1,nqdq-p8e2 \
  --baseline none \
  --qalign-auto off \
  --qalign-mode off \
  --jobs 25 \
  --limit 5000 \
  --warmup 0 \
  --iters 1 \
  --no-benchmark \
  --quire off \
  --task-progress on \
  --progress 10 \
  --output-alps-auto on     \
  --output-alps-formats p8e2     \
  --output-alps-collect-limit 500     \
  --output-alps-collect-jobs 25     \
  --record-preds on  \
  2>&1 | tee /home/lai/onnx_mlir/ImageNet100/build_posit11_resnet18_alps_check_0521/resnet18_nqdq_p8e2_output_alps_5000.log

mobilenetv2 extra (qdq) (p4e2~p8e2) build 

方法A ： INT8 → posit bits 存儲
POSIT_FORMATS=p4e0,p4e1,p4e2,p5e0,p5e1,p5e2,p6e0,p6e1,p6e2,p7e0,p7e1,p7e2,p8e0,p8e1,p8e2 \
INCLUDE_F32_BASELINES=1 \
POSIT_STORE_DQ_AS_POSIT=1 \
bash /home/lai/onnx_mlir/ImageNet100/build_imagenet100_mobilenetv2_11_sos.sh \
  /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_qdq_p4top8_storedq \
  --posit-source qdq \
  --runtime-format-scope single \
  --runtime-qalign-mode full \
  --runtime-mixed-accum off \
  --runtime-output-alps off

方法B ： 直接 posit，不經 INT8
POSIT_FORMATS=p4e0,p4e1,p4e2,p5e0,p5e1,p5e2,p6e0,p6e1,p6e2,p7e0,p7e1,p7e2,p8e0,p8e1,p8e2 \
INCLUDE_F32_BASELINES=1 \
POSIT_PREFER_DIRECT_FROM_QDQ=1 \
bash /home/lai/onnx_mlir/ImageNet100/build_imagenet100_mobilenetv2_11_sos.sh \
  /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_qdq_p4top8_directposit \
  --posit-source qdq \
  --runtime-format-scope single \
  --runtime-qalign-mode full \
  --runtime-mixed-accum off \
  --runtime-output-alps off

解釋：
POSIT_FORMATS=p4e0,p4e1,p4e2,p5e0,p5e1,p5e2,p6e0,p6e1,p6e2,p7e0,p7e1,p7e2,p8e0,p8e1,p8e2 \
INCLUDE_F32_BASELINES=1 \
POSIT_PREFER_DIRECT_FROM_QDQ=1 \  控制直接f32 to posit
POSIT_STORE_DQ_AS_POSIT=1 \  控制int8轉posit真的有拿來儲存 而非直接轉回f32存
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_mobilenet_extra_sos.sh \
  ./temp/mobilenet_extra_qdq_0512 \
  --posit-source qdq \ 從 QDQ 模型出發
  --align-to-int8-qdomain \ 盡量保留對 int8 DQ 域的對齊 
  --strict-qdq-mode \  優先走 strict QDQ lowering
  --runtime-format-scope single \
  --runtime-qalign-mode alps-only \
  --runtime-mixed-accum off \  不要混進 p16/p32 promoted mixed path
  --runtime-output-alps off   不要混進 output ALPS

mobilenetv2 extra (qdq) (p4e2~p8e2) runtime 

POSIT_QOP_F32_MATH=on \
POSIT_QALIGN_ALPS_TARGET=int8_dq \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_model11_dataset_parallel.sh \
  --model-name imagenet100_mobilenetv2 \
  --out-dir /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_qdq_p4top8_storedq \
  --image-dir /home/lai/onnx_mlir/ImageNet100/imagenet100_hf/validation \
  --image-preprocess-script /home/lai/onnx_mlir/ImageNet100/preprocess_imagenet100_tensor.py \
  --shape 1x3x224x224 \
  --suffixes qdq-f32,qdq-p4e0,qdq-p4e1,qdq-p4e2,qdq-p5e0,qdq-p5e1,qdq-p5e2,qdq-p6e0,qdq-p6e1,qdq-p6e2,qdq-p7e0,qdq-p7e1,qdq-p7e2,qdq-p8e0,qdq-p8e1,qdq-p8e2 \
  --baseline none \
  --qalign-auto off --qalign-mode off \
  --jobs 25 --limit 5000 \
  --warmup 0 --iters 1 --no-benchmark --quire off \
  --progress 100 --task-progress on \
  --output-alps-auto off \
  --record-preds on

記得換資料夾！！！
  
// 
  POSIT_QOP_F32_MATH=on \
  POSIT_QOP_F32_MATH_OPS=conv2d,gemm 這兩行決定了運算用f32 重要

resnet18 build 0518
POSIT_CONST_ALPS_JOBS=25 \
POSIT_CONST_DEBUG=1 \
ONNX_MLIR_POSIT_CONST_ALPS=1 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0078125 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=16 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=25 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.95 \
ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001 \
ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=0 \
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1,p8e2 \
POSIT_GP_RS_VALUES_P8=7,6,5 \
POSIT_GP_SC_VALUES_P8=5,3,1,0,-1,-3 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_model11_sos.sh \
  --model-name imagenet100_resnet18 \
  --qdq-onnx /home/lai/onnx_mlir/ImageNet100/model/imagenet100_resnet18-int8-qdq.onnx \
  --nqdq-onnx /home/lai/onnx_mlir/ImageNet100/model/imagenet100_resnet18.onnx \
  --out-dir /home/lai/onnx_mlir/ImageNet100/build_posit11_resnet18_alps \
  --posit-source nqdq \
  --posit-formats p8e0,p8e1,p8e2 \
  --runtime-format-scope single \
  --runtime-qalign-mode alps-only \
  --runtime-mixed-accum off \
  --runtime-output-alps offline

0525：
POSIT_CONST_ALPS_JOBS=25 POSIT_CONST_DEBUG=1 ONNX_MLIR_POSIT_CONST_ALPS=1 ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0078125 ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=16 ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=25 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.95 ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001 ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=0 POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1,p8e2 POSIT_GP_RS_VALUES_P8=7,6,5 POSIT_GP_SC_VALUES_P8=3,-3 POSIT_FORMATS=p8e0,p8e1,p8e2 bash /home/lai/onnx_mlir/ImageNet100/build_imagenet100_resnet18_11_sos.sh   /home/lai/onnx_mlir/ImageNet100/build_posit11_resnet18_alps_check_0521   --posit-source nqdq   --runtime-format-scope single   --runtime-qalign-mode alps-only   --runtime-mixed-accum off   --runtime-output-alps offline



resnet18 runtime 0518

POSIT_QOP_F32_MATH=on \
POSIT_QOP_F32_MATH_OPS=conv2d,gemm 
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_model11_dataset_parallel.sh \
  --model-name imagenet100_resnet18 \
  --out-dir /home/lai/onnx_mlir/ImageNet100/build_posit11_resnet18_alps \
  --txt-dir /home/lai/onnx_mlir/ImageNet100/val_224_txt \
  --label-map /home/lai/onnx_mlir/ImageNet100/val_224_labels_imagenet1k.txt \
  --shape 1x3x224x224 \
  --suffixes qdq-f32,nqdq-f32,nqdq-p8e0,nqdq-p8e1,nqdq-p8e2 \
  --baseline none \
  --qalign-auto off \
  --qalign-mode off \
  --jobs 25 \
  --limit 5000 \
  --warmup 0 \
  --iters 1 \
  --no-benchmark \
  --quire off \
  --progress 1 \
  --task-progress on \
  --output-alps-auto on \
  --output-alps-formats p8e0,p8e1,p8e2 \
  --output-alps-collect-limit 100 \
  --output-alps-collect-jobs 25

0525：
env   -u POSIT_MIXED_PRECISION_P16   -u POSIT_MIXED_PRECISION_P16_OPS   -u POSIT_MIXED_PRECISION_P32   -u POSIT_MIXED_PRECISION_P32_OPS   -u POSIT_QOP_F32_MATH   -u POSIT_QOP_F32_MATH_OPS   -u POSIT_DOT_F32_MATH   -u POSIT_DOT_F32_MATH_OPS   -u POSIT_DOT_MIXED_P16   -u POSIT_DOT_MIXED_P16_OPS   -u POSIT_QUIRE_P8   bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_model11_dataset_parallel.sh     --model-name imagenet100_resnet18     --out-dir /home/lai/onnx_mlir/ImageNet100/build_posit11_resnet18_alps_check_0521     --image-dir /home/lai/onnx_mlir/ImageNet100/imagenet100_hf/validation     --image-preprocess-script /home/lai/onnx_mlir/ImageNet100/preprocess_imagenet100_tensor.py     --shape 1x3x224x224     --suffixes qdq-f32,nqdq-f32,nqdq-p8e2     --baseline none     --qalign-auto off     --qalign-mode off     --jobs 25     --limit 5000     --warmup 0     --iters 1     --no-benchmark     --quire off     --progress 10     --task-progress on     --output-alps-auto on     --output-alps-formats p8e2     --output-alps-collect-limit 500     --output-alps-collect-jobs 25     --record-preds on   2>&1 | tee /home/lai/onnx_mlir/ImageNet100/build_posit11_resnet18_alps_check_0521/resnet18_nqdq_p8e2_output_alps_5000.log

env   -u POSIT_MIXED_PRECISION_P16   
      -u POSIT_MIXED_PRECISION_P16_OPS   
      -u POSIT_MIXED_PRECISION_P32   
      -u POSIT_MIXED_PRECISION_P32_OPS   
      -u POSIT_QOP_F32_MATH   
      -u POSIT_QOP_F32_MATH_OPS   
      -u POSIT_DOT_F32_MATH   
      -u POSIT_DOT_F32_MATH_OPS   
      -u POSIT_DOT_MIXED_P16   
      -u POSIT_DOT_MIXED_P16_OPS   
      -u POSIT_quire_P8   
      bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_model11_dataset_parallel.sh     
      --model-name imagenet100_resnet18     \
      --out-dir /home/lai/onnx_mlir/ImageNet100/build_posit11_resnet18_alps_check_0521     \
      --image-dir /home/lai/onnx_mlir/ImageNet100/imagenet100_hf/validation     \
      --image-preprocess-script /home/lai/onnx_mlir/ImageNet100/preprocess_imagenet100_tensor.py     \
      --shape 1x3x224x224     \
      --suffixes qdq-f32,nqdq-f32,nqdq-p8e2     \
      --baseline none     \
      --qalign-auto off     \
      --qalign-mode off     \
      --jobs 25     \
      --limit 5000     \
      --warmup 0     \
      --iters 1     \
      --no-benchmark     \
      --quire off     \
      --progress 10     \
      --task-progress on     \
      --output-alps-auto on     \
      --output-alps-formats p8e2     \
      --output-alps-collect-limit 500     \
      --output-alps-collect-jobs 25     \
      --record-preds on   2>&1 | tee /home/lai/onnx_mlir/ImageNet100/build_posit11_resnet18_alps_check_0521/resnet18_nqdq_p8e2_output_alps_5000.log



resnet18 extra (qdq) (p4e0~p7e3) build 0603

POSIT_FORMATS=p4e0,p4e1,p4e2,p5e0,p5e1,p5e2,p6e0,p6e1,p6e2,p7e0,p7e1,p7e2,p8e0,p8e1,p8e2 \
INCLUDE_F32_BASELINES=1 \
bash /home/lai/onnx_mlir/ImageNet100/build_imagenet100_resnet18_11_sos.sh \
  /home/lai/onnx_mlir/ImageNet100/build_posit11_resnet18_qdq_p4top8 \
  --posit-source qdq \
  --runtime-format-scope single \
  --runtime-qalign-mode full \
  --runtime-mixed-accum off \
  --runtime-output-alps off

resnet18 extra (qdq) (p4e0~p7e3) runtime 0603

POSIT_QOP_F32_MATH=on \
POSIT_QALIGN_ALPS_TARGET=int8_dq \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_model11_dataset_parallel.sh \
  --model-name imagenet100_resnet18 \
  --out-dir /home/lai/onnx_mlir/ImageNet100/build_posit11_resnet18_qdq_p4top8 \
  --image-dir /home/lai/onnx_mlir/ImageNet100/imagenet100_hf/validation \
  --image-preprocess-script /home/lai/onnx_mlir/ImageNet100/preprocess_imagenet100_tensor.py \
  --shape 1x3x224x224 \
  --suffixes qdq-f32,qdq-p4e0,qdq-p4e1,qdq-p4e2,qdq-p5e0,qdq-p5e1,qdq-p5e2,qdq-p6e0,qdq-p6e1,qdq-p6e2,qdq-p7e0,qdq-p7e1,qdq-p7e2,qdq-p8e0,qdq-p8e1,qdq-p8e2 \
  --baseline none \
  --qalign-auto off \
  --qalign-mode off \
  --jobs 25 \
  --limit 5000 \
  --warmup 0 \
  --iters 1 \
  --no-benchmark \
  --quire off \
  --progress 100 \
  --task-progress on \
  --output-alps-auto off \
  --record-preds on


GPT-2

build
POSIT_FORMATS=p8e1,p16e1,p32e1 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_gpt2_hf_11_sos.sh \
  build_gpt2_nqdq_p8e1 --posit-source nqdq --posit-formats p8e0,p8e1,p8e2 --runtime-format-scope single \
  --runtime-qalign-mode full --runtime-mixed-accum off --runtime-output-alps off

runtime

D=build_gpt2_nqdq_p8e1; TXT="--text-file eval_text/wikitext2_test.txt --max-tokens 512 --progress 128"

# ① f32 (ORT, 全精度參考, 快)
gpt2/bin/python run_gpt2_text_eval.py --model model/gpt2_onnx_community/onnx/model.onnx --mode score --text-file eval_text/wikitext2_test.txt --max-tokens 512 --progress 128

# ② int8 (ORT, 快)
gpt2/bin/python run_gpt2_text_eval.py --model model/gpt2_onnx_community/onnx/model_int8.onnx --mode score  --text-file eval_text/wikitext2_test.txt --max-tokens 512 --progress 128

# ③ posit p8e1
POSIT_OMP_THREADS=24 gpt2/bin/python run_gpt2_text_eval.py --model build_gpt2_nqdq_p8e1/gpt2-hf-debug-nqdq-p8e1.so --mode score --text-file eval_text/wikitext2_test.txt --max-tokens 512 --progress 128

# ④ posit p16e1
POSIT_OMP_THREADS=24 gpt2/bin/python run_gpt2_text_eval.py --model build_gpt2_nqdq_p16e1/gpt2-hf-debug-nqdq-p16e1.so --mode score --text-file eval_text/wikitext2_test.txt --max-tokens 512 --progress 128

# ⑤ posit p32e1
POSIT_OMP_THREADS=24 gpt2/bin/python run_gpt2_text_eval.py --model build_gpt2_nqdq_p32e1/gpt2-hf-debug-nqdq-p32e1.so --mode score --text-file eval_text/wikitext2_test.txt --max-tokens 512 --progress 128

7/8 mobilenetv2 build 路徑要記得改
POSIT_FORMATS=p8e0,p8e1,p8e2
INCLUDE_F32_BASELINES=1
ONNX_MLIR_POSIT_CONST_ALPS=1
POSIT_CONST_ALPS_JOBS=25
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0001
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=5
ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=150
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.99
ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001
ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=0
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1,p8e2
POSIT_GP_RS_VALUES_P8=7,6,5,4,3
POSIT_GP_SC_VALUES_P8=3,2,1,0,-1,-2,-3
bash /home/lai/onnx_mlir/ImageNet100/build_imagenet100_mobilenetv2_11_sos.sh
/home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps_sqnr_p8_v2
--posit-source nqdq
--posit-formats p8e0,p8e1,p8e2
--runtime-format-scope single
--runtime-qalign-mode alps-only
--runtime-mixed-accum off
--runtime-output-alps offline
要的話複製這個 上面沒有反斜線：POSIT_FORMATS=p8e0,p8e1,p8e2 INCLUDE_F32_BASELINES=1 ONNX_MLIR_POSIT_CONST_ALPS=1 POSIT_CONST_ALPS_JOBS=25 ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0001 ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=5 ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=150 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.99 ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001 ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=0 POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1,p8e2 POSIT_GP_RS_VALUES_P8=7,6,5,4,3 POSIT_GP_SC_VALUES_P8=3,2,1,0,-1,-2,-3 bash /home/lai/onnx_mlir/ImageNet100/build_imagenet100_mobilenetv2_11_sos.sh   /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps_sqnr_p8_v2   --posit-source nqdq --posit-formats p8e0,p8e1,p8e2 --runtime-format-scope single   --runtime-qalign-mode alps-only --runtime-mixed-accum off --runtime-output-alps offline

7/8 mobilenetv2 run  路徑要記得改

POSIT_QOP_F32_MATH=on \
POSIT_QOP_F32_MATH_OPS=conv2d,gemm 
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_model11_dataset_parallel.sh
  --model-name imagenet100_mobilenetv2
  --out-dir /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps_sqnr_p8_v2
  --image-dir /home/lai/onnx_mlir/ImageNet100/imagenet100_hf/validation
  --image-preprocess-script /home/lai/onnx_mlir/ImageNet100/preprocess_imagenet100_tensor.py
  --shape 1x3x224x224
  --suffixes nqdq-p8e0,nqdq-p8e2
  --baseline none
  --qalign-auto off
  --qalign-mode off
  --jobs 25
  --limit 5000
  --warmup 0
  --iters 1
  --no-benchmark
  --quire off
  --output-alps-auto on
  --output-alps-formats p8e0,p8e2
  --output-alps-collect-limit 500
  --output-alps-collect-jobs 25
  --record-preds on
  --progress 100
  --task-progress on
  2>&1 | tee /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps_sqnr_p8_v2/mobilenetv2_alps_sqnr_p8e2e0_offline_5000.log

要的話複製這個 上面沒有反斜線：POSIT_QOP_F32_MATH=on POSIT_QOP_F32_MATH_OPS=conv2d,gemm  bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_model11_dataset_parallel.sh   --model-name imagenet100_mobilenetv2   --out-dir /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps_sqnr_p8_v2   --image-dir /home/lai/onnx_mlir/ImageNet100/imagenet100_hf/validation   --image-preprocess-script /home/lai/onnx_mlir/ImageNet100/preprocess_imagenet100_tensor.py   --shape 1x3x224x224   --suffixes nqdq-p8e0,nqdq-p8e2   --baseline none --qalign-auto off --qalign-mode off   --jobs 25 --limit 5000 --warmup 0 --iters 1 --no-benchmark --quire off   --output-alps-auto on   --output-alps-formats p8e0,p8e2   --output-alps-collect-limit 500   --output-alps-collect-jobs 25   --record-preds on --progress 100 --task-progress on   2>&1 | tee /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps_sqnr_p8_v2/mobilenetv2_alps_sqnr_p8e2e0_offline_5000.log


跑之前先跑這兩行，路徑一樣要記得改： 
cmake -G Ninja .. -DMLIR_DIR=/home/lai/mlir_toy/llvm-project/build/lib/cmake/mlir -DLLVM_DIR=/home/lai/mlir_toy/llvm-project/build/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release
cmake --build /home/lai/onnx_mlir/onnx-mlir/build --target onnx-mlir-opt onnx-mlir -- -j4

8/8
改用一鍵編譯後：
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build clang lld python3 python3-venv python3-pip git
git clone -b tungtung9487/posit-work-20260329 https://github.com/tungtung9487/onnx-mlir.git
bash onnx-mlir/experiments/imagenet100/reproduce_alps.sh
unset WS
WS=/home/lai/test_rebuild_project bash /home/lai/onnx_mlir/onnx-mlir/experiments/imagenet100/reproduce_alps.sh

export WS=/home/lai/test_rebuild_project
export om="$WS/onnx-mlir"
export in100="$WS/ImageNet100"
export py="$in100/venv/bin/python"
cd "$in100"
[ -d val_224_txt ] || "$py" gen_val_224_txt.py --data-root imagenet100_hf/validation --out-dir val_224_txt --limit 500

POSIT_FORMATS=p8e0,p8e1,p8e2 INCLUDE_F32_BASELINES=1 \
ONNX_MLIR_POSIT_CONST_ALPS=1 POSIT_CONST_ALPS_JOBS="$(nproc)" \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0001 ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=5 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=150 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.99 ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001 \
ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=0 \
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1,p8e2 \
POSIT_GP_RS_VALUES_P8=7,6,5,4,3 POSIT_GP_SC_VALUES_P8=3,2,1,0,-1,-2,-3 \
ONNX_MLIR_ROOT="$om" \
  bash "$in100/build_imagenet100_mobilenetv2_11_sos.sh" \
    "$in100/build_posit11_mobilenetv2_alps_sqnr_p8_v2" \
    --posit-source nqdq --posit-formats p8e0,p8e1,p8e2 --runtime-format-scope single \
    --runtime-qalign-mode alps-only --runtime-mixed-accum off --runtime-output-alps offline


POSIT_QOP_F32_MATH=on POSIT_QOP_F32_MATH_OPS=conv2d,gemm \
  bash "$om/src/bash/time_model11_dataset_parallel.sh" \
    --model-name imagenet100_mobilenetv2 \
    --out-dir "$in100/build_posit11_mobilenetv2_alps_sqnr_p8_v2" \
    --image-dir "$in100/imagenet100_hf/validation" \
    --image-preprocess-script "$in100/preprocess_imagenet100_tensor.py" \
    --shape 1x3x224x224 --suffixes nqdq-p8e0,nqdq-p8e2 \
    --baseline none --qalign-auto off --qalign-mode off \
    --jobs "$(nproc)" --limit 5000 --warmup 0 --iters 1 --no-benchmark --quire off \
    --output-alps-auto on --output-alps-formats p8e0,p8e2 \
    --output-alps-collect-limit 500 --output-alps-collect-jobs "$(nproc)" \
    --record-preds on --progress 100 --task-progress on \
    2>&1 | tee "$in100/build_posit11_mobilenetv2_alps_sqnr_p8_v2/mobilenetv2_alps_p8e2e0_offline_5000.log"
