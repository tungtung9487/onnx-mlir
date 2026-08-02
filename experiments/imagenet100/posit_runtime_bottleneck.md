# MobileNetV2 posit 推論為何仍慢——實測瓶頸分析(論文用)

## 1. 實測(單張、單執行緒、POSIT_PROFILE 計時器)

在 `conv2d_nchw_kernel` 與 `gemm_kernel` 插入 wall-time 計時器,跑 MobileNetV2 p8e2:

| 階段 | 時間 | 佔比 |
|---|---|---|
| conv 權重/輸入解碼(含 ALPS sinh) | 0.017 s | ~0.07% |
| conv MAC(depthwise 卷積的 fma) | 0.36 s | ~1.5% |
| **gemm(1×1 pointwise 卷積 + FC)** | **22.3 s** | **~98%** |
| (其餘:啟動/載入) | ~2 s | ~8% |
| 單張總計 | ~24.6 s | |

`strace -c` 顯示 syscall 總時間僅 0.5 s → 這 22 s 是**純 CPU 運算**,不是 I/O。

## 2. 根因:MobileNetV2 的算力集中在被降階成 matmul 的 1×1 pointwise 卷積

印出每個 gemm 呼叫的維度,發現一次推論有 **35 個 gemm 呼叫**(不是 1 個):MobileNetV2 是
**depthwise-separable** 架構,其 3×3 **depthwise** 卷積走 `conv2d_nchw_kernel`(通道各自獨立、
MAC 很少 → 0.36 s),但**佔絕大多數乘加的 1×1 pointwise 卷積被編譯器降階成 (im2col) matmul**,
走 `gemm_kernel`。例:

```
GEMM#0  a=[16,32]  × b=[1,32,12544]  → 16×32×12544 = 6.4M MAC   (112×112 pointwise)
GEMM#33 a=[1280,320]×b=[1,320,49]    → 1280×320×49 = 20M MAC
GEMM#34 a=[1,1280] × b=[100,1280]    → FC 分類器(小)
```

全部 pointwise gemm 合計約 3 億次 MAC——這才是 MobileNetV2 的主要算力,而它全部落在 posit 的
`gemm_kernel`。

## 3. 為何 gemm 路徑特別慢(每個 MAC 的固定成本)

`gemm_kernel` 的 metadata/ALPS(metaF32)內層迴圈,對**每一個乘加**都做:

1. **兩次帶 mutex 的 per-channel metadata 查詢**(`lookupTensorGPMetadataForChannel`)——
   GEMM#0 就是 6.4M 次上鎖;
2. **重複解碼**:權重 `a[m,k]` 不隨輸出空間位置 `s` 變,卻被重解 `S` 次(GEMM#0 的 S=12544);
   輸入 `b[n,k,s]` 不隨輸出通道 `m` 變,卻被重解 `M` 次;
3. 每次解碼還要做 ALPS 的 `sinh` 解壓(transcendental)。

換言之,3 億次 MAC × (每次 ~數十 ns 的鎖+重解+sinh) ≈ 22 s。這與 `conv2d_nchw_kernel`
在**優化前**的問題完全相同——差別是先前的預解碼/LUT 優化只加進了 `conv2d`(depthwise),
**pointwise 的 gemm 路徑尚未套用**。

## 4. 論文可用結論

> 於本 posit 軟體實作中,MobileNetV2 單張推論時間幾乎完全由 1×1 pointwise 卷積主導
> (實測佔 98%,約 22 s),而 3×3 depthwise 卷積僅佔約 1.5%。原因在於 MobileNetV2 為
> depthwise-separable 架構,其乘加運算集中於 pointwise 層,而編譯器將 pointwise 卷積降階為
> (im2col) 矩陣乘,於 posit runtime 走 `gemm_kernel`。該路徑對每一次乘加皆需 (i) 兩次帶鎖的
> per-channel metadata 查詢、(ii) 對不隨迴圈變動之運算元重複解碼、(iii) ALPS 之 asinh/sinh
> companding,累積約 3 億次而成為瓶頸。相對地,depthwise 卷積因每通道乘加極少而可忽略。
> 這說明軟體 posit 推論的成本不由參數量、而由**乘加次數與其上的每元素解碼/companding 開銷**
> 決定;要顯著加速,應對 pointwise 的矩陣乘路徑套用與卷積相同的「每元素只解碼一次」預解碼與
> 查表,或改以整層向量化/多執行緒執行。

## 5. 加速實作與結果(已完成、已驗證)

把已在 `conv2d_nchw_kernel` 驗證過的做法搬到 `gemm_kernel` 的 im2col metadata 路徑:
- 將 `aMetaCh`/`bMetaCh` 的 per-channel metadata 查詢(帶 mutex)**移出**內層 `k`/`s` 迴圈
  (a 只隨 m、b 只隨 n);
- **預解碼** `a`(M×K)與 `b`(N×K×S)各一次,內層迴圈只剩純 `fma`;
- 對輸出維掛 intra-op OpenMP(`POSIT_OMP_THREADS`)。

**注意路徑**:MobileNetV2 的 GP+ALPS build 走的是 **`gp_metadata_f64`** 分支(以
`POSIT_PROFILE` 印 `modeForProbe` 確認),而非 `gp_metadata_f32`;兩個分支都已套用預解碼。

**實測(單張、單執行緒、GP+ALPS offline、p8e2)**:

| | 優化前 | 優化後 |
|---|---|---|
| gemm(pointwise 卷積) | 6.5 s | **0.74 s(~9×)** |
| conv_decode + conv_mac | 0.05 s | 0.05 s |
| 單張總 wall | ~9 s | **3.8 s** |
| Top1(limit-100) | 71% | **71%(bit-identical)** |
| 單張 top5(某圖) | [22,42,90,10,62] | [22,42,90,10,62](完全相同) |

計算本身(conv+gemm)由 ~6.6 s 降到 ~0.8 s;剩下約 3 s 為 process 啟動/載入 `.so`。
消除的主要成本是每個乘加一次的 mutex 鎖定 metadata 查詢與重複解碼。要進一步降低單張延遲,
可設 `POSIT_OMP_THREADS>1`(對輸出通道平行),或減少 per-process 的 `.so` 載入開銷。
