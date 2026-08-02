# 為何 MobileNetV2 的 posit8 `.so` 比 int8 大這麼多，ResNet18 卻幾乎相同（論文用）

## 1. 觀察到的現象（實測）

| 模型 | posit8 `.so` | int8（qdq-f32.so / int8 ONNX） | 比值 |
|---|---|---|---|
| **MobileNetV2** | 7.4 MB | 2.76 MB | **2.7×** |
| **ResNet18** | 12.3 MB | 11.4 MB | **1.08×** |

## 2. 段落分解（`size -A`，實測）

| 檔案 | `.text`（程式碼） | `.rodata`（常數/權重/metadata） |
|---|---|---|
| MobileNetV2 posit8 | 1.13 MB | 6.05 MB |
| MobileNetV2 int8   | 0.30 MB | 2.42 MB |
| ResNet18 posit8    | 0.99 MB | 11.19 MB |
| ResNet18 int8      | 0.14 MB | 11.21 MB |

模型權重元素數（實測，BatchNorm 在匯出時已 fold 進 conv，兩者皆 BN=0）：
MobileNetV2 ≈ **2.33 M**、ResNet18 ≈ **11.2 M**。

## 3. 關鍵拆解：`.rodata` 的「每權重位元組數」

用 `.rodata ÷ 權重元素數`：

| | int8 | posit8 |
|---|---|---|
| MobileNetV2 | 2.42MB / 2.33M = **1.04 B/w** | 6.05MB / 2.33M = **2.6 B/w** |
| ResNet18    | 11.21MB / 11.2M = **1.00 B/w** | 11.19MB / 11.2M = **1.00 B/w** |

- int8 兩個模型都是乾淨的 **1 byte/權重**。
- posit8 對 **ResNet18 也是 1 byte/權重**（和 int8 相同）；但對 **MobileNetV2 卻是 2.6 byte/權重**。

換言之，把 `.rodata` 寫成「權重（1 B/權重）＋固定開銷」：

$$\texttt{.rodata} \approx 1\text{B}\times N_{\text{weight}} + \text{OVERHEAD}$$

- MobileNetV2：$2.33\text{MB} + \text{OVERHEAD} = 6.05\text{MB} \Rightarrow \text{OVERHEAD}\approx 3.7\text{MB}$
- ResNet18：$11.2\text{MB} + \text{OVERHEAD} = 11.19\text{MB} \Rightarrow \text{OVERHEAD}\approx 0$

加上 `.text`（posit 的 metadata-aware 解碼機制）MobileNetV2 也比 int8 多約 **0.8 MB**。

## 4. 這個「固定開銷」是什麼、為何只在 MobileNetV2 顯著

**開銷來自 posit 執行期的 metadata / 解碼機制，其大小隨「層數、tensor 數、通道數」而非「權重量」成長。**

- `.text` 多出的部分：posit 的 GP/metadata-aware 解碼與**逐通道 const metadata 註冊**程式碼。
  （實測：MobileNetV2 有 ALPS 時 `.text`=1.13MB、無 ALPS 時=0.30MB，差 ~0.83MB 即 ALPS 逐通道
  註冊；而 `.rodata` 有/無 ALPS 幾乎相同 6.0MB，代表 rodata 的開銷是 **posit 基礎的逐 tensor/
  通道 metadata**，並非 ALPS 參數本身。）
- `.rodata` 多出的部分：每個常數 tensor 的 posit 描述子 / 逐通道 metadata 陣列 / GP 解碼表 /
  對齊填補。這些是**每個 tensor、每個通道各一份**的固定成本。

MobileNetV2 與 ResNet18 的結構差異（實測）正好放大這個成本：

| | Conv 層 | 輸出通道總數 | 權重參數 | 每通道平均權重 |
|---|---|---|---|---|
| MobileNetV2 | 52 | **17,156** | 2.33 M | **~136** |
| ResNet18 | 20 | 4,900 | 11.2 M | **~2,290** |

MobileNetV2 是 **depthwise-separable** 架構：大量「薄」層與通道（17k 通道），但每通道只有極少
權重。因此「每 tensor／每通道」的固定 metadata 成本（~3.7MB rodata + ~0.8MB text ≈ 4.5MB）
**與其極小的權重負載（2.3MB）相當甚至更大**，於是 posit8 膨脹到 int8 的 2.7×。

ResNet18 是「胖」架構：少數大 conv（4.9k 通道、每通道 ~2,290 權重）。同一份固定 metadata 成本
相對於其 11.2MB 權重**可忽略**，所以 posit8 ≈ int8（1.08×）。

## 5. 一句話結論（可放論文）

> posit `.so` 的體積約為「每權重 1 byte × 權重數 ＋ 與層數/通道數成正比的 posit 執行期
> metadata 固定開銷」。ResNet18 屬少層、大通道的「胖」網路，權重（11.2 M）主導體積，故其
> posit8 幾乎等於 int8；MobileNetV2 屬 depthwise-separable 的「薄」網路，權重僅 2.3 M 卻有 52 層、
> 17,156 個通道，逐 tensor/逐通道的 metadata 與解碼機制（≈4.5 MB）無法被其微小的權重負載
> 攤銷，因而 posit8 約為 int8 的 2.7 倍。此差異源於**架構**（薄的深度可分離 vs 胖的殘差），而非
> posit 位元寬度或 ALPS 參數本身。

> 註：第 1–3 節數字皆為 `size -A` / `nm` 實測；「固定開銷 = 逐 tensor/通道 metadata」為由
> 「有/無 ALPS 的 rodata 幾乎相同」「ResNet18 posit8 每權重僅 1 byte」兩項實測**推論**而得的機制
> 解釋（非直接逐位元組歸因）。若審稿需要，可再用 `readelf -x .rodata` 對單一 tensor 做逐位元組拆解。
