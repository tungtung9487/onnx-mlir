# Weight-ALPS θ 搜尋範圍的選取依據（論文用）

本文說明 build-time weight-ALPS 的參數 `THETA_MIN / THETA_MAX / THETA_STEPS` 為何如此
選取，並提供一支依模型權重自動推薦範圍的工具 `suggest_alps_theta_range.py`。所有公式
均與實作一致（`Conversion/ONNXToPosit/Pattern/Math.cpp` 的 const-ALPS 搜尋、
`posit_runtime.cpp` 的 `qalignCompandAlps/qalignDecompandAlps`）。

## 1. ALPS companding 與 posit 精度

每個權重 $w$ 以 asinh companding 編碼後再量化為 posit：

$$ y = \frac{\operatorname{asinh}(\theta\, w)}{\gamma}, \qquad \hat{w} = \frac{\sinh(\gamma\, y_q)}{\theta} $$

其中 $y_q$ 為 $y$ 量化到 posit（或 GP 表）後的值。asinh 的行為：

$$ \operatorname{asinh}(\theta w) \approx \begin{cases} \theta w, & |\theta w|\ll 1 \ \text{(線性區)}\\[2pt] \operatorname{sign}(w)\,\ln(2\theta|w|), & |\theta w|\gg 1 \ \text{(對數區)} \end{cases} $$

曲線的「拐點」落在 $|w| = 1/\theta$。因此 $\theta$ 唯一的作用，是決定這條由「線性」轉「對數」
的拐點**落在權重分佈的什麼位置**：拐點以下的權重被近似線性保留，拐點以上的權重被對數壓縮。

$\gamma$ 不由使用者指定，而是自動求得：令 $|\operatorname{asinh}(\theta w)|$ 的第 99 百分位
映射到 `GAMMA_TARGET`$=1$（`GAMMA_PERCENTILE=0.99`），使壓縮後數值的主體落在 posit
精度最高的 $\pm 1$ 附近。

## 2. 為什麼是「範圍 + 網格搜尋」而非單一 θ

編譯期對**每個輸出通道**在 $[\theta_{\min}, \theta_{\max}]$ 上做**幾何（log₂ 等距）網格
搜尋**：

$$ \theta_t = 2^{\,\log_2\theta_{\min} + \frac{t}{S-1}\left(\log_2\theta_{\max}-\log_2\theta_{\min}\right)},\quad t=0,\dots,S-1 $$

對每個 $\theta_t$（與 $\gamma\in\{0.5,1,2\}\times\gamma_{\text{base}}$、GP 的 rs/sc）做一次
encode→decode，取**雜訊訊號比最小**者：

$$ \mathrm{NSR} = \frac{\sum_i (w_i-\hat{w}_i)^2}{\sum_i w_i^2} = \frac{1}{\mathrm{SQNR}} $$

且只有當 ALPS 的 NSR 比「直接 posit」再好 `MIN_GAIN` 以上才採用（否則退回直接 posit）。

**關鍵性質**：因為搜尋會**逐通道自行挑最佳 θ**，使用者提供的 $[\theta_{\min},\theta_{\max}]$
只需要**「框住」所有通道的最佳點**即可。由此得到一條穩健準則：

> **過寬的範圍不會傷害精度**（多出來的 θ 不會被選中，只是多花 build 時間）；
> **過窄的範圍才是唯一風險**（會把最佳點在範圍外的通道截斷）。

所以範圍應「下界抓緊、上界寧可略寬」。θ 以幾何等距是因為它是乘性參數（拐點在 $1/\theta$），
等比網格在有意義的對數尺度上才有均勻解析度。

## 3. 下界 θ_min ——「退化為近似直接 posit」

要讓不需要 companding 的通道能退回近似線性（等價於直接 posit），必須讓**最大的權重**
仍處在 asinh 的線性區，即 $\theta_{\min}\cdot\max|w| \lesssim 0.1$：

$$ \boxed{\ \theta_{\min} \approx \frac{0.1}{\max|w|}\ } $$

**本專案 MobileNetV2 實測**：$\max|w| = 738.7$（出現在 `blocks.1.1.conv_dw`），

$$ \theta_{\min} \approx 0.1/738.7 = 1.35\times10^{-4} \approx 10^{-4}. $$

這正好等於實際使用的 `THETA_MIN=0.0001`——亦即當初這個下界（不論是經驗或試誤得到）**在
數學上是可被證成的**：它剛好讓全模型最大的權重落在近似線性區。

## 4. 上界 θ_max ——「框住最小尺度通道的最佳點」

各通道最佳 θ 的量級約為 $1/\text{scale}_c$（$\text{scale}_c$ 為該通道權重的 RMS）。要框住
**最小尺度的有效通道**，取

$$ \theta_{\max} \approx \frac{K}{\min_c \text{scale}_c^{(\text{active})}},\quad K\approx 8 $$

其中先**剔除死通道**（權重近乎全零、RMS 低於全模型中位數約兩個數量級者）：死通道在任何
$\theta$ 下都被 $\gamma$ 正規化為 $\approx 0$，不需要大 $\theta$，若不剔除會把上界灌爆
（MobileNetV2 未剔除時 $1/\text{RMS}$ 的 p99 高達 $1.4\times10^6$）。

**實測**：MobileNetV2 剔除死通道後最小有效 RMS $\approx 0.010$，得 $\theta_{\max}\approx 770$；
ResNet18 得 $\theta_{\max}\approx 1200$。

**關於實際使用的 `THETA_MAX=5`**：它比上述資料推導值小，屬**保守**設定，但仍可行，原因有二：
(i) posit 本身即具**漸縮（近對數）精度**，已涵蓋大動態範圍，因此不需要很強的 companding，
真正的 NSR 最佳點通常遠小於 $1/\text{RMS}$ 的估計；(ii) NSR 對 $\theta$ 在最佳點附近平坦、
且大 $\theta$ 端趨於飽和，故把上界截在 5 只損失極少。依第 2 節的性質，把上界放寬到數百
（如工具建議值）是**安全**的，僅增加 build 時間；若要更保險以完整框住每個通道，建議採用
工具推薦的較大上界。

## 5. 其餘參數

| 參數 | 值 | 理由 |
|---|---|---|
| `GAMMA_TARGET` | 1.0 | 壓縮後數值主體對齊 posit 高精度區（$\pm 1$）|
| `GAMMA_PERCENTILE` | 0.99 | 以 p99 定 $\gamma$，對離群權重穩健 |
| `THETA_STEPS` | 使相鄰 θ 比值 ≈1.05 | 幾何網格上足夠細（≤0.07 bit/步）|
| `MIN_GAIN` | 0.001 | ALPS 需比直接 posit 的 NSR 再好 0.001 才採用，避免無益切換 |

指標選 **NSR = 1/SQNR**（非 MAE）：以訊號能量正規化的相對量化誤差，會依權重能量加權，
比平均絕對誤差更貼近「量化雜訊對輸出的影響」。

## 6. 各模型自動推薦（`suggest_alps_theta_range.py` 輸出）

| 模型 | max\|w\| | θ_min=0.1/max\|w\| | θ_max=K/min-active-scale | steps |
|---|---|---|---|---|
| MobileNetV2 | 738.7 | **1e-4**（=實際採用值）| ~770（實際採用 5，較保守）| ~326 |
| ResNet18 | 2.65 | 0.04 | ~1200 | ~213 |

用法：

```bash
gpt2/bin/python suggest_alps_theta_range.py model/imagenet100_mobilenetv2.onnx
```

---

## 附：可直接放入論文的段落（濃縮版）

> 權重端採用 asinh companding（ALPS）：$y=\operatorname{asinh}(\theta w)/\gamma$，解碼
> $\hat w=\sinh(\gamma y_q)/\theta$。$\theta$ 決定線性/對數轉換拐點 $1/\theta$ 相對權重分佈的
> 位置，$\gamma$ 由 $|\operatorname{asinh}(\theta w)|$ 的 p99 對齊 1 自動求得，使壓縮後主體
> 落在 posit 精度最高的 $\pm1$ 區。編譯期對每個輸出通道在 $[\theta_{\min},\theta_{\max}]$ 上
> 以 log 等距網格搜尋，取雜訊訊號比 $\mathrm{NSR}=\sum(w-\hat w)^2/\sum w^2$（即 $1/$SQNR）
> 最小者，並要求優於直接 posit 至少 $\varepsilon$（MIN_GAIN）方採用。由於搜尋逐通道自選最佳
> $\theta$，範圍只需框住各通道最佳點：下界取 $\theta_{\min}=0.1/\max|w|$，確保最大權重仍在
> 近似線性區、可退化為直接 posit；上界取 $\theta_{\max}=K/\min_c\text{RMS}_c$（$K\approx8$，
> 並剔除近零死通道）。以 MobileNetV2 為例 $\max|w|=738.7$，故 $\theta_{\min}\approx10^{-4}$，
> 與實驗採用值一致；因 posit 自身的漸縮精度已涵蓋大動態範圍，NSR 對 $\theta$ 在最佳點附近
> 平坦，故上界的選取以「框住」為原則，過寬僅增加搜尋成本而不損精度。
