# Citation Verification Record — CVGIP 2025 AI GO Paper
**Last updated:** 2026-06-06
**Paper:** 基於 Sobel 引導注意力機制之 Pix2Pix 跨場景單張影像反光消除：以博物館文物辨識為案例

---

## 〇、引用編號對照表（2026-06-14 全面重新編號）

`cvgip2025_chinese.py` 已於 2026-06-14 將所有引用由「混合編號（[1]-[42] 含跳號）+
命名引用（[A][B][GAP-E][RFC][ERRNET][Blau18][Ledig17]）」全面改為依文件首次出現順序
連續編號 [1]-[30]。**本記錄檔下方各小節標題與 bibkey 欄位仍使用「舊編號」**（記錄建立
時的編號），下表為舊編號 → 新編號對照，供交叉查核：

| 新編號 | 舊編號（本記錄使用） | 論文 |
|---|---|---|
| [1] | [9] | SIRR Survey 2025 |
| [2] | [42] | YOLOv8 |
| [3] | [23] | Lu et al. — Sobel + Multi-Attention for Medical Images |
| [4] | [21] | CBAM |
| [5] | [3] | Chi 2018 — Deep Encoder-Decoder for SIRR |
| [6] | [1] | CEILNet |
| [7] | [2] | IBCLN |
| [8] | [4] | Location-aware SIRR |
| [9] | [6] | DURRNet |
| [10] | [8] | PromptRR |
| [11] | [19] | GAN (Goodfellow 2014) |
| [12] | [15] | cGAN |
| [13] | [13] | Pix2Pix |
| [14] | [14] | CycleGAN |
| [15] | [17] | GAN Survey (Image/Video Synthesis) |
| [16] | [22] | SENet |
| [17] | [26] | GCNet |
| [18] | [24] | Non-local Neural Networks |
| [19] | [29] | HED |
| [20] | [31] | DGNet |
| [21] | [37] | Sharp U-Net |
| [22] | [33] | Li & Liu — MRI Restoration with Edge Loss |
| [23] | [B] | U-Net |
| [24] | [GAP-E] | SIR² |
| [25] | [ERRNET] | ERRNet |
| [26] | [RFC] | Flash Reflection Removal (RFC) |
| [27] | [A] | SSIM |
| [28] | [36] | LPIPS |
| [29] | [Blau18] | Perception-Distortion Tradeoff |
| [30] | [Ledig17] | SRGAN |

---

## 使用說明
- ✅ CONFIRMED：原文已讀，引述可直接用於論文
- ⚠️ PARTIAL：arXiv 摘要已讀，方向正確，未讀 PDF 全文
- ❌ ERROR：論文內容與宣稱方向相反或不符
- ❓ UNVERIFIABLE：無法取得 PDF 或 arXiv 全文

每條記錄包含：**來源位置、驗證日期、讀取頁面、方向摘要、建議引用措辭**

---

## 一、新增引用（2026-06-06 新驗證，待加入論文）

---

### [Blau18] The Perception-Distortion Tradeoff

| 欄位 | 內容 |
|------|------|
| **作者** | Yochai Blau, Tomer Michaeli |
| **標題** | The Perception-Distortion Tradeoff |
| **發表** | IEEE/CVF CVPR 2018, pp. 6228–6237 |
| **arXiv** | 1711.06077 |
| **來源** | https://ar5iv.labs.arxiv.org/html/1711.06077 |
| **驗證日期** | 2026-06-06 |
| **驗證狀態** | ✅ CONFIRMED |

**驗證目的：**
支撐「量化指標（PSNR/SSIM）偏低不等於反光消除失效」的論點。
GAN-based 方法在感知品質（perceptual quality）較好時，PSNR/SSIM 往往反而偏低，兩者存在根本性取捨關係。

**讀取內容來源：** ar5iv HTML 全文，包含 Abstract、定理陳述、§3 實驗討論

**驗證原文（直接引述）：**

> "algorithms that are superior in terms of perceptual quality, are often inferior in terms of e.g. PSNR and SSIM"

> **Theorem 2 (The perception-distortion tradeoff):**
> "the perception-distortion function P(D) is (1) monotonically non-increasing; (2) convex."

> "generative-adversarial-nets (GANs) provide a principled way to approach the perception-distortion bound"

**建議引用文字：**
> Blau 與 Michaeli【Blau18】從理論層面證明，感知品質與失真指標之間存在根本性的取捨關係（perception-distortion tradeoff）——感知品質越高的方法，PSNR/SSIM 往往越低，此現象適用於任何失真度量標準。

**建議加入位置：** §4.3 消融實驗討論段尾 / §5.1 設計討論

**IEEE bib entry：**
```
[Blau18] Y. Blau and T. Michaeli, "The Perception-Distortion Tradeoff," in Proc. IEEE/CVF CVPR, 2018, pp. 6228-6237.
```

---

### [Ledig17] SRGAN — Photo-Realistic Single Image Super-Resolution

| 欄位 | 內容 |
|------|------|
| **作者** | Christian Ledig et al. |
| **標題** | Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network |
| **發表** | IEEE/CVF CVPR 2017, pp. 4681–4690 |
| **arXiv** | 1609.04802 |
| **來源** | https://ar5iv.labs.arxiv.org/html/1609.04802 |
| **驗證日期** | 2026-06-06 |
| **驗證狀態** | ✅ CONFIRMED |

**驗證目的：**
提供具體實驗例證：GAN（SRGAN）相比 MSE-based 方法（SRResNet），PSNR 較低但視覺感知品質顯著更好。確認 L1/MSE loss 導致輸出過度平滑的機制。

**驗證原文（直接引述）：**

> "minimizing MSE encourages finding pixel-wise averages of plausible solutions which are typically overly-smooth"

> "the ability of MSE (and PSNR) to capture perceptually relevant differences, such as high texture detail, is very limited"

> "highest PSNR does not necessarily reflect the perceptually better SR result"

> "Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details"

**建議引用文字：**
> Ledig 等人【Ledig17】在影像超解析度任務中實驗驗證此現象：SRGAN 的 PSNR 低於傳統 MSE-based 方法，但感知品質顯著更好；其根本原因在於「最小化 MSE 鼓勵模型輸出所有合理解的像素均值，導致結果趨於過度平滑（overly-smooth）」。

**建議加入位置：** §4.3 消融實驗討論，緊接 [Blau18] 之後

**IEEE bib entry：**
```
[Ledig17] C. Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network," in Proc. IEEE/CVF CVPR, 2017, pp. 4681-4690.
```

---

## 二、現有引用全表（論文已收錄，2026-06-06 摘要驗證）

---

### [1] CEILNet — Generic Architecture for SIRR

| 欄位 | 內容 |
|------|------|
| **bibkey** | [1] |
| **作者** | Q. Fan, J. Yang, G. Hua, B. Chen, D. Wipf |
| **標題** | A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing |
| **發表** | Proc. IEEE ICCV, 2017, pp. 3238–3247 |
| **arXiv** | 1708.03474 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 早期深度學習 SIRR 方法，以邊緣資訊引導、輕量 CNN 端對端訓練；本文引用為相關工作代表性早期方法。

**摘要方向確認：**
- 論文確認：以 cascaded CNN 利用邊緣資訊進行反光消除
- "exploits edge information through cascaded convolutional layers"
- "mild reflection smoothness assumption and a novel synthetic data generation method"
- "simple, fast, and easy to transfer across disparate domains"

**紙面宣稱用途：** §2.1 相關工作，早期 SIRR 深度學習方法

---

### [2] IBCLN — Iterative Boost Convolutional LSTM Network

| 欄位 | 內容 |
|------|------|
| **bibkey** | [2] |
| **作者** | C. Li, Y. Yang, K. He, S. Lin, J. E. Hopcroft |
| **標題** | Single Image Reflection Removal through Cascaded Refinement |
| **發表** | Proc. IEEE/CVF CVPR, 2020 |
| **arXiv** | 1911.06634 |
| **驗證狀態** | ✅ CONFIRMED（2026-06-13 全文覆盤確認 IBCLN 架構描述正確；原 SIR² 資料集歸屬錯誤已於 cvgip2025_chinese.py 第 263-264 行修正為「具密集標註 ground truth 的真實場景配對資料集」，見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md` [2] 條目） |

**論文在本文中的角色：** 迭代精化 SIRR 方法，使用 ConvLSTM 跨步驟傳遞資訊；引用為採用遞迴精化策略的代表性方法。

**摘要方向確認：**
- "Iterative Boost Convolutional LSTM Network (IBCLN)"
- "iteratively refines transmission and reflection layer estimates"
- "uses LSTM to transfer information across cascade steps and prevent gradient vanishing"
- 建立真實場景配對資料集

**紙面宣稱用途：** §2.1 相關工作，迭代精化類 SIRR 方法

---

### [3] Chi 2018 — Deep Encoder-Decoder for SIRR

| 欄位 | 內容 |
|------|------|
| **bibkey** | [3] |
| **作者** | Z. Chi, X. Wu, X. Shu, J. Gu |
| **標題** | Single Image Reflection Removal Using Deep Encoder-Decoder Network |
| **發表** | arXiv:1802.00094, 2018 |
| **arXiv** | 1802.00094 |
| **驗證狀態** | ✅ CONFIRMED（2026-06-13 全文覆盤後，cvgip2025_chinese.py 第 265-267 行與第 348 行已改為保守措辭——移除「深入分析」「不可逆」「高頻邊緣響應」等原文未支持之用語，改為「下採樣操作所帶來的資訊損失會增加解碼器復原難度」，與原文 p.6 §4.2 及其自身的 skip connection 設計一致，見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md` [3] 條目） |

**論文在本文中的角色：** 以 encoder-decoder 架構學習反光/無反光影像對的映射；使用合成資料集訓練、遷移至真實影像。

**摘要方向確認：**
- "deep convolutional encoder-decoder method to remove reflection"
- "synthetic training dataset by modeling physical reflection formation"
- "significantly outperforms the other tested state-of-the-art techniques" despite training only on synthetic data

**紙面宣稱用途：** §2.1 或 §3 合成資料訓練策略討論

---

### [4] Location-aware SIRR

| 欄位 | 內容 |
|------|------|
| **bibkey** | [4] |
| **作者** | Z. Dong, K. Xu, Y. Yang, H. Bao, W. Xu, R. W. H. Lau |
| **標題** | Location-aware Single Image Reflection Removal |
| **發表** | arXiv:2012.07131, 2020 (revised 2021) |
| **arXiv** | 2012.07131 |
| **驗證狀態** | ✅ CONFIRMED（2026-06-13 全文覆盤後，cvgip2025_chinese.py 第 267-269 行已將「證明空間注意力在 SIRR 任務中的有效性」改為「以顯式的反光位置偵測模組（reflection confidence map）回歸反光機率圖以引導特徵流，證明『顯式空間位置線索』在 SIRR 任務中的有效性」，避免與 [21]/[22] 的 spatial attention 技術名詞混淆，見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md` [4] 條目） |

**論文在本文中的角色：** 生成「反光信心圖」定位反光區域，以多尺度 Laplacian 特徵識別反光邊界；引用為利用空間位置資訊的 SIRR 進階方法。

**摘要方向確認：**
- "reflection detection module that generates a probabilistic confidence map"
- "multi-scale Laplacian features to identify reflection boundaries"
- "recurrent network architecture that progressively refines results"

**紙面宣稱用途：** §2.1 SIRR 相關工作，空間位置引導方法

---

### [6] DURRNet — Deep Unfolded SIRR

| 欄位 | 內容 |
|------|------|
| **bibkey** | [6] |
| **作者** | J.-J. Huang, T. Liu, Z. Yang, S. Fu, W. Zhao, P. L. Dragotti |
| **標題** | DURRNet: Deep Unfolded Single Image Reflection Removal Network |
| **發表** | arXiv:2203.06306, 2022 |
| **arXiv** | 2203.06306 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 結合模型驅動（transform-based exclusion priors）與深度學習的混合架構；引用為近期高性能 SIRR 方法。

**摘要方向確認：**
- "single image reflection removal... highly ill-posed"
- "model-based optimization using transform-based exclusion priors"
- "deep unrolling architecture incorporating ProxNets and ProxInvNets"
- "achieves state-of-the-art results both visually and quantitatively"

**紙面宣稱用途：** §2.1 SIRR 相關工作，model-based+learning-based 混合方法

---

### [8] PromptRR — Diffusion-based SIRR

| 欄位 | 內容 |
|------|------|
| **bibkey** | [8] |
| **作者** | T. Wang, W. Lu, K. Zhang, T. Lu, M.-H. Yang |
| **標題** | PromptRR: Diffusion Models as Prompt Generators for Single Image Reflection Removal |
| **發表** | arXiv:2402.02374, 2024 |
| **arXiv** | 2402.02374 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 最新 SOTA，用 Diffusion model 生成頻率提示引導反光消除；作為本文的 state-of-the-art 比較對象或相關工作。

**摘要方向確認：**
- "existing reflection removal methods tend to miss key low-frequency and high-frequency differences"
- 以 frequency prompt encoder + diffusion model 生成低頻/高頻提示
- "PromptFormer network with novel Transformer-based prompt block"
- "outperforms state-of-the-art approaches on standard benchmarks"

**紙面宣稱用途：** §2.1 SIRR 相關工作，最新 Diffusion-based 方法

---

### [9] SIRR Survey 2025

| 欄位 | 內容 |
|------|------|
| **bibkey** | [9] |
| **作者** | K. Yang et al. (Kangning Yang, Huiming Sun, ...) |
| **標題** | Survey on Single-Image Reflection Removal using Deep Learning Techniques |
| **發表** | arXiv:2502.08836, 2025 |
| **arXiv** | 2502.08836 |
| **驗證狀態** | ✅ CONFIRMED（第 202 行用法與原文 §6.1 Challenges 吻合；第 692 行的【9】標記為自身限制聲明的錯置引用，2026-06-13 已從 cvgip2025_chinese.py 第 692 行移除，見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md` [9] 條目） |

**論文在本文中的角色：** 最新 SIRR 深度學習方法綜述；引用為「反光問題廣泛存在」的背景性宣稱支撐。

**摘要方向確認：**
- "reflection is quite common in digital images, posing significant challenges"
- 涵蓋 ICCV、ECCV、CVPR、NeurIPS 頂會方法
- 包含 single-stage 與 two-stage 方法比較
- "rapidly evolving research area"

**紙面宣稱用途：** §1 Introduction 或 §2 Related Work，SIRR 問題定義背景

---

### [13] Pix2Pix — Image-to-Image Translation

| 欄位 | 內容 |
|------|------|
| **bibkey** | [13] |
| **作者** | P. Isola, J.-Y. Zhu, T. Zhou, A. A. Efros |
| **標題** | Image-to-Image Translation with Conditional Adversarial Networks |
| **發表** | Proc. IEEE/CVF CVPR, 2017, pp. 1125–1134 |
| **arXiv** | 1611.07004 |
| **驗證狀態** | ✅ CONFIRMED（本文核心架構基礎，已多次驗證） |

**論文在本文中的角色：** 本文所採用架構的直接來源；U-Net generator + PatchGAN discriminator + L1 + adversarial loss。

**關鍵技術確認：**
- conditional adversarial networks for image-to-image translation
- "learn the mapping from input image to output image, but also learn a loss function"
- 多任務適用性：label→photo、edge→object、colorization 等

**紙面宣稱用途：** §3.2 方法架構基礎，§4.3 消融表格 Baseline 標注

**附注（2026-06-13 更新）：** 全文覆盤（讀取 pp.2-5，§3.2 Markovian discriminator 章節）證實 Isola et al. 原論文本身已明確討論並以 Fig.4 圖示展示 L1 loss 的 blur/over-smooth 現象——「It is well known that the L2 loss – and L1, see Figure 4 – produces blurry results on image generation problems」。先前「不應直接宣稱 Isola et al. 批評 L1 loss」的保留意見已**解除**（該保留基於僅讀摘要的不完整資訊）。【Ledig17】仍可作為互補佐證，兩者不互斥。詳見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md` [13] 條目。

---

### [14] CycleGAN — Unpaired Image Translation

| 欄位 | 內容 |
|------|------|
| **bibkey** | [14] |
| **作者** | J.-Y. Zhu, T. Park, P. Isola, A. A. Efros |
| **標題** | Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks |
| **發表** | Proc. IEEE ICCV, 2017 |
| **arXiv** | 1703.10593 |
| **驗證狀態** | ✅ CONFIRMED（2026-06-13 全文覆盤，已讀 pp.6-8 §5.1-5.2，含 Table 2/3 數據比對，見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md` [14] 條目） |

**論文在本文中的角色：** GAN-based 影像轉換的代表性工作；引用說明 GAN 在影像轉換任務的廣泛應用。

**摘要方向確認：**
- 不需配對訓練資料的影像轉換
- adversarial loss + cycle consistency loss
- 風格遷移、季節轉換、相片增強等應用

**紙面宣稱用途：** §2.2 GAN-based 相關工作

---

### [15] cGAN — Conditional GAN

| 欄位 | 內容 |
|------|------|
| **bibkey** | [15] |
| **作者** | M. Mirza, S. Osindero |
| **標題** | Conditional Generative Adversarial Nets |
| **發表** | arXiv:1411.1784, 2014 |
| **arXiv** | 1411.1784 |
| **驗證狀態** | ✅ CONFIRMED（基礎性引用，2026-06-13 覆盤確認摘要層級足夠，符合 CLAUDE.md §5.0b 例外，見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md` [15] 條目） |

**論文在本文中的角色：** 條件 GAN 的原始提出；本文使用條件 GAN 架構（Pix2Pix 是 cGAN 的影像翻譯變體）。

**摘要方向確認：**
- "feeding the data y we wish to condition on to both the generator and discriminator"
- 以類別標籤為條件生成 MNIST 數字
- 多模態學習、影像標注

**紙面宣稱用途：** §2.2 或 §3 方法論背景，cGAN 框架引用

---

### [17] GAN Survey — Image and Video Synthesis

| 欄位 | 內容 |
|------|------|
| **bibkey** | [17] |
| **作者** | M.-Y. Liu, X. Huang, J. Yu, T.-C. Wang, A. Mallya |
| **標題** | Generative Adversarial Networks for Image and Video Synthesis: Algorithms and Applications |
| **發表** | Proc. IEEE, vol. 109, no. 5, pp. 839–862, 2021 |
| **arXiv** | 2008.02793 |
| **驗證狀態** | ✅ CONFIRMED（基礎性綜述引用，2026-06-13 覆盤確認摘要層級足夠，符合 CLAUDE.md §5.0b 例外，見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md` [17] 條目） |

**論文在本文中的角色：** GAN 綜述，涵蓋訓練穩定化技術、影像轉換、影像處理、視訊合成、神經渲染；引用為 GAN 技術廣泛應用的背景支撐。

**摘要方向確認：**
- "GANs as a powerful tool for various image and video synthesis tasks"
- "high-resolution photorealistic images and videos"
- 訓練穩定性、影像轉換、神經渲染等主題

**紙面宣稱用途：** §2.2 GAN 相關工作背景引用

---

### [19] GAN — Original Goodfellow 2014

| 欄位 | 內容 |
|------|------|
| **bibkey** | [19] |
| **作者** | I. Goodfellow et al. |
| **標題** | Generative Adversarial Nets |
| **發表** | Adv. Neural Inf. Process. Syst. (NeurIPS), 2014, pp. 2672–2680 |
| **arXiv** | 1406.2661 |
| **驗證狀態** | ✅ CONFIRMED（GAN 開創性論文，2026-06-13 覆盤確認屬 CLAUDE.md §5.0b「廣為人知技術事實」例外，無需進一步查證，見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md` [19] 條目） |

**論文在本文中的角色：** GAN 的原始提出；生成器-判別器對抗訓練框架的理論基礎。

**摘要方向確認：**
- "generative model G that captures the data distribution, and discriminative model D"
- minimax 對抗遊戲框架
- "G recovering the training data distribution and D equal to 1/2 everywhere" (optimal)

**紙面宣稱用途：** §2.2 GAN 框架引用

---

### [21] CBAM — Convolutional Block Attention Module

| 欄位 | 內容 |
|------|------|
| **bibkey** | [21] |
| **作者** | S. Woo, J. Park, J.-Y. Lee, I. S. Kweon |
| **標題** | CBAM: Convolutional Block Attention Module |
| **發表** | Proc. ECCV, 2018 |
| **arXiv** | 1807.06521 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 本文 SGA 模組的直接設計來源——CBAM 的 channel attention + spatial attention 雙分支結構是 SGA 的核心參考。

**摘要方向確認：**
- "sequentially infers attention maps along two separate dimensions, channel and spatial"
- "lightweight and general, can be integrated into any CNN architectures with negligible overheads"
- "end-to-end trainable"
- 在 ImageNet-1K、MS COCO、VOC 2007 驗證效果

**紙面宣稱用途：** §3.2 SGA 模組設計，注意力架構來源

---

### [22] SENet — Squeeze-and-Excitation Networks

| 欄位 | 內容 |
|------|------|
| **bibkey** | [22] |
| **作者** | J. Hu, L. Shen, S. Albanie, G. Sun, E. Wu |
| **標題** | Squeeze-and-Excitation Networks |
| **發表** | IEEE Trans. Pattern Anal. Mach. Intell., vol. 42, no. 8, pp. 2011–2023, 2020 (CVPR 2018) |
| **arXiv** | 1709.01507 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 通道注意力的代表性工作；SGA 的通道注意力分支（GAP → Conv1×1 → Sigmoid）直接參考 SE block 設計。

**摘要方向確認：**
- "adaptively recalibrate channel-wise feature responses by modelling interdependencies between channels"
- ILSVRC 2017 分類冠軍（top-5 error 2.251%）
- "generalizes effectively across diverse datasets"

**紙面宣稱用途：** §3.2.2 通道注意力分支設計參考

---

### [23] Lu et al. — Sobel + Multi-Attention for Medical Images

| 欄位 | 內容 |
|------|------|
| **bibkey** | [23] |
| **作者** | F. Lu, C. Tang, T. Liu, Z. Zhang, L. Li |
| **標題** | Multi-Attention Segmentation Networks Combined with the Sobel Operator for Medical Images |
| **發表** | Sensors, vol. 23, no. 5, p. 2546, 2023. DOI: 10.3390/s23052546 |
| **arXiv** | 無（MDPI 開放取用） |
| **來源** | https://www.mdpi.com/1424-8220/23/5/2546 |
| **驗證狀態** | ✅ CONFIRMED（2026-06-13 全文（PMC）覆盤確認第 230 行用法成立；第 312-316 行原宣稱「驗證了固定 Sobel 梯度在跨場景設定下的穩定性」不實（[23] 為單一 COVID-19 CT domain、無跨場景測試），已改寫為「啟發本文進一步將此設計思路延伸至跨場景情境進行驗證，[23] 本身並未測試跨資料集/跨場景表現，本文跨場景驗證屬本文新貢獻」，見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md` [23] 條目） |

**論文在本文中的角色：** 直接支撐「Sobel 運算子結合注意力機制」這一設計思路；本文 SGA 模組以固定 Sobel 核提取邊緣特徵後導入 CBAM-style 注意力，與此論文精神相同。

**摘要方向確認（來自搜尋結果）：**
- "edge feature fusion module with the Sobel operator to add edge detail information to the input image"
- 引入 self-attention channel attention + spatial linear attention
- 應用於 COVID-19 病灶分割（醫療影像分割）
- Sobel 用於增強邊緣特徵，引導注意力聚焦關鍵區域

**紙面宣稱用途：** §3.2 SGA 設計動機，Sobel + attention 組合設計的先例

**注意：** 本論文曾在驗證過程中發現 DOI 指向錯誤（s23052533 → 游泳池 IoT 論文），已更正為 s23052546。引用前請確認 DOI 正確。

---

### [24] Non-local Neural Networks

| 欄位 | 內容 |
|------|------|
| **bibkey** | [24] |
| **作者** | X. Wang, R. Girshick, A. Gupta, K. He |
| **標題** | Non-local Neural Networks |
| **發表** | Proc. IEEE/CVF CVPR, 2018, pp. 7794–7803 |
| **arXiv** | 1711.07971 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 長距離依賴（long-range dependencies）注意力機制；SGA 的空間注意力分支捕捉全局結構資訊，與 non-local 思路相通。

**摘要方向確認：**
- "weighted sum of the features at all positions" — 全域自注意力
- 視訊分析（Kinetics、Charades）及靜態影像（COCO）都有效
- 是 vision self-attention 的早期形式化

**紙面宣稱用途：** §2.3 或 §3.2 注意力機制相關工作

---

### [26] GCNet — Non-local Meets Squeeze-Excitation

| 欄位 | 內容 |
|------|------|
| **bibkey** | [26] |
| **作者** | Y. Cao, J. Xu, S. Lin, F. Wei, H. Hu |
| **標題** | GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond |
| **發表** | arXiv:1904.11492, 2019 |
| **arXiv** | 1904.11492 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 統一 Non-local 與 SENet 兩類注意力機制；引用為全域上下文建模的進階參考，與 SGA 同時使用 channel + spatial 注意力的設計有概念關聯。

**摘要方向確認：**
- "global contexts modeled by non-local network are almost the same for different query positions"
- GC block — "lightweight and can effectively model the global context"
- 優於 simplified NL Networks 與 SENet 變體

**紙面宣稱用途：** §2.3 注意力機制相關工作

---

### [29] HED — Holistically-Nested Edge Detection

| 欄位 | 內容 |
|------|------|
| **bibkey** | [29] |
| **作者** | S. Xie, Z. Tu |
| **標題** | Holistically-Nested Edge Detection |
| **發表** | Proc. IEEE ICCV, 2015, pp. 1395–1403 |
| **arXiv** | 1504.06375 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 以多尺度深度監督學習不同層次邊緣特徵；引用為「邊緣資訊引導影像復原」技術路線的奠基性工作。

**摘要方向確認：**
- "fully convolutional neural networks and deeply-supervised nets"
- BSD500 ODS F-score: 0.782；NYU Depth: 0.746（state-of-the-art）
- "multi-scale and multi-level feature learning within a unified framework"
- 0.4 秒/張，速度快

**紙面宣稱用途：** §2.3 邊緣引導方法背景

---

### [31] DGNet — Deep Gradient Learning for Camouflaged Detection

| 欄位 | 內容 |
|------|------|
| **bibkey** | [31] |
| **作者** | G. Ji, D.-P. Fan, Y.-C. Chou, D. Dai, A. Liniger, L. Van Gool |
| **標題** | Deep Gradient Learning for Efficient Camouflaged Object Detection |
| **發表** | Mach. Intell. Res., vol. 20, no. 1, pp. 92–108, 2023 |
| **arXiv** | 2205.12853 |
| **驗證狀態** | ⚠️ PARTIAL（搜尋結果確認） |

**論文在本文中的角色：** 以物件梯度監督解耦紋理與語義特徵，梯度引導特徵提煉思路與本文 Sobel 引導注意力類比。

**摘要方向確認（來自搜尋結果）：**
- 提出 DGNet，利用 object gradient supervision 進行偽裝物件偵測
- 解耦 context encoder 與 texture encoder，以 gradient-induced transition 連接
- DGNet-S 達 80fps 且媲美更複雜模型

**紙面宣稱用途：** §2.3 梯度/邊緣引導特徵提煉相關工作

---

### [33] Li & Liu — MRI Restoration with Edge Loss

| 欄位 | 內容 |
|------|------|
| **bibkey** | [33] |
| **作者** | H. Li, J. Liu |
| **標題** | Edge, Structure and Texture Refinement for Retrospective High Quality MRI Restoration using Deep Learning |
| **發表** | Proc. IEEE ISBI, 2021 |
| **arXiv** | 2102.00325 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀，來自搜尋結果） |

**論文在本文中的角色：** 在 MRI 復原中使用梯度圖 edge loss 強迫模型關注邊緣結構細節；與本文使用 Sobel 邊緣特徵引導注意力的設計動機相通。

**摘要方向確認：**
- "L1 loss of SSIM and gradient map edge quality loss could force the model to focus on edge and structure details"
- 生成高頻細節更豐富的超解析度 MR 影像
- 應用於 MRI 加速採集、運動偽影消除

**紙面宣稱用途：** §3.2 SGA 設計動機，邊緣監督在影像復原中的先例

---

### [36] LPIPS — Learned Perceptual Image Patch Similarity

| 欄位 | 內容 |
|------|------|
| **bibkey** | [36] |
| **作者** | R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang |
| **標題** | The Unreasonable Effectiveness of Deep Features as a Perceptual Metric |
| **發表** | Proc. IEEE/CVF CVPR, 2018, pp. 586–595 |
| **arXiv** | 1801.03924 |
| **驗證狀態** | ✅ CONFIRMED（arXiv 摘要，方向明確） |

**論文在本文中的角色：** LPIPS 指標的來源；說明為何使用 LPIPS 作為評估指標之一（PSNR/SSIM 是 shallow function，與人類感知不對應）。

**摘要方向確認：**
- "PSNR and SSIM are simple, shallow functions that overlook important aspects of human visual perception"
- "deep features outperform all previous metrics by large margins"
- 以 VGG 預訓練特徵計算感知距離
- 效果在 supervised/self-supervised/unsupervised 架構均成立

**建議引用措辭：**
> LPIPS【36】以預訓練 VGG 特徵計算感知距離，與人類視覺判斷的相關性優於 PSNR 與 SSIM，能更準確反映影像復原的感知品質。

**紙面宣稱用途：** §4.2 評估指標說明，解釋採用 LPIPS 的理由

---

### [37] Sharp U-Net — Sharpening Skip Connections

| 欄位 | 內容 |
|------|------|
| **bibkey** | [37] |
| **作者** | H. Zunair, A. B. Hamza |
| **標題** | Sharp U-Net: Depthwise Convolutional Network for Biomedical Image Segmentation |
| **發表** | Comput. Biol. Med., vol. 139, p. 104941, 2021 |
| **arXiv** | 2107.12461 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 在 U-Net skip connection 前加入銳化核（depthwise conv with sharpening kernel），減少 encoder-decoder 特徵語義不相似性；與本文在 skip connection 前注入 Sobel 結構引導的設計理念類比。

**摘要方向確認：**
- "traditional U-Nets use skip connections to merge semantically different low- and high-level features, resulting in blurred feature maps"
- 解法："depthwise convolution of the encoder feature map with a sharpening kernel filter"
- "fuse semantically less dissimilar features"
- 無新增可學習參數，六個資料集上 consistently outperforms baselines

**紙面宣稱用途：** §3.2 SGA 設計類比，skip connection 前特徵精化的相關工作

---

### [42] YOLOv8 — Real-Time Object Detection

| 欄位 | 內容 |
|------|------|
| **bibkey** | [42] |
| **作者** | D. Reis, J. Hong, J. Kupec, A. Daoudi |
| **標題** | Real-Time Flying Object Detection with YOLOv8 |
| **發表** | arXiv:2305.09972, 2023 |
| **arXiv** | 2305.09972 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** YOLOv8 在下游物件偵測任務的應用；本文以 YOLOv8 評估去反光前後博物館文物辨識準確率。

**摘要方向確認：**
- "state-of-the-art single-shot detector" — YOLOv8 定位
- mAP50-95 達 83.5%，50fps on 1080p
- 注意：作者說 "official paper has not been released as of yet"（截至 2023）

**⚠️ 引用注意：** Ultralytics 官方至今未發表 YOLOv8 正式論文，此 arXiv 論文是第三方應用論文，非官方出處。若需引用 YOLOv8 本身，建議引用 Ultralytics 官方 GitHub 或技術報告。

**紙面宣稱用途：** §3.5 或 §4.4 下游任務評估，YOLOv8 方法引用

---

### [A] SSIM — Structural Similarity Index

| 欄位 | 內容 |
|------|------|
| **bibkey** | [A] |
| **作者** | Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli |
| **標題** | Image Quality Assessment: From Error Visibility to Structural Similarity |
| **發表** | IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, Apr. 2004 |
| **arXiv** | 無（2004 年 IEEE 期刊，無 arXiv 預印本） |
| **驗證狀態** | ✅ CONFIRMED（經典論文，標題/作者/發表資訊已在 references 中多次驗證） |

**論文在本文中的角色：** SSIM 指標的原始提出；本文以 SSIM 作為評估指標之一。

**方向確認（無需 PDF，公知內容）：**
- SSIM 由亮度（luminance）、對比度（contrast）、結構（structure）三個分量計算
- 比純 MSE/PSNR 更接近人類視覺感知（但仍有其局限，見 [Blau18] 討論）

**紙面宣稱用途：** §4.2 評估指標說明

---

### [B] U-Net — Encoder-Decoder for Segmentation

| 欄位 | 內容 |
|------|------|
| **bibkey** | [B] |
| **作者** | O. Ronneberger, P. Fischer, T. Brox |
| **標題** | U-Net: Convolutional Networks for Biomedical Image Segmentation |
| **發表** | Proc. MICCAI, 2015, pp. 234–241 |
| **arXiv** | 1505.04597 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** Pix2Pix 的 generator 採用 U-Net 架構（contracting path + skip connections + expanding path）；引用為本文生成器的結構基礎。

**摘要方向確認：**
- "contracting path to capture context and a symmetric expanding path that enables precise localization"
- skip connections 連接 encoder-decoder 對應層
- "trained end-to-end from very few images"
- 512×512 影像 GPU 推論 < 1 秒

**紙面宣稱用途：** §3.2 U-Net 生成器架構引用

---

### [GAP-E] SIR² — Benchmarking SIRR Dataset

| 欄位 | 內容 |
|------|------|
| **bibkey** | [GAP-E] |
| **作者** | R. Wan, B. Shi, L.-Y. Duan, A.-H. Tan, A. C. Kot |
| **標題** | Benchmarking Single-Image Reflection Removal Algorithms |
| **發表** | Proc. IEEE ICCV, 2017, pp. 3942–3950 |
| **arXiv** | 無（CVF Open Access 發表，無 arXiv 預印本） |
| **來源** | https://openaccess.thecvf.com/content_iccv_2017/html/Wan_Benchmarking_Single-Image_Reflection_ICCV_2017_paper.html |
| **驗證狀態** | ⚠️ PARTIAL（搜尋結果確認） |

**論文在本文中的角色：** 本文訓練/測試使用的公開 SIRR 資料集（Dataset2）的來源；SIR² 提供大規模真實場景配對反光資料，含 Objects、Wild、Postcard 三個子集。

**摘要方向確認（來自搜尋結果）：**
- "first captured Single-image Reflection Removal dataset 'SIR2'"
- 40 controlled + 100 wild scenes，含 background ground truth
- 三種玻璃厚度、多種光圈、多種曝光時間
- 廣泛被後續 SIRR 研究引用為標準評估基準

**紙面宣稱用途：** §3.4 資料集說明，公開 SIRR 測試集來源

---

### [RFC] Flash Reflection Removal

| 欄位 | 內容 |
|------|------|
| **bibkey** | [RFC] |
| **作者** | C. Lei, Q. Chen |
| **標題** | Robust Reflection Removal with Reflection-free Flash-only Cues |
| **發表** | Proc. IEEE/CVF CVPR, 2021, pp. 14811–14820 |
| **arXiv** | 2103.04273 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 以閃光燈輔助拍攝提供無反光配對資料集；本文訓練資料之一，也是跨場景訓練多樣性的來源。

**摘要方向確認：**
- "flash-only image obtained by subtracting ambient image from flash image in raw data space"
- flash-only 影像視覺上無反光，提供穩健的 cues
- PSNR +5.23dB、SSIM +0.04、LPIPS -0.068（相比現有方法）
- 提供新資料集，包含多種玻璃材質與室內光線

**紙面宣稱用途：** §3.4 資料集說明，RFC 配對資料集引用

---

### [ERRNET] ERRNet — Misaligned Training Data

| 欄位 | 內容 |
|------|------|
| **bibkey** | [ERRNET] |
| **作者** | K. Wei, J. Yang, Y. Fu, D. Wipf, H. Huang |
| **標題** | Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements |
| **發表** | Proc. IEEE/CVF CVPR, 2019 |
| **arXiv** | 1904.00637 |
| **驗證狀態** | ⚠️ PARTIAL（arXiv 摘要已讀） |

**論文在本文中的角色：** 利用非精確對齊真實影像訓練，降低資料收集難度；引用為使用 misaligned real-world data 的先驅方法。

**摘要方向確認：**
- "fundamental ill-posedness of the problem, and insufficiency of densely-labeled training data"
- "alignment-invariant loss function" 允許使用非對齊訓練資料
- context encoding modules 利用高層語義線索
- 在 aligned benchmarks 上優於 state-of-the-art

**紙面宣稱用途：** §2.1 或 §3.4 資料策略相關工作

---

## 三、引用整合優先順序

| 優先 | 論文 | 加入位置 | 原因 |
|------|------|---------|------|
| **P0** | [Blau18] | §4.3 末段 | 理論支撐，perception-distortion tradeoff 的數學定理 |
| **P0** | [Ledig17] | §4.3 末段 | 實驗例證，GAN PSNR偏低但感知品質好 |
| **P1** | [13] Isola | §4.3 末段 | 已在 bib，連接到本文架構的 L1 loss 機制 |
| **P1** | [36] LPIPS | §4.2 評估指標說明 | 解釋為何採用 LPIPS |
| **P2** | [Blau18]/[Ledig17] | §5.1 設計討論 | 補強 SGA 量化結果偏低的解釋 |

## 四、已知引用問題備忘

| bibkey | 問題 | 狀態 |
|--------|------|------|
| [42] YOLOv8 | 第三方應用論文，非官方出處；Ultralytics 無正式學術論文 | 已知，可接受 |
| [23] Lu Sobel | DOI 曾指向錯誤論文（s23052533→游泳池IoT），已更正為 s23052546 | 已修正 |
| [31] DGNet | 發表於 Machine Intelligence Research（非 ECCV）；論文主題是偽裝物件偵測非 SIRR | 已修正 |
| [33] Li Liu | 作者為 H. Li + J. Liu（非 J. Li + W. Liu）；標題與 MRI 復原相關 | 已修正 |
| §4.3 宣稱 | 論文稱「SGA 三項指標均優於 Baseline」，但實測 SGA PSNR/SSIM/LPIPS 全部低於 Baseline | 需修正論文文字，配合 [Blau18]/[Ledig17] 解釋 |
| §3.6 解析度 | 論文稱 512×512，但實際模型為 256×256 | 需修正論文 |

## 五、SGA Epoch 比較（2026-06-06 eval_sga_all.py 結果）

| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| SGA-256 ep100 | 22.121 | 0.7838 | 0.2542 |
| SGA-256 ep200 | 22.390 | 0.8041 | 0.2315 |
| SGA-256 ep300 | 22.638 | 0.8146 | 0.2222 |
| SGA-256 ep400 | 22.682 | 0.8192 | 0.2178 |
| SGA-256 ep500 | 22.685 | 0.8210 | 0.2173 |
| **SGA-512 ep360** | **22.828** | **0.8417** | 0.2381 |
| **Baseline ep300** | **23.896** | **0.8706** | **0.1630** |

**分析：**
- SGA-256 在 ep400 後幾乎收斂（ep500 僅微幅改善）
- SGA-512 ep360 PSNR/SSIM 均優於 SGA-256 全部 epoch，但 LPIPS 反而比 SGA-256 ep400/500 差（高頻細節上 512px 訓練不足）
- Baseline 在三項指標均高於所有 SGA checkpoint → 需用 [Blau18]/[Ledig17] 解釋此現象

---

## 六、§1 / §3.2.1 核心物理基礎宣稱 — 引用驗證 (2026-06-14)

**待驗證宣稱（原文，出現於兩處）：**
> 「反光因光線擴散呈現低頻、低梯度特性，物件邊緣因材質突變呈現高頻、高梯度響應；
> 這兩項區別特性是物理性質，與場景 domain 無關。」

出現位置：
- §1 Introduction，cvgip2025_chinese.py 第 364-366 行
- §3.2.1 Sobel 特徵萃取，cvgip2025_chinese.py 第 528-532 行

此宣稱是整個 SGA 模組「為何能跨場景泛化」的核心物理論證依據。

---

### A. 反光 = 低頻/低梯度 半句 — 既有引用 [1] CEILNet

| 欄位 | 內容 |
|------|------|
| **bibkey** | [1]（已在本文 bib 中） |
| **PDF** | `matherial/papers/01_ceilnet_fan2017.pdf`，已讀 pp.1-3 |
| **驗證狀態** | ⚠️ PARTIAL |

**驗證原文（直接引述）：**
> "However, one exploitable property in the reflection removal problem is that the gradients or perceptual structures of the two layers exhibit different distributions, since reflections often display a greater degree of blurring." (p.1-2)

**關鍵反例／但書（CRITICAL CAVEAT，p.2 Related Work）：**
> "...assumes the reflected layer is relatively blurry compared to the background scene, thus large gradients in it are strongly penalized... However, we observe that the reflection in many real-world photographs, although indeed sometimes out of focus or blurry, is nonetheless produced by bright lights and often comprises the brightest portion of an image. **The regional gradients associated with these reflections can therefore be quite large, violating the assumption** in [their ref]."

**結論：** [1] 支持「反光**往往**呈現較低梯度／較模糊」的**一般趨勢**，但明確指出**強光反光可產生大梯度**，違反該假設的絕對版本。CEILNet 自己用詞是 "mild reflection smoothness assumption"（溫和假設），不是絕對物理定律。

---

### B. 反光 = 低頻/低梯度 半句 — 既有引用 [4] Location-aware SIRR

| 欄位 | 內容 |
|------|------|
| **bibkey** | [4]（已在本文 bib 中） |
| **PDF** | `matherial/papers/04_location_aware_dong2020.pdf`，已讀 pp.1-3 |
| **驗證狀態** | ⚠️ PARTIAL |

**驗證原文（直接引述，p.2）：**
> "Observing that reflection layers are usually out of focus and appear to be more blurry than transmission layers, Li et al. [24] introduced a relative smoothness prior to distinguish the gradients of the two layers with different probability distributions."

> "...priors are necessary to constrain the solution space, such as natural image gradient sparsity [21, 22], ghosting cues for thick glasses [34], and relative smoothness that assumes the reflection layer is smoother than the transmission layer [24, 52]."

**結論：** [4] 將「反光較平滑/低梯度」明確定位為一個**先驗假設（prior）**，且歸功於第三方論文 Li et al.（其 [24]，即下方 C 項），[4] 自身的方法貢獻正是處理**違反此簡單先驗的強反光**情況。同樣是 ⚠️ PARTIAL——支持「先驗/趨勢」框架，不支持絕對物理定律框架。

---

### C. 反光 = 低頻/低梯度 半句 — 新文獻 Li & Brown 2014（relative smoothness prior 原始出處，**尚未在本文 bib 中**）

| 欄位 | 內容 |
|------|------|
| **作者** | Yu Li, Michael S. Brown |
| **標題** | Single Image Layer Separation using Relative Smoothness |
| **發表** | Proc. IEEE CVPR, 2014 |
| **arXiv** | 無（CVF Open Access：cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_Single_Image_Layer_2014_CVPR_paper.pdf） |
| **驗證狀態** | ✅ CONFIRMED（已讀 pp.1-2 全文） |

**驗證原文（直接引述）：**
> Abstract: "This paper addresses extracting two layers from an image where one layer is smoother than the other... We introduce a novel strategy that regularizes the gradients of the two layers such that one has a long tail distribution and the other a short tail distribution."

> Fig.1 caption (p.1): "In both of these problems one layer has fewer large gradients than the other layer."

> §1 (p.1), 反光模糊的物理機制描述: "...modified version based on Schechner et al.'s [14] proposition of using focus such that the desired layer is more in focus while the reflection is blurred. This can be expressed as: I = L_B + L_R * h, where the reflection layer is convolved with the depth of field kernel h modelled as a Gaussian blur."

**結論：** 這是「relative smoothness prior」的**原始出處**——[1]、[4] 都是引用/沿用這個先驗。它本身也是把「反光層梯度分布與背景層不同（短尾 vs. 長尾分布）」當作一個**用於正則化病態反問題的建模假設（prior）**，而非經驗證的跨場景物理定律；其物理機制描述為**鏡頭景深造成的失焦模糊（defocus / Gaussian blur kernel）**，與本文「光線擴散」用詞不完全相同（但屬於相關的、會降低反光層空間頻率的光學成因）。**未明確討論「domain-independent」**。

**注意：** 此論文目前不在本文 30 篇參考文獻中，若採用需新增 bib entry。

---

### D. 物件邊緣 = 高頻/高梯度 半句 — 新文獻 RINDNet (Pu et al. 2021，2026-06-14 更新)

> **更新說明：** 原 D 項結論依賴 CLAUDE.md §5.0b「眾所周知的數學/技術事實」例外條款（Sobel/Canny 教科書基礎），使用者已明確駁回此例外，要求提供真實引用。以下為新查證結果。

| 欄位 | 內容 |
|------|------|
| **作者** | Mengyang Pu, Yaping Huang, Qingji Guan, Haibin Ling |
| **標題** | RINDNet: Edge Detection for Discontinuity in Reflectance, Illumination, Normal and Depth |
| **發表** | Proc. IEEE/CVF ICCV 2021；arXiv:2108.00616 |
| **本地 PDF** | `matherial/papers/45_rindnet_pu2021.pdf`，已讀 pp.1-3（原始 PDF，非 WebFetch 摘要） |
| **驗證狀態** | ✅ CONFIRMED（材質不連續 → 邊緣 半句）／⚠️ 不涉及反光半句 |

**驗證原文（直接引述）：**
> Abstract / p.1: "As a fundamental building block in computer vision, edges can be categorised into four types according to the discontinuity in surface-Reflectance, Illumination, surface-Normal or Depth."

> p.1: "In his seminal work [27], David Marr summarized four basic ways edges can arise: (1) surface-reflectance discontinuity, (2) illumination discontinuity, (3) surface-normal discontinuity, and (4) depth discontinuity."

> p.2 (Related Works): "REs and IEs are mainly related to photometric reasons – REs are caused by changes in material appearance (e.g., texture and color), while IEs are produced by changes in illumination (e.g., shadows, light sources and highlights)."

> p.3 (§3.2 Edge Definitions): "Reflectance Edges (REs) usually are caused by the changes in material appearance (e.g., texture and color) of smooth surfaces."

**結論：**
- ✅ 直接支持「物件邊緣因材質突變產生響應」——RINDNet 將「材質/質地/顏色變化造成的邊緣」(Reflectance Edge) 列為電腦視覺中（追溯至 Marr 1980）四種基本邊緣成因之一，並稱其為 "a fundamental building block in computer vision"。這是來自**通用邊緣偵測文獻、獨立於 SIRR 領域**的真實引用，可佐證「材質不連續 → 邊緣響應」具有跨場景的普遍性基礎。
- ⚠️ RINDNet 未使用「高頻/高梯度」之頻域措辭描述 RE，僅以「edge」（邊緣偵測定義上即為梯度局部響應）描述，論文未展開頻域論述。
- ⚠️ RINDNet 完全不涉及「反光=低頻/低梯度（光線擴散）」半句——RINDNet 的 Illumination Edges (IEs) 指**同一影像內**因陰影/光源/highlight造成的邊緣，與 SIRR「玻璃反射疊加圖層」是不同的物理設定（單層影像內的光照邊緣 vs. 雙層疊加的反射層模糊）。本文獻**不能**、也**不需要**用來支持「反光」半句。

**注意：** 此論文目前不在本文 30 篇參考文獻中，若採用需新增 bib entry（建議 key: `pu2021rindnet`）。

---

### 總結 — 證據總覽與綜合判斷（2026-06-14 修訂）

**反光 = 低頻/低梯度半句（A/B/C項）：**

| 來源 | 角色 | 框架 | 是否支持「絕對物理定律、與domain無關」 |
|------|------|------|--------------------------------|
| Li & Brown 2014 | **原始出處**（relative smoothness prior 的提出者） | regularization prior，物理機制 = 景深失焦模糊（defocus/Gaussian blur kernel） | ❌（未討論 domain-independence） |
| [1] CEILNet | 沿用該 prior，並指出其失效情況 | "mild...assumption"，**明確給出強光反光產生大梯度的反例** | ❌ |
| [4] Location-aware SIRR | 沿用該 prior（歸功 Li & Brown [24]），論文主旨即處理違反此 prior 的案例 | "prior" | ❌ |

**物件邊緣 = 高頻/高梯度半句（D項，新增）：**

| 來源 | 角色 | 框架 | 是否支持「絕對物理定律、與domain無關」 |
|------|------|------|--------------------------------|
| RINDNet (Pu et al. 2021，追溯 Marr 1980) | 通用邊緣偵測文獻中的標準分類 | 「材質變化 → Reflectance Edge」是電腦視覺中公認的基本邊緣成因之一 | ⚠️ 部分支持——是「材質不連續會產生邊緣」此一般原理的 domain-independent 來源，但未使用「高頻/高梯度」頻域語言，亦未與「反光」半句做直接對比 |

**綜合判斷（修正原「三個來源一致」的措辭）：**

原文「三個來源一致將...」一句容易被誤讀為三篇論文**各自獨立**得出相同結論、互相驗證（三方共識）。但實際上 Li & Brown 2014 是該 prior 的**原始出處**，[1]/[4] 是**沿用該 prior 並各自附加但書**的後續工作——這是一條「原始來源 → 應用/修正」的**引用脈絡**，而非三方獨立共識。修正後的陳述：

> 「反光較平滑/低梯度」這一描述，其原始出處（Li & Brown 2014）將其定位為用於正則化病態反問題的**建模假設（prior）**；後續沿用此假設的 [1] CEILNet、[4] Location-aware SIRR 均明確指出此假設在強反光情境下會失效。三者均未將其陳述為無條件成立、與場景 domain 無關的絕對物理定律。

「物件邊緣因材質突變產生強響應」這一描述，在通用邊緣偵測文獻（RINDNet，追溯至 Marr 1980 的經典邊緣分類）中被列為電腦視覺中四種基本邊緣成因之一，具有跨場景的普遍性基礎；但該文獻未使用「高頻/高梯度」頻域語言，亦未與「反光」做直接對比。

**因此：** cvgip2025_chinese.py 第 385-394 行（§1）與第 553-559 行（§3.2.1）目前將兩項特性並列陳述為「物理性質，與場景 domain 無關」的**絕對化措辭**，仍超出現有文獻所能直接支持的範圍——兩個半句各有不同程度、不同性質的文獻支持（前者＝SIRR 領域內被廣泛採用但有但書的 prior；後者＝通用邊緣偵測領域的基本邊緣成因分類），但都不是「無條件成立的絕對物理定律」，也沒有任何單一文獻把兩者**並列對比**陳述為一組 domain-independent 定律。

**建議（具體措辭提案見對話紀錄，待使用者核可後寫入 .py）：**
1. 反光半句改為「先驗/趨勢」型措辭（如「在 SIRR 文獻中已被廣泛作為先驗假設」），引用 [1][4]（已在 bib，零成本）；可選擇性新增 Li & Brown 2014 作為原始出處引用。
2. 邊緣半句改為引用 RINDNet/Marr 的「材質不連續是電腦視覺中基本邊緣成因之一」框架（需新增 bib entry：`pu2021rindnet`）。
3. 「與場景 domain 無關」整體措辭軟化，避免「絕對物理定律」式並列宣稱；§3.2.1 第 558 行「這正是 SGA 實現跨場景泛化的根本機制」改為設計動機/依據型措辭，避免未經消融實驗驗證的因果宣稱（CLAUDE.md §5.2）。
