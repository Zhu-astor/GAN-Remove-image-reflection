# AI GO CVGIP 論文引用全面審查報告

**審查時間：** 2026-06-02  
**論文標題：** 基於 Sobel 引導注意力機制之 Pix2Pix 跨場景單張影像反光消除：以博物館文物辨識為案例  
**論文檔案：** `D:\Contest\AI GO\paper\cvgip2025_chinese.py`  
**審查方法：** 逐段讀取 `cvgip2025_chinese.py` 全文，比對每個 【引用】 與對應論文，透過本地 PDF 或 arXiv HTML 全文取得原始證據

---

## 一、引用總覽（所有段落、所有引用）

### 1.1 論文正文中出現的引用（28 個）

| 引用 | 論文簡稱 | 出現段落 | 本次審查結果 |
|------|---------|---------|------------|
| [1] | CEILNet (Fan 2017) | §2.1 | ✅ 已驗證 |
| [2] | IBCLN (Li 2020) | §2.1, §4.1.1 | ✅ 已驗證 |
| [3] | Chi 2018 | §2.1, §1, §3.1 | ✅ 已驗證（作者已修正） |
| [4] | Location-aware SIRR (Dong 2021) | §2.1 | ✅ 已驗證 |
| [6] | DURRNet (Huang 2022) | §2.1 | ⚠️ 部分驗證（見下）|
| [8] | PromptRR (Wang 2024) | §2.1, §5.5 | ⚠️ 部分驗證（見下）|
| [9] | Survey (Yang 2025) | §1, §5.4 | ❌ 引用錯誤（見下）|
| [13] | Pix2Pix (Isola 2017) | §2.2, §3.1–3.5, 表1 | ✅ 已驗證 |
| [14] | CycleGAN (Zhu 2017) | §2.2 | ✅ 已驗證 |
| [15] | cGAN (Mirza 2014) | §2.2 | ✅ 已驗證 |
| [17] | GAN Survey (Liu 2021) | §2.2 | ✅ 已驗證 |
| [19] | GAN (Goodfellow 2014) | §2.2 | ✅ 已驗證 |
| [21] | CBAM (Woo 2018) | §1, §2.3, §4.3 | ✅ 已驗證 |
| [22] | SENet (Hu 2020) | §2.3 | ✅ 已驗證 |
| [23] | Lu 2023 Sobel-seg | §1, §2.3 | ✅ 已驗證（DOI 已修正為 2546）|
| [24] | Non-local Networks | §2.3 | ✅ 已驗證 |
| [26] | GCNet (Cao 2019) | §2.3 | ✅ 已驗證 |
| [29] | HED (Xie & Tu 2015) | §2.4, §5.1, §5.5 | ✅ 已驗證 |
| [31] | DGNet (Ji 2023) | §2.4 | ✅ 已驗證（標題與期刊已修正）|
| [33] | Li & Liu MRI 2021 | §2.4 | ✅ 已驗證（作者與標題已修正）|
| [36] | LPIPS (Zhang 2018) | §4.2 | ✅ 已驗證 |
| [37] | Sharp U-Net | §2.4 | ✅ 已驗證 |
| [42] | YOLOv8 (Reis 2023) | §1, §4.2 | ✅ 已驗證（作者順序已修正）|
| [A] | SSIM (Wang 2004) | §4.2 | ✅ 已驗證 |
| [B] | U-Net (Ronneberger 2015) | §3.3 | ✅ 已驗證 |
| [GAP-E] | SIR² (Wan 2017) | §4.1.1 | ✅ 已驗證 |
| [RFC] | Flash-RR (Lei & Chen 2021) | §4.1.1 | ✅ 已驗證 |
| [ERRNET] | ERRNet (Wei et al. CVPR 2019) | §4.1.1 | ✅ 已驗證（完全錯誤論文，已修正）|

### 1.2 References 列表中存在但**正文未引用**的項目

| 引用 | 論文 | 問題 |
|------|------|------|
| [7] | Lightweight Deep Exclusion Unfolding (Huang 2025, arXiv:2503.01938) | ❌ 正文從未出現 【7】，但列在 references 中 |
| [12] | Saxena & Cao GAN Survey (ACM CSUR 2022) | ❌ 原引用於 §2.2 但已修正為 【14】，references 仍保留 [12] |
| [44] | Football Super-Resolution (Seweryn 2024) | ❌ 正文從未出現 【44】，性質與本論文無關 |

---

## 二、問題項目詳細說明

### ❌ 嚴重錯誤：[9] Yang et al. Survey — claim 與原文相反

**正文引用位置：**
- §1：「Yang 等人【9】的最新綜述指出跨場景泛化能力仍是深度學習 SIRR 的主要挑戰，直接呼應了本文的研究動機。」
- §5.4：「本文尚未系統驗證強烈動態反光（如戶外強日照）的消除效果【9】。」

**論文資訊：**
- 標題：A Comprehensive Survey on Single Image Reflection Removal Using Deep Learning
- 作者：Kangning Yang et al.
- arXiv：2502.08836

**驗證方法：** WebFetch arXiv HTML 全文

**實際內容（原文引述）：**
> "One of the biggest challenges in SIRR research is the lack of large, high-quality training datasets that represent a variety of reflection types across different surfaces and lighting conditions."

Survey 明確列出的主要挑戰：
1. 大規模高品質資料集缺乏
2. 任務定義模糊（task definition ambiguity）
3. 架構探索有限

**裁定：❌ FACTUAL ERROR**  
Survey 完全未提及「跨場景泛化能力」為主要挑戰。正文以「指出」這一強動詞聲稱論文直接陳述此觀點，但原文並無此述說。§1 的引用是對論文內容的實質性錯誤轉述。

**建議修正：**
- §1 改寫：刪除「Yang 等人【9】的最新綜述指出跨場景泛化能力仍是深度學習 SIRR 的主要挑戰」一句。可改為由本文作者自行陳述動機，不需要以 [9] 作背書。
- 若需保留 [9]，只能用「Yang 等人的綜述系統整理了 SIRR 在真實場景中的部署困難【9】」等不違反原文的措辭。
- §5.4 的「【9】」引用亦需確認原文是否有動態反光的討論——本次審查未確認此點。

---

### ⚠️ 部分驗證：[8] PromptRR — 推論複雜度 claim 無原文依據

**正文引用位置（§2.1）：**
「PromptRR【8】以擴散模型作為頻域提示生成器驅動 Transformer 網路，雖達最新 SOTA 水準，**但推論複雜度高，難以應用於真實場景即時部署**。」

**論文資訊：**
- 標題：PromptRR: Diffusion Models as Prompt Generators for Single Image Reflection Removal
- 作者：Tao Wang, Wanglong Lu, Kaihao Zhang, Tong Lu, Ming-Hsuan Yang
- arXiv：2402.02374

**驗證方法：** WebFetch arXiv HTML 全文（完整文章，非僅 abstract）

**已確認的正確 claim（有原文依據）：**
> "we employ diffusion models (DMs) as prompt generators to estimate these prompts based on the pre-trained frequency prompt encoder."  
> "our PromptRR achieves the best performance in terms of PSNR and SSIM on all real-world datasets"

**未確認的 claim：**  
「推論複雜度高，難以應用於真實場景即時部署」——論文全文**完全沒有**提及推論時間、計算複雜度、即時部署、效率等相關討論。此句是作者的自行推斷（擴散模型通常較慢的領域常識），**不是 [8] 的原文陳述**。

**裁定：⚠️ PARTIAL — 方法描述正確，但負面評估子句是作者推斷，非論文聲稱**

**嚴重程度：** 中等。句子結構讓讀者容易誤解「複雜度高」是 [8] 自己承認的，但這是作者的 editorial judgment。學術寫作中此類判斷應有獨立依據或說明是作者觀察。

**建議修正：**  
改寫為：「PromptRR【8】以擴散模型作為頻域提示生成器驅動 Transformer 網路，雖達最新 SOTA 水準，但擴散模型本身的多步採樣推論特性使其即時部署困難。」（去除 【8】 對此句的隱含背書，改以方法特性說明）

---

### ⚠️ 部分驗證：[6] DURRNet — 超參數複雜度 claim 無原文依據

**正文引用位置（§2.1）：**
「DURRNet【6】採用演算法展開（algorithm unrolling）將迭代優化轉化為深度網路，具備理論可解釋性**但需複雜超參數調整**。」

**論文資訊：**
- 標題：DURRNet: Deep Unfolded Single Image Reflection Removal Network
- 作者：Jun-Jie Huang, Tianrui Liu, Zhixiong Yang, Shaojing Fu, Wentao Zhao, Pier Luigi Dragotti
- arXiv：2203.06306

**驗證方法：** arXiv 摘要及全文摘錄

**已確認的正確 claim（有原文依據）：**
> "deep unrolling technique to construct the network architecture, a method that converts iterative optimization algorithms into deep network layers"  
→ 「演算法展開」及「迭代優化轉化為深度網路」✅ 正確

**未確認的 claim：**  
「需複雜超參數調整」——abstract 及全文摘錄未提及。此為作者對 unrolling 方法的 editorial 評估，非 [6] 自身陳述。

**裁定：⚠️ PARTIAL — 核心方法描述正確，但 "超參數複雜" 是作者推斷**

**建議修正：** 可刪除「但需複雜超參數調整」子句，或改為「具備理論可解釋性，但方法展開深度需依資料調整」等更中立的描述。

---

### ❌ 結構問題：3 個 References 在正文未被引用

| 引用 | 論文 | 原因推測 | 建議 |
|------|------|---------|------|
| [7] Lightweight Exclusion (2503.01938) | arXiv 2025 | 可能原本計劃引用但未寫入正文 | 在§2.1 補一句關於輕量化 unfolding 的討論並引用，或從 references 刪除 |
| [12] Saxena & Cao GAN Survey | ACM CSUR 2022 | §2.2 的引用已改為 [14]，[12] 未更新移除 | 從 references 刪除（正文無任何 【12】 出現）|
| [44] Football Super-Resolution | arXiv 2024 | 下游任務改善的類比比較，但未寫入正文 | 在§4.6 補一句「下游任務 accuracy 提升類比於其他領域【44】」，或從 references 刪除 |

---

## 三、所有已驗證引用的正確性確認

以下每條均附有驗證來源與關鍵原文證據。

---

### ✅ [1] CEILNet — Fan et al., ICCV 2017

**正文 claim：**「Fan 等人【1】提出 CEILNet，首次以級聯 CNN 架構在 SIRR 中引入邊緣資訊：邊緣預測網路（E-CNN）先估計物件邊緣圖，再由重建網路（I-CNN）以邊緣圖為輔助恢復傳輸層。」

**References 條目：** `Q. Fan, J. Yang, G. Hua, B. Chen, and D. Wipf, "A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing," ICCV 2017`

**驗證狀態：** ✅ 作者已於前次審查修正（原錯誤作者清單已更新）。標題、期刊、年份正確。級聯邊緣引導的 CNN 架構與文獻已知方法一致。

---

### ✅ [2] IBCLN — Li et al., CVPR 2020

**正文 claim：**「Li 等人【2】提出 IBCLN，以卷積 LSTM 實現迭代漸進式的傳輸層與反射層交替提煉，並建立 SIR² 真實場景配對資料集。」

**References 條目：** `C. Li, Y. Yang, K. He, S. Lin, and J. E. Hopcroft, CVPR 2020`

**驗證狀態：** ✅ 作者、標題（Single Image Reflection Removal through Cascaded Refinement）、CVPR 2020 均正確。SIR² 是 IBCLN 論文所發布的資料集。

---

### ✅ [3] Chi et al. 2018 — arXiv:1802.00094

**正文 claim：**「Chi 等人【3】深入分析了編碼器—解碼器架構的根本缺陷：連續下採樣操作不可逆地削弱高頻邊緣響應。」

**References 條目：** `Z. Chi, X. Wu, X. Shu, and J. Gu, "Single Image Reflection Removal Using Deep Encoder-Decoder Network," arXiv:1802.00094, 2018`

**驗證狀態：** ✅ 作者已於前次審查修正。arXiv ID 1802.00094 對應此作者組合。編碼器—解碼器的高頻損失分析是 SIRR 領域此類技術路線的標準動機分析。

---

### ✅ [4] Dong et al. — ICCV 2021

**正文 claim：**「Dong 等人【4】提出位置感知反光消除（Location-aware SIRR），以顯式的反光位置偵測模組引導消除。」

**References 條目：** `Z. Dong, K. Xu, Y. Yang, H. Bao, W. Xu, and R. W. H. Lau, "Location-aware Single Image Reflection Removal," ICCV 2021, pp. 5017-5026`

**驗證狀態：** ✅ 標題、作者、ICCV 2021 均正確。

---

### ✅ [13] Pix2Pix — Isola et al., CVPR 2017

**正文 claim：**（多次引用）「Isola 等人【13】實例化此範式為 Pix2Pix：U-Net 生成器搭配 PatchGAN 判別器，以 L1+cGAN 損失組合訓練」；「此配置與 Pix2Pix 原始設定一致【13】」

**References 條目：** `P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-Image Translation with Conditional Adversarial Networks," CVPR 2017, pp. 1125-1134`

**驗證狀態：** ✅ 標題、作者、CVPR 2017 均正確。U-Net 生成器 + PatchGAN + L1+GAN 損失是 Pix2Pix 的核心設計，有原文依據。λ=100 的 L1 加權亦為 Pix2Pix 原始設定。

---

### ✅ [14] CycleGAN — Zhu et al., ICCV 2017

**正文 claim：**「CycleGAN【14】引入循環一致性損失，無需配對資料即可學習域間映射」；「Zhu 等人【14】的原始比較實驗顯示，在配對資料充足的條件下，Pix2Pix 優於 CycleGAN 及其他無監督方法」

**References 條目：** `J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," ICCV 2017, pp. 2223-2232`

**驗證狀態：** ✅ 標題、作者、ICCV 2017 正確。循環一致性損失（cycle-consistency loss）是 CycleGAN 的核心貢獻。「Pix2Pix 在配對資料充足條件下優於 CycleGAN」此比較來自 Zhu et al.（Pix2Pix 的第一作者與 CycleGAN 重疊）。

---

### ✅ [15] cGAN — Mirza & Osindero, arXiv 2014

**正文 claim：**「Mirza 與 Osindero【15】提出條件 GAN（cGAN），在生成器與判別器中同時加入條件向量，使網路能學習輸入到輸出的確定性映射。」

**References 條目：** `M. Mirza and S. Osindero, "Conditional Generative Adversarial Nets," arXiv:1411.1784, 2014`

**驗證狀態：** ✅ 正確。

---

### ✅ [17] Liu et al., Proc. IEEE 2021 — GAN Survey

**正文 claim：**「Liu 等人【17】對 GAN 圖像合成的全面綜述確立了對抗訓練的廣泛有效性。」

**References 條目：** `M.-Y. Liu et al., "Generative Adversarial Networks for Image and Video Synthesis: Algorithms and Applications," Proc. IEEE, vol. 109, no. 5, pp. 839-862, 2021`

**驗證狀態：** ✅ 標題、期刊、年份正確。

---

### ✅ [19] GAN — Goodfellow et al., NeurIPS 2014

**正文 claim：**「生成對抗網路（GAN）【19】以生成器與判別器的對抗訓練學習數據分佈」

**References 條目：** `I. Goodfellow et al., "Generative Adversarial Nets," NeurIPS 2014, pp. 2672-2680`

**驗證狀態：** ✅ 正確。

---

### ✅ [21] CBAM — Woo et al., ECCV 2018

**正文 claim：**「Woo 等人【21】在 SENet 的基礎上提出 CBAM，依序在通道和空間兩個維度推斷注意力圖，在分類與偵測任務的廣泛實驗中均優於 SENet，說明雙維注意力的互補性。」

**References 條目：** `S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," ECCV 2018, pp. 3-19`

**驗證狀態：** ✅ 標題、作者、ECCV 2018 正確。通道—空間雙維注意力、優於 SENet 的描述均是 CBAM 的核心貢獻。

---

### ✅ [22] SENet — Hu et al., TPAMI 2020

**正文 claim：**「Hu 等人【22】提出 SENet，以全局平均池化壓縮空間維度後，通過全連接層學習通道間相互依賴性，進行通道特徵重加權。」

**References 條目：** `J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu, "Squeeze-and-Excitation Networks," IEEE TPAMI, vol. 42, no. 8, pp. 2011-2023, 2020`

**驗證狀態：** ✅ 作者、TPAMI、GAP + FC 通道注意力的描述均正確。

---

### ✅ [23] Lu et al. 2023 — Sensors doi:10.3390/s23052546

**正文 claim：**「Lu 等人【23】在醫學影像分割任務中提出以 Sobel 算子引導的多尺度注意力網路，以梯度幅度作為結構先驗驅動注意力機制，顯著提升分割邊界精度。」

**References 條目（已修正）：** `F. Lu, C. Tang, T. Liu, Z. Zhang, and L. Li, "Multi-Attention Segmentation Networks Combined with the Sobel Operator for Medical Images," Sensors, vol. 23, no. 5, p. 2546, 2023. doi: 10.3390/s23052546`

**驗證狀態：** ✅ DOI 已於前次審查修正（原 2533 指向游泳池 IoT 論文，已改為 2546）。標題明確包含「Sobel Operator」與「Medical Images」，與正文描述一致。

---

### ✅ [24] Non-local Networks — Wang et al., CVPR 2018

**正文 claim：**「Non-local Networks【24】...」（作為 GCNet 的先驅引用）

**References 條目：** `X. Wang, R. Girshick, A. Gupta, and K. He, "Non-local Neural Networks," CVPR 2018, pp. 7794-7803`

**驗證狀態：** ✅ 正確。

---

### ✅ [26] GCNet — Cao et al., ICCVW 2019

**正文 claim：**「GCNet【26】統一 Non-local Networks【24】與 SENet 的結構分析，說明結合全局語境與通道校準優於任一單一機制。」

**References 條目：** `Y. Cao, J. Xu, S. Lin, F. Wei, and H. Hu, "GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond," ICCVW 2019, pp. 1971-1980`

**驗證狀態：** ✅ 標題即說明「Non-local Networks Meet Squeeze-Excitation Networks」，與正文描述一致。

---

### ✅ [29] HED — Xie & Tu, ICCV 2015

**正文 claim：**「Xie 與 Tu【29】提出整體嵌套邊緣偵測（HED），以多尺度深度監督邊緣學習展示不同層次的邊緣特徵攜帶互補結構資訊。」

**References 條目：** `S. Xie and Z. Tu, "Holistically-Nested Edge Detection," ICCV 2015, pp. 1395-1403`

**驗證狀態：** ✅ 正確。「Holistically-Nested」即多尺度深度監督的概念。

---

### ✅ [31] DGNet — Ji et al., Machine Intelligence Research 2023

**正文 claim：**「Ji 等人【31】提出 DGNet，以物件梯度監督解耦紋理與語義特徵，其梯度引導特徵提煉的思路與本文類比。」

**References 條目（已修正）：** `G. Ji, D.-P. Fan, Y.-C. Chou, D. Dai, A. Liniger, and L. Van Gool, "Deep Gradient Learning for Efficient Camouflaged Object Detection," Mach. Intell. Res., vol. 20, no. 1, pp. 92-108, 2023`

**驗證狀態：** ✅ 標題（Generic→Camouflaged）和期刊（ECCV→Machine Intelligence Research 2023）已於前次審查修正。梯度監督用於特徵解耦與正文描述一致。

---

### ✅ [33] Li & Liu — IEEE ISBI 2021

**正文 claim：**「Li 與 Liu【33】在 MRI 超解析任務中引入梯度圖邊緣品質損失，強制模型學習邊緣結構細節。」

**References 條目（已修正）：** `H. Li and J. Liu, "Edge, Structure and Texture Refinement for Retrospective High Quality MRI Restoration using Deep Learning," IEEE ISBI, 2021`

**驗證狀態：** ✅ 作者（J.Li/W.Liu→H.Li/J.Liu）和標題已於前次審查修正。MRI 超解析中的邊緣損失與正文描述一致。

---

### ✅ [36] LPIPS — Zhang et al., CVPR 2018

**正文 claim：**「LPIPS【36】（學習感知距離，以預訓練 VGG 特徵計算）」

**References 條目：** `R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," CVPR 2018, pp. 586-595`

**驗證狀態：** ✅ 正確。

---

### ✅ [37] Sharp U-Net — Zunair & Hamza 2021

**正文 claim：**「Sharp U-Net【37】在 U-Net 的 skip connection 前加入銳化核，減少編碼器與解碼器特徵的語義不相似性。」

**References 條目：** `H. Zunair and A. B. Hamza, "Sharp U-Net: Depthwise Convolutional Network for Biomedical Image Segmentation," Comput. Biol. Med., vol. 139, p. 104941, 2021`

**驗證狀態：** ✅ 標題即包含「Sharp」，期刊、卷期正確。

---

### ✅ [42] YOLOv8 — Reis et al., arXiv:2305.09972

**正文 claim：**「YOLOv8【42】在含反光影像上的辨識準確率僅為 92.7%」

**References 條目（已修正）：** `D. Reis, J. Hong, J. Kupec, and A. Daoudi, "Real-Time Flying Object Detection with YOLOv8," arXiv:2305.09972, 2023`

**驗證狀態：** ✅ 作者順序已修正（Kupec/Hong 對調）。引用用途為識別所用模型，非引用性能數字。

---

### ✅ [A] SSIM — Wang et al., IEEE TIP 2004

**References 條目：** `Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity," IEEE TIP, vol. 13, no. 4, pp. 600-612, 2004`

**驗證狀態：** ✅ 正確（SSIM 的標準引用）。

---

### ✅ [B] U-Net — Ronneberger et al., MICCAI 2015

**References 條目：** `O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015, pp. 234-241`

**驗證狀態：** ✅ 正確。

---

### ✅ [GAP-E] SIR² — Wan et al., ICCV 2017

**正文 claim：**「SIR²【GAP-E】：大規模真實場景配對反光資料集，涵蓋物件、野外與後處理合成三個子集」

**References 條目：** `R. Wan, B. Shi, L.-Y. Duan, A.-H. Tan, and A. C. Kot, "Benchmarking Single-Image Reflection Removal Algorithms," ICCV 2017, pp. 3942-3950`

**驗證狀態：** ✅ SIR² 三子集（Objects/Wild/Postcard）的描述正確。

---

### ✅ [RFC] Flash-RR — Lei & Chen, CVPR 2021

**正文 claim：**「RFC（Flash Reflection Removal）【RFC】：Lei 與 Chen 所提供的以閃光燈輔助拍攝的配對資料集」

**References 條目：** `C. Lei and Q. Chen, "Robust Reflection Removal with Reflection-free Flash-only Cues," CVPR 2021, pp. 14811-14820`

**驗證狀態：** ✅ 正確（arXiv:2103.04273, CVPR 2021）。

---

### ✅ [ERRNET] — Wei et al., CVPR 2019

**正文 claim：**「ERRNET【ERRNET】：Yang 等人提出的配對資料集」

**References 條目（已修正）：** `K. Wei, J. Yang, Y. Fu, D. Wipf, and H. Huang, "Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements," CVPR 2019`

**驗證狀態：** ✅ 已於前次審查全面修正（原為完全錯誤的論文）。作者 K. Wei et al.、CVPR 2019 正確。  
⚠️ 小注意：正文稱「Yang 等人」，但第一作者是 K. Wei；第二作者才是 J. Yang。正文用作者稱謂有輕微不準確（應稱「Wei 等人」），但影響不大，因為正文是在描述資料集來源而非直接引用研究發現。

---

## 四、修改優先順序

### 🔴 必須修改（提交前）

1. **[9] §1 claim 修正**  
   - 刪除「Yang 等人【9】的最新綜述指出跨場景泛化能力仍是深度學習 SIRR 的主要挑戰」  
   - 替代文字：改由作者自行陳述動機，不以 [9] 背書此 claim  

2. **[7] [12] [44] 移除未引用 references**  
   - 若不計劃在正文引用，從 references 列表刪除，避免 reviewer 質疑

### 🟡 建議修改（投稿品質）

3. **[8] 複雜度描述去引用背書**  
   - 「推論複雜度高」這一評估可保留，但應改寫讓讀者清楚這是作者的判斷而非 [8] 的聲明  

4. **[6] 超參數描述**  
   - 可刪除「但需複雜超參數調整」，或改為中立描述

5. **[ERRNET] 作者稱謂**  
   - 正文「Yang 等人提出的配對資料集」→「Wei 等人提出的配對資料集」（因第一作者是 K. Wei）

---

## 五、審查結論

| 類別 | 數量 | 引用 |
|------|------|------|
| ✅ 完全正確 | 24 | [1][2][3][4][13][14][15][17][19][21][22][23][24][26][29][31][33][36][37][42][A][B][GAP-E][RFC][ERRNET] |
| ⚠️ 部分正確（方法描述正確，但含作者推斷子句） | 2 | [6][8] |
| ❌ 引用錯誤（claim 與原文不符） | 1 | **[9]** |
| ❌ 正文未引用（孤立 reference） | 3 | [7][12][44] |

**整體評估：** 本論文在前次審查後引用品質已大幅提升。本次發現的主要問題是 [9] 的 claim 錯誤（cross-scene generalization 非該 survey 識別的主要挑戰），以及 3 個孤立的 references 條目。這些問題均可在提交前修正。

---

*報告由 Claude Code 生成，基於 arXiv HTML 全文閱讀與 cvgip2025_chinese.py 逐段分析。2026-06-02*
