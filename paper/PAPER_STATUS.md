# PAPER_STATUS.md — AI GO CVGIP 論文狀態追蹤

Last updated: 2026-06-20（本次 session續6：Fig.5換成Original/Baseline/SGA三排對比圖、縮排房規廢除全文統一、v14新版本誕生，詳見文件末新章節）（前次同日續3：C1/C2引用查證+修正、D2/D3過度confirmatory斷言修正、全文可讀性重寫、v13新版本誕生）（前次同日續2：§4.3反光嚴重程度分層實證分析 + 512px解析度排除實驗 + v12新版本誕生（中英文）+ 修正Table1/2英文版整表未翻譯bug）（前次同日：Fig.1/Fig.2 v2 重畫 + 英文版v11縮排bug修正並已套用至docx）（前次 2026-06-17：內容精簡3輪 + template 間距修正 + OMML 公式 + Baseline YOLO 重訓實驗 + 統一 v11，最新交付為 `D:\Download\cvgip2025_SGA_chinese_v11.docx` / `cvgip2025_SGA_english_v11.docx`）（前次 2026-06-15 續3：§4.5/§4.6 一致性修正）（內容縮減第一輪：已試做並修正存檔方式，現為
**兩份獨立檔案並存**——
(1) `cvgip2025_SGA_chinese.docx/.pdf`＝原版（FIG-1修正後、縮減前，9頁，
未變動）；
(2) `cvgip2025_SGA_chinese_reduced.docx/.pdf`＝內容縮減試驗版（套用
A1-A3/B1/B3/B4/C/D1/D2，9頁，第9頁References幾乎填滿、接近8頁臨界點）。
產生方式：`cvgip2025_chinese.py`（原版，已還原9處編輯）與
`cvgip2025_chinese_reduced.py`（縮減版，含9處編輯+輸出路徑改為
`_reduced.docx`）+ 新增 `docx_to_pdf_reduced.py`（輸出`_reduced.pdf`）。
未套用：B2（cut Liu et al[15]需citation重新編號）、Tier E（FIG-4/FIG-8圖片
數量精簡），待使用者決定方向；詳見本文件最新章節。
舊版（FIG-1修正、§4.6補充、[31]RINDNet查證、A1-A4過度推論修正）摘要省略，
見git歷史）

---

## 基本資訊

- **論文標題**：基於 Sobel 引導注意力機制之 Pix2Pix 跨場景單張影像反光消除：以博物館文物辨識為案例
- **投稿目標**：CVGIP 2025（台灣國內研討會）
- **主要語言**：中文版（主線）
- **主要成果**：AI GO 2024 競賽最佳實作獎
- **生成指令**：`C:\Users\bubbl\anaconda3\python.exe cvgip2025_chinese.py`

---

## 現有檔案版本鏈

| 檔案 | 說明 | 狀態 |
|------|------|------|
| `cvgip2025_chinese.py` | 中文版 docx 生成腳本（主線） | ✅ v5 整合使用者編輯 + 引用重新編號[1]-[30]（2026-06-14） |
| `cvgip2025_SGA_chinese.docx` | 上述腳本生成的 docx | ✅ 已生成，9 頁，符合官方範本版面規格 |
| `cvgip2025_reflection_removal.py` | 英文版腳本 | ⚠️ 舊版，尚未同步 v2 修改 |
| `cvgip2025_SGA_reflection_removal.docx` | 英文版 docx | ⚠️ 舊版 |

---

## 核心技術架構（不可改動）

- **模型**：Pix2Pix（U-Net Generator + PatchGAN Discriminator）
- **創新**：Sobel-Guided Attention（SGA）模組，插入於 U-Net 第一個 Encoder 塊之前
- **訓練資料**：SIR²、IBCLN、ERRNET、RFC 四個公開資料集合併後**隨機分割**
  - SIR²：sir2data.github.io
  - IBCLN：github.com/JHL-HUST/IBCLN
  - ERRNET：github.com/Vandermode/ERRNet（[ERRNET] 引用需確認，見 references）
  - RFC：github.com/ChenyangLEI/flash-reflection-removal（閃光燈輔助配對）
  - 訓練集：810 對，測試集：248 對（隨機抽樣，非按資料集劃分）
- **博物館評估**：699 張影像（下游辨識用，跨 domain 設計，**未用於訓練**）——現存最接近的資料夾為 `GAN_Test/Dataset/Test/jpg`（651 張），原 699 張證據可能在另一台電腦（使用者確認不需追查）
- **下游驗證**：92.7% → 94.5%（+1.8pp）✅ **確認沿用**（2026-06-13 逆轉方案 A）。來源：`反光處理論文準備.pdf` §6（699 張測試集，原始 92.7%，40 張失敗中 10 張在去反光版成功 → 94.5%，人工計數，權威版本）+ `FY113-機器學習了沒實證成果簡報v2.pptx` slide 11（514/554→524/554，同組數字另一呈現，不寫入正文）
- **訓練設定**：RTX 4090，**256×256**（§3.6 誤寫為 512×512，待修正），batch 8，epoch 500，Adam lr=5e-5，epoch 400 checkpoint

---

## 論文敘事定位（v2 已修正）

**核心故事（Cross-Domain Generalization via Structural Prior）**：
> 以公開 SIRR 資料集（自然場景）訓練，博物館藏品完全未出現在訓練資料中。
> 加入固定 Sobel 核引導的注意力（SGA）後，模型能對訓練時從未見過的博物館藏品
> 產生有效的反光消除效果，提升下游 YOLOv8 辨識準確率。
> 關鍵在於固定 Sobel 梯度的 domain-agnostic 特性：反光低梯度、邊緣高梯度
> 這兩個物理特性與場景 domain 無關，使注意力機制能跨 domain 遷移。

---

## ✅ 已修正

**v2（2026-06-01）**
- §4.1 資料集章節：完整重寫為跨 domain 設計說明
- Abstract：移除「本文建立博物館場景配對資料集」之誤述
- 移除無法重新訓練的 CEILNet/IBCLN 比較列
- 損失函數：補充完整的 Ladv 與 LL1 數學公式

**v3（2026-06-02）敘事框架重構**
- 論文定位：從「博物館應用論文」改為「跨場景 SIRR 方法論論文，以博物館為案例」
- 標題：加入「跨場景」並改為「以博物館文物辨識為案例」副標
- Abstract：首句改為「監督式 SIRR 的根本瓶頸在於配對資料難以取得」，以研究痛點開場
- §1 Introduction：重構段落順序——①SIRR訓練困境（通用痛點）→②博物館為極端案例→③研究問題→④Pix2Pix+注意力→⑤Sobel先驗假設→貢獻
- 貢獻（2）：改為「以固定結構先驗實現跨場景 SIRR 的方法論」
- §4.1：改標題為「實驗設計」，明確說明跨 domain 研究框架
- §4.2：加入「兩層次評估策略」說明
- §4.6：標題改為「跨場景下游辨識結果」，強調零微調部署
- §5：增加「方法論的適用範圍」子節（工廠/建築/文件等其他場景）
- §6 Conclusion：改以「監督式 SIRR 的配對資料困境」開場，強化方法論貢獻

---

## ⏳ 空白待填（提交前必填）

| 標記 | 位置 | 內容 | 狀態 |
|------|------|------|------|
| MUST-1 | 表 1 | Baseline Pix2Pix 的 PSNR / SSIM / LPIPS（公開 SIRR 測試集，491 對） | ✅ 23.896 / 0.8706 / 0.1630 |
| MUST-2 | 表 1 | Pix2Pix+SGA 的 PSNR / SSIM / LPIPS（公開 SIRR 測試集，491 對） | ✅ 22.682 / 0.8192 / 0.2178 |
| MUST-3 | 表 2 | ~~下游驗證數據（Baseline Pix2Pix 列）~~ | ✅ 已解除（2026-06-13）：新敘事將 Table 2 簡化為「原始影像 92.7%」與「Pix2Pix+SGA 94.5%」兩列，不再需要 Baseline Pix2Pix 下游數值 |
| OPT-1 | 表 1 | CA only 的 PSNR/SSIM/LPIPS（有 checkpoint 才填，沒有刪整行） | ✅ 使用者確認不需要（2026-06-13，無對應 checkpoint，表1維持2列） |
| OPT-2 | 表 1 | SA only 的 PSNR/SSIM/LPIPS（有 checkpoint 才填，沒有刪整行） | ✅ 使用者確認不需要（2026-06-13，無對應 checkpoint，表1維持2列） |

---

## ✅ 圖表清單（共 8 張，全部已插入；2026-06-13 依文件出現順序全面重新編號為圖1~圖8）

> 重新編號原因：原「圖3」（visual_comparison.png）之前還有 §4.1.1（compare_original1/generate1.jpg）
> 與 §4.3（show.png）兩組圖，但當時未編號，導致編號順序錯亂。本次依出現順序連續編號，
> 原圖3→圖5、圖4→圖6、圖5→圖7、圖6→圖8，並為前述兩組圖新增圖3、圖4。
> 對應的 caption 與正文引用已於 `cvgip2025_chinese.py` 全部同步修改並重新產出 docx/pdf，
> PDF 逐頁檢查 9 頁，圖號與引用一致。

| 編號 | 檔名 | 內容 | 狀態（2026-06-13 更新） |
|------|------|------|------|
| 圖1 | overall_architecture.png | 整體架構圖（輸入→SGA→U-Net→輸出+PatchGAN+Loss） | ✅ `matherial/overall_architecture.png`（依程式碼與 §3.1/3.5 公式繪製，300dpi；腳本 `matherial/draw_overall_architecture.py`；已插入 §3.1，PDF 第 3 頁視覺驗證通過） |
| 圖2 | sga_module_architecture.png | SGA 模組詳細結構 | ✅ `matherial/sga_module_architecture.png`（依程式碼繪製，300dpi；腳本 `matherial/draw_sga_architecture.py`） |
| 圖3 | compare_original1.jpg + compare_generate1.jpg | 訓練資料集樣本範例（§4.1.1，上：Original，下：Generated） | ✅ 已插入，本次新增「圖3.」編號（原無編號） |
| 圖4 | show.png | 訓練 domain 反光消除效果範例（§4.3，3 組 Original/Generated 對比） | ✅ 已插入，本次新增「圖4.」編號（原無編號） |
| 圖5 | visual_comparison.png | Before/After 博物館視覺比較（5 組展品，§4.4） | ✅ `matherial/visual_comparison.png`（原圖3，編號改為圖5；圖檔本身未變更） |
| 圖6 | reflection_sobel_feature.png + nonreflection_sobel_feature.png | Sobel 梯度幅度視覺化（§4.4） | ✅ `matherial/reflection_sobel_feature.png` + `nonreflection_sobel_feature.png`（原圖4，編號改為圖6） |
| 圖7 | loss_function.png | G loss / D loss 訓練曲線（§4.5） | ✅ `matherial/loss_function.png`（原圖5，編號改為圖7；G/D loss，x 軸為 iteration 0~70000+） |
| 圖8 | `原跑原7.jpg`（上）/ `消跑原7.jpg`（下） | YOLOv8 偵測信心值對比圖（原始 vs SGA 處理後，bbox+confidence，§4.6） | ✅ 已插入（原圖6，編號改為圖8），方向已確認正確：`原跑原7.jpg`=0.76/0.48/0.64 與畫面較霧；`消跑原7.jpg`=0.93/0.85/0.81 與畫面較清晰，與 caption 數字及敘事完全吻合 |

> `compare_original1.jpg`/`compare_generate1.jpg`（§4.1.1）與 `show.png`（§4.3）為 2026-06-13 v2 新增
> 之補充圖，當時未編號；本次重新編號後已分別納入圖3、圖4（見上表）。

---

## 未來工作（已寫入 §5.4）

1. 可學習邊緣偵測器（HED）補充固定 Sobel 核，研究跨 domain 影響
2. 建立多博物館含反光/無反光配對資料集（燈光開關控制取得）
3. SGA 整合至 Transformer 架構（PromptRR）
4. 半監督/對比學習框架降低配對資料依賴

---

## 引用索引驗證狀態（cite-papers 第二次全面審查完成，2026-06-02）

**完整報告：** `D:\Contest\AI GO\paper\citation_audit_report.md`

### ❌ 第一次審查已修正的嚴重錯誤（7 項）

| 引用 | 問題 | 修正內容 |
|------|------|---------|
| [1] CEILNet | 作者完全錯誤 | Q. Fan, J. Yang, G. Hua, B. Chen, D. Wipf |
| [3] Chi 2018 | 作者完全錯誤 | Z. Chi, X. Wu, X. Shu, J. Gu |
| [12] Saxena Cao | arXiv ID 錯誤（2112.12625）→ 2005.00065；期刊改為 ACM Comput. Surv. 2022 |
| [23] Lu 2023 | **DOI 10.3390/s23052533 指向游泳池 IoT 論文！** 改為文章 2546，標題/作者均修正 |
| [31] DGNet | 標題 "Generic" → "Camouflaged"；期刊 ECCV 2022 → Machine Intelligence Research 2023 |
| [33] Li Liu | 作者 "J. Li/W. Liu" → "H. Li/J. Liu"；標題修正為 MRI Restoration |
| [ERRNET] | 完全錯誤論文；改為 K. Wei et al., CVPR 2019, arXiv:1904.00637 |

### 第一次審查已修正次要錯誤（1 項）

| 引用 | 問題 | 修正 |
|------|------|------|
| [42] YOLOv8 | J. Kupec/J. Hong 作者順序對調 | D. Reis, J. Hong, J. Kupec, A. Daoudi |

### 已修正 body text claim（第一次審查）
- §2.2：原「Saxena 等人【12】系統比較顯示 Pix2Pix 優於 CycleGAN」→改為「Zhu 等人【14】的原始比較顯示」

---

### ✅ 第二次審查問題 — 全部已修正（2026-06-02）

| 引用 | 問題 | 修正動作 |
|------|------|---------|
| **[9] §1 claim** | ❌ 「指出跨場景泛化能力仍是主要挑戰」但 Survey 未說此話 | §2.1 刪除該句，改為作者自身陳述（不引用 [9] 背書）|
| [6] §2.1 | ⚠️「需複雜超參數調整」無原文依據 | 刪除此子句 |
| [8] §2.1 | ⚠️「推論複雜度高，難以應用於真實場景即時部署」無原文依據 | 改寫為「擴散模型本身的多步採樣特性使即時部署的計算代價較高」（移除隱含引用背書）|
| [6] reference | ❌ 作者 "C.-H. Huang, J.-L. Wu, Y.-C. F. Wang" 與 arXiv:2203.06306 不符 | 改為 `J.-J. Huang, T. Liu, Z. Yang, S. Fu, W. Zhao, P. L. Dragotti` |
| [8] reference | ❌ 作者 "T. Wang, J. Li, K. He, Y. Liu" 與 arXiv:2402.02374 不符 | 改為 `T. Wang, W. Lu, K. Zhang, T. Lu, M.-H. Yang` |
| [9] reference | ❌ 第一作者 "Z. Yang" 應為 "K. Yang"（Kangning） | 改為 `K. Yang et al.` |
| [7] | ❌ 正文未引用 | 從 references 刪除 |
| [12] | ❌ 正文未引用（已改引 [14]） | 從 references 刪除 |
| [44] | ❌ 正文未引用 | 從 references 刪除 |
| [ERRNET] 正文 | ⚠️「Yang 等人提出」（第一作者是 K. Wei） | 改為「Wei 等人提出」|

### ✅ 審查後當前狀態：所有引用均已驗證

| 類別 | 數量 | 引用 |
|------|------|------|
| ✅ 完全正確 | 28 | 所有正文引用（[1][2][3][4][6][8][9][13][14][15][17][19][21][22][23][24][26][29][31][33][36][37][42][A][B][GAP-E][RFC][ERRNET]）|
| ❌ 已刪除（孤立 ref） | 3 | [7][12][44] — 已從 references 移除 |

**完整報告位置：** `D:\Contest\AI GO\paper\citation_audit_report.md`

---

## 2026-06-06 新增工作記錄

### 量化評估（eval_metrics.py 完成）
- Baseline Pix2Pix (ep300, 256px): PSNR 23.896 / SSIM 0.8706 / LPIPS 0.1630（491 對）
- Pix2Pix+SGA (ep400, 256px): PSNR 22.682 / SSIM 0.8192 / LPIPS 0.2178（491 對）
- SGA 三項指標均低於 Baseline → 需用 Perception-Distortion Tradeoff 解釋
- SGA-512 ep360 指標最佳（PSNR 22.828 / SSIM 0.8417）但仍低於 Baseline

### citation_verification_record.md 建立完成
- 位置：`D:\Contest\AI GO\paper\citation_verification_record.md`
- 已收錄：全部 28 條現有引用 + [Blau18] + [Ledig17] 兩條新引用（共 30 條）
- 新引用 [Blau18]、[Ledig17]：✅ CONFIRMED（ar5iv 全文已讀，含直接原句）
- 現有引用：⚠️ PARTIAL（arXiv 摘要已讀，方向確認正確）
- 已知問題備忘表：§4.3 宣稱錯誤、§3.6 解析度錯誤均記錄在案

### 已套用至 cvgip2025_chinese.py（2026-06-06）
- §3.6：「512×512」→「256×256」✅
- §4.3：刪除「SGA 三項指標均優於 Baseline」，改為 Perception-Distortion Tradeoff 說明 ✅
- §4.3 intro：刪除無 checkpoint 的 CA only / SA only 變體描述 ✅
- 加入 [Blau18] 和 [Ledig17] 到 references 列表 ✅
- 表 1 數據更新（測試集 491 對，MUST-1/2 已填）✅
- cvgip2025_SGA_chinese.docx 已重新生成 ✅

---

## 2026-06-06 下游評估腳本建立

### eval_downstream.py（新增）
- 路徑：`D:\Contest\AI GO\paper\eval_downstream.py`
- 執行環境：`python389`（TF 2.6.0 + ultralytics 8.2.94）
- 功能：4 個 phase 完整評估所有組合
  - Phase 1：GAN 預處理（Baseline + SGA × Test/Reflection + Test/jpg → 暫存資料夾）
  - Phase 2：有標籤 val split mAP50/precision/recall（CLASS 模型 × config.yaml，DATASETS 模型 × museum_data.yaml）
  - Phase 3：無標籤 predict（detection_rate + avg_top1_conf）
  - Phase 4：配對類別一致率（Test/Reflection vs Test/NonReflection 作為偽標籤）
- 輸出：`eval_downstream_results.csv`、`eval_downstream_per_image.csv`、`eval_downstream_summary.txt`
- 執行指令：
  ```
  cd "D:\Contest\AI GO\paper"
  C:\Users\bubbl\Desktop\Virtualenv\python389\Scripts\python.exe eval_downstream.py
  ```

### 評估架構說明
- 699 張：未找到確切對應資料夾，測試集涵蓋範圍如下：
  - `Test/Reflection`（198）：配對下游評估主力（Phase 4 一致率 ≈ 92.7%/94.5% 的最接近代理）
  - `Test/jpg`（651）：SSIM 評估用測試集，也跑一遍
  - `val_split`（374 for CLASS，139 for DATASETS）：有標籤 mAP50 評估
- 使用 NonReflection 預測作為偽 ground-truth 計算一致率，無法用真實 label 計算 accuracy 的原因：Test/Reflection 無 YOLO 格式 label 檔

---

## 2026-06-08 eval_downstream.py 執行結果（先前未記錄）

執行於 2026-06-08 11:18，輸出三檔皆在 paper/ 下。重點數據：

**Phase 2（有標籤 val mAP50）**：CLASS_train3 = 0.0021（**已壞，不可用**）、CLASS_train8 = 0.9950、DATASETS_train2 = 0.9030、DATASETS_train3 = 0.9542

**Phase 3（Test_Reflection 198 張，DetRate raw→Baseline→SGA）**：
- CLASS_train8：0.9596 → 0.9697 → **0.9848（SGA 為正向）**
- DATASETS_train2：0.0657 → 0.1212 → 0.1566（正向但基數過低）
- DATASETS_train3：0.1667 → 0.1667 → 0.1414（負向；domain mismatch，見下）

**Phase 4（配對一致率，NonReflection 為偽 GT，raw→Baseline→SGA）**：四個模型全部負向（CLASS_train8: 0.7486→0.6175→0.5137；DATASETS_train3: 0.7333→0.5333→0.3333）

⚠️ summary 檔尾註 "raw≈92.7%, SGA_GAN≈94.5%" 與任何實跑結果**皆不符**，該註解為撰寫腳本時的錯誤預期，不可引用。

---

## 2026-06-10 工作記錄（本次 session）

### 92.7%/94.5% 來源調查 — 已結案
- **出處**：競賽簡報 `FY113-機器學習了沒實證成果簡報v2.pptx` slide 11（514/554=92.7%，+10 張救回 → 524/554=94.5%）+ 使用者 PDF `反光處理論文準備.pdf` §6（權威版本：同一辨識模型、699 張測試集、raw 92.7%，40 張失敗中 10 張在去反光版成功 → 94.5%，**人工計數**）
- **結論**：無任何現存程式碼/輸出可重現此數字；原始證據可能在另一台電腦，使用者確認不需追查
- **決策（使用者核准方案 A，2026-06-10）**：論文以新的可重現數據**完全取代** 92.7/94.5，不保留舊數字
- **⚠️ 2026-06-13 更新：此決策已逆轉**——使用者確認 92.7/94.5 為人工篩選的真實數據，明確指示沿用；詳見下方「2026-06-13 工作記錄」

### CLASS_train8 vs DATASETS_train3 深度比較 — 已完成
- CLASS_train8（`Classification/runs/detect/train8`）：訓練資料為**黑底裁切展品圖**（datasets/datasets，train 2560 / valid 374，nc=8 含 other），300 epochs，val mAP50=0.995（同質性高，泛化未真正檢驗）；診斷檔不全（無 confusion matrix / PR curve）；**但在 651/198 張真實測試集上 DetRate ≥ 0.95，是唯一可用模型**
- DATASETS_train3（`museum_data_annotation.v9i.yolov8/runs/detect/train3`）：訓練資料為**真實博物館場景照**（train 415 / valid 139 / test 210，nc=7），1200 epochs（~220 收斂），mAP50=0.954，診斷檔完整（混淆矩陣對角線強，主要錯誤 obj3↔背景 11 次）；**但對 Test_Reflection DetRate 僅 0.17 — 測試集 domain 對不上，不適合當主要證據**
- 數量對應：PDF 的「2000 訓練 / 699 測試」較接近 CLASS 系列（2560/651）而非 museum 系列（415/210）→ 當年競賽模型較可能是 CLASS 系列

### 下游驗證新方案 — 已定案並建腳本
**⚠️ 2026-06-13 更新：此方案（CLASS_train8 × 651 張 → 98.62%/98.77%）已依使用者指示棄用，不用於論文** —— 使用者確認「之前98.62 98.77都是從錯誤的模型判斷閾值、條件 錯誤模型出來的」，請丟棄。以下為原始記錄，僅供歷史參考：
- **腳本**：`paper/eval_class_train8_651.py`（v1.0.0，2026-06-10）
- **組合**：CLASS_train8 × `GAN_Test/Dataset/Test/jpg`（651 張）× {raw, Baseline_GAN, SGA_GAN}
- GAN 處理影像直接重用 `paper/_downstream_temp/{baseline_jpg,sga_jpg}`（各 651 張，2026-06-08 產生），**不需 TensorFlow**
- 主指標：DetRate + ObjRate（top-1 非 other）+ AvgConf；另輸出「救回案例」清單（raw 失敗→GAN 成功），對齊 PDF §6 原始方法論但全自動可重現
- 執行：`C:\Users\bubbl\Desktop\Virtualenv\python389\Scripts\python.exe eval_class_train8_651.py`
- 輸出：`eval_class_train8_651_per_image.csv` + `eval_class_train8_651_summary.txt`

### 誠實呈現原則（寫入 §4.6 時必守）
**⚠️ 2026-06-13 更新：新敘事（視覺證據優先 + 92.7/94.5）不使用 eval_downstream.py 的 Phase 3/4 數據，以下原則暫不適用，僅供歷史參考**：
- Phase 3 DetRate/AvgConf 對 SGA 正向，Phase 4 配對一致率對 SGA 負向——兩者都要呈現，負向結果沿用 §4.3 的 Perception-Distortion Tradeoff [Blau18] 框架解釋
- 訓練 domain 證據（`compare_original1.jpg`/`compare_generate1.jpg`、`show.png`，皆為公開 SIRR 資料）可證明「訓練時完全未見博物館資料」的跨 domain 敘事

---

## 2026-06-13 工作記錄（本次 session）

### 「漏判」超集資料夾 / 反光消除訓練辨識模型調查 — 已告一段落
- 依使用者指示，建立 `paper/_rescue_check/run_reflex_models.py` 與 `identify_missed_annotator.py`，測試 reflex2000.pt / no_reflex2000.pt 及其他 8 個 checkpoint 對 `Classification/Compare/` 10 組「漏判」影像的偵測結果
- **關鍵發現**：`GAN_Test/reflex2000.pt` 在 `Compare/img-N.jpg`（"rescued" 版本）上的偵測結果幾乎精確重現該圖上已標註的信心值（例如 img-100 obj3=0.54、img-283 obj7=0.84、img-316 obj3=0.82），但對全部 10 張 `漏判objX-img-N.jpg` 均輸出 `(none)`
- 視覺檢查發現先前假設方向錯誤：`漏判objX-img-N.jpg`（較模糊）並非乾淨原圖，`img-N.jpg`（較清晰，已有標註框）視覺上更接近 `Test_Data` 的 raw699
- 未找到更大的「漏判」超集資料夾。使用者表示「我不確定 不過這邊先告一段落」——本調查暫停，不影響 §4.6 改寫工作

### 92.7/94.5 方案 A — 逆轉，恢復沿用
- 使用者明確指示（2026-06-12）：「92.7/94.5 我沒有放棄...請使用該組」「之前98.62 98.77...請丟棄這部分」
- **Task #1 完成**：重新讀取 `反光處理論文準備.pdf`（pages 1-10），verified §6 原文：
  > 「我使用國立歷史博物館的展品數據集(七個展品，2000 張訓練集，699 張測試試集)，訓練一個展品辨識模型，透過對比原照片與反光去除後照片的辨識率來做下游驗證。其中原始測試集的辨識率是 92.7%，其中 40 張沒有成功辨識的照片在去反光測試集中有 10 張成功辨識。去反光測試集的辨識率是 94.5%，辨識率提升了 1.8%。」
- **重要發現**：檢查 `cvgip2025_chinese.py` 現況後，發現 92.7%/94.5%/699 張的寫法**從未被方案 A 實際取代**——8 處引用（行 182, 209, 249, 499, 514, 591-592, 689）目前皆已是正確數字。**因此這 8 處不需要任何數字改動**，方案 A 的逆轉純粹是「取消一個尚未執行的計畫」，無需回滾程式碼

### §4.6 新敘事：視覺證據優先 + 量化結果脈絡化
- 依使用者指示（「我們不跟pix2pix(without sga)的數值、下游辨識率做比較，而是秀出最直接證據：圖片比較...然後最後數據只有92.7->94.5 但是有很多說明、引用論證...」），完成 §4.6 與 §5.4 改寫草案
- **草案位置**：`D:\Contest\AI GO\paper\SECTION_4_6_REWRITE_PROPOSAL_2026-06-13.md`（狀態：待使用者審閱，尚未套用）
- **核心變更**：
  1. §4.6 開場改為引用圖 3（視覺比較）+ 新增 FIG-6（YOLO 偵測信心值對比圖，取代原 bar chart 設計）
  2. 92.7%→94.5%（+1.8pp）維持，新增「40 張失敗中 10 張（25%）救回」描述（直接衍生自已驗證的 PDF §6 原文，純算術）
  3. 新增脈絡化討論段落：以「92.7%+5.7pp理論上限=100%」與「699張/7類評估集規模」解釋提升幅度有限的原因，不依賴未驗證外部主張
  4. Table 2 移除「Baseline Pix2Pix」列與 `[MUST-3]` 佔位符（不再需要 `eval_class_train8_651.py` 或任何額外評估）
- **Task #4（引用研究）結論**：草案論點均可由「已驗證的 PDF §6 數字」+「純算術」+「常識性規模比較」支撐，**無需新增外部引用**；若使用者要為「高基準準確率限制可量測增益」之一般化現象補充文獻，可另行透過 `/cite-papers` 搜尋

### Task #2 — FIG 素材需求 ✅ 使用者已提供（2026-06-13），已套用

---

## 2026-06-13 v2 完成記錄

使用者提供 `matherial/` 新素材並指示「可以開始更新 新版本的論文了 v2」，已完成以下變更於 `cvgip2025_chinese.py`：

1. **§4.1.1 新增**：`compare_original1.jpg` + `compare_generate1.jpg`（訓練資料集樣本，建築+植栽場景，證明訓練資料與博物館評估場景無 domain 重疊）。
2. **§4.3 末新增**：`show.png`（訓練 domain 上 3 組 Original/Generated 反光消除效果範例，呼應 Perception-Distortion Tradeoff 討論）。
3. **§4.6 全面改寫**（依 `SECTION_4_6_REWRITE_PROPOSAL_2026-06-13.md`）：
   - 標題改為「4.6. 跨場景下游效益：視覺證據與辨識準確率」
   - 開場引用圖 3（§4.4）視覺證據，再以 FIG-6 呈現 YOLOv8 信心值對比
   - 量化段落：92.7%→94.5%（+1.8pp），699 張中 40 張失敗、10 張（25%）救回
   - 新增脈絡化討論段（算術上界 5.7pp、高基準準確率、699張/7類規模偏小）
   - 表 2 簡化為 2 列（原始影像 92.7% / Pix2Pix+SGA 94.5%），移除 `[MUST-3]` 與 Baseline Pix2Pix 列 → **MUST-3 已解除**
   - FIG-6 = `原跑原7.jpg`（上）+ `消跑原7.jpg`（下）
4. **§5.4 新增第四點限制**：說明 +1.8pp 幅度受評估集規模（699張/7類，僅40張原始失敗）與辨識模型高基準準確率（92.7%）共同制約，呼應未來工作方向。
5. 更新檔頭 TODO 清單（MUST-3/FIG-2~6 標記為已完成）。
6. 執行 `cvgip2025_chinese.py` → `cvgip2025_SGA_chinese.docx`；新增 `docx_to_pdf.py`（Word COM 自動化）→ `cvgip2025_SGA_chinese.pdf`（8 頁）。已逐頁檢查 PDF，所有新圖均正確渲染，無 `[fig] missing` 警告。

### ✅ FIG-6（`原跑原7.jpg` / `消跑原7.jpg`）方向已確認

上一 session 的記錄（`SECTION_4_6_REWRITE_PROPOSAL_2026-06-13.md` §2、本檔案 Task #2 舊版）寫的是「`消跑原7.jpg`（未經 SGA，信心值較低）/ `原跑原7.jpg`（經 SGA，信心值較高）」。

**本次直接開啟兩張圖片重新檢視後，發現實際內容方向相反**：
- `消跑原7.jpg`：畫面較清晰，信心值 **0.93 / 0.85 / 0.81**（較高）
- `原跑原7.jpg`：畫面略帶反光/霧感，信心值 **0.76 / 0.48 / 0.64**（較低）

本次 v2 採用的方向判斷依據（**非使用者逐一確認，為本次推論**）：
1. 檔名語意：「消跑原7」=「消[反光]+跑+原7」→ SGA 處理後輸出；「原跑原7」=「原[始]+跑+原7」→ 原始輸入
2. 視覺：`消跑原7.jpg` 反光/霧感明顯較少，與「反光消除後」的敘事一致
3. 信心值方向與全文 92.7%→94.5%（反光消除提升辨識）的敘事一致

因此 v2 docx/pdf 中 FIG-6 為：**上＝`原跑原7.jpg`（原始含反光，信心值 0.76/0.48/0.64）／下＝`消跑原7.jpg`（SGA處理後，信心值 0.93/0.85/0.81）**。

**若此方向判斷錯誤**（即兩檔案的「原始/SGA後」標籤與本文相反），只需在 `cvgip2025_chinese.py` 的 §4.6 對調 `fig(doc, '原跑原7.jpg')` 與 `fig(doc, '消跑原7.jpg')` 兩行的順序、並同步修改正文敘述中 0.76→0.93 等數字的方向即可，其餘內容不受影響。

**結論（2026-06-13，使用者確認）**：上述方向判斷正確，FIG-6 不需調整。`原跑原7.jpg`=0.76/0.48/0.64（畫面較霧）對應上方「原始」、`消跑原7.jpg`=0.93/0.85/0.81（畫面較清晰）對應下方「SGA處理後」，與 caption 數字及全文 92.7%→94.5% 敘事完全吻合。

---

## 2026-06-13 全面審查記錄（不足盤點 + 引用全面複核）

使用者指示「幫我分析與盤點該篇論文的不足，另外請再次全面檢查論文引用是否正確，記得寫
引用處所、引用內容、引用證據的檔案下來」。完整審查結果見新檔
**`PAPER_AUDIT_2026-06-13.md`**（與本檔同目錄）。本次為唯讀分析，**未修改** `cvgip2025_chinese.py`，
所有建議修正均為待使用者核准之提案。

### 不足盤點重點
1. 標題頁佔位符（lines 165-167）：學生姓名/指導教授/系所/學校/Email 均未填。
2. **FIG-1 缺失 + 懸空引用**：line 342「如圖1所示」但 line 350 僅有 caption 佔位文字、
   無對應 `fig()` 呼叫；`matherial/` 中亦無 `overall_architecture.png`。
3. 致謝佔位符（line 735）：`【博物館/合作單位名稱】` 未填（疑似應為「國立歷史博物館」）。
4. **「810對/248對」vs「491對」數字不一致 — 已查明根因**：
   - 實測 `D:\Contest\AI GO\github\Dataset2\Train\Reflection` = **1951** 個檔案，
     `Dataset2\Test\Reflection`/`NonReflection` = **491/491** 個檔案。
   - 1951/(1951+491)=79.9%≈80%，491/2442=20.1%≈20% → §4.1.1 的「80%/20%隨機分割」**框架正確**，
     但具體數字「810對/248對」（lines 486-487, 518, 679, 727）與實際 `Dataset2` 不符；
     「491對」（lines 528, 534，即本表1 MUST-1/2 數據來源 `eval_metrics.py` 的 `n_total`）
     與實際資料夾及既有 MUST-1/2 記錄一致。
   - **建議修正**：810→1951、248→491（4處），「80%/20%」描述不需更動。
5. line 417 `【13, B】` 括號風格與全文不一致（純格式，低優先）。

### 引用全面複核重點（47 處標記，24 個 bibkey 逐一核對）
- 約 39 處與既有 `citation_verification_record.md` 記錄方向一致，本次未發現新問題。
- **6 個議題需處理**（詳見 `PAPER_AUDIT_2026-06-13.md` §3）：
  1. 【高】[2](IBCLN)/[GAP-E](Wan2017) **SIR² 資料集歸屬疑似錯置**：line 263 將「建立SIR²資料集」
     歸功於 [2]，但 [GAP-E] 驗證記錄顯示 SIR² 實際出自 Wan et al. 2017（"first captured...SIR2"），
     早於 IBCLN(2020)。建議移除 line 263 的「SIR²」具體命名。
  2. 【高】[3](Chi 2018) "編碼器-解碼器根本缺陷：下採樣不可逆損失高頻邊緣"（lines 265, 348）
     ——此為 SGA 插入位置的核心設計論證，但 [3] 僅有摘要級⚠️PARTIAL驗證，未見對應 bullet。
     需讀全文驗證或軟化措辭。
  3. 【高】[23](Lu et al.) line 312-316 "此設計的成功**驗證了**...跨場景設定下的穩定性"
     ——[23] 本身未做跨domain實驗，此為作者類比推論卻用「驗證了」包裝成事實，建議改為
     推論措辭（"為...假設提供類比性支持"）。
  4. 【中】[9](SIRR Survey 2025) 兩處：(a) line 202 "往往大幅退化【9】" 摘要未直接支撐此具體
     方向；(b) line 692 引用[9]支撐「本文尚未驗證」邏輯不通，應改為支撐「此為已知挑戰」。
  5. 【中】[14](CycleGAN) line 290-291 "Zhu等人的原始比較實驗顯示Pix2Pix優於CycleGAN"
     ——廣為人知但歸因到具體論文實驗，建議補充驗證。
  6. 【低】[33]/[29] 已用「進一步支持」/「可能」等緩和措辭，風險低，可不處理。

---

## 2026-06-13 引用全文覆盤驗證（cite-papers，第三次審查）— 已完成

使用者指示「重新確認這些引用是否正確，必須從原文所有內容（不能只是開頭摘要），
全面覆盤與嚴格檢查，記得整理所有引用的來源、引用內容等成一份檔案，
務必確保所有引用不是空穴來風而是真有內容引用」。

**完整報告：** `D:\Contest\AI GO\paper\CITATION_FULLTEXT_VERIFICATION_2026-06-13.md`（全新檔案，兩批 + 補充項，共 28 個 bibkey / 27 個表格列）

### 範圍與方法
- 針對 `PAPER_AUDIT_2026-06-13.md` §3 列出的 6 項高/中風險問題（第一批）+ 其餘 20 個 bibkey（第二批）+ 補充項 [36] LPIPS，
  實際讀取本地 PDF 多個章節（非僅摘要）或 WebFetch 全文（[23] 經 PMC 全文）。
- 基礎性引用（[15][17][19][36][A][B] 等）依 CLAUDE.md §5.0b 例外條款，確認摘要層級已足夠。
- **覆蓋率**：`cvgip2025_chinese.py` 全部 30 個 reference 條目中，本檔案涵蓋 28 個；
  餘 2 個（[Blau18]、[Ledig17]）已於 2026-06-06 session 以 ar5iv 全文驗證過（程式碼第 801-804 行註解），不重複驗證。
  **至此 30/30 個 reference 條目皆已完成全文層級驗證**。

### 結果總表（28 個 bibkey，含補充項 [36]）
- **23 個 ✅ CONFIRMED**（含新增 [36] LPIPS），無需修改。
- **5 個項目的「建議修正」提案，已於 2026-06-13 經使用者核准並套用至 `cvgip2025_chinese.py`**（依 §5.2 流程：提案 → 核准 → 套用）：
  1. **[2] IBCLN**（line 263-264）：SIR² 資料集歸屬錯置（SIR² 實際出自 [GAP-E] Wan et al. 2017，非 [2]）→ ✅ 已改為「具密集標註 ground truth 的真實場景配對資料集」
  2. **[3] Encoder-Decoder**（lines 265-267, 348）：「深入分析...不可逆地削弱高頻邊緣響應」過度引申 → ✅ 已改為保守措辭（「下採樣操作所帶來的資訊損失會增加解碼器復原難度」，呼應 [3] 自身的 skip connection 設計）
  3. **[9] SIRR Survey**（line 692）：「本文尚未驗證...【9】」屬自身限制聲明 → ✅ 已移除【9】標記
  4. **[23] SMA-Net**（lines 312-316）：「驗證了固定 Sobel 梯度在跨場景設定下的穩定性」——[23] 全文（PMC）證實為單一 COVID-19 CT domain、無任何跨資料集/跨場景測試 → ✅ 已改寫，明確劃清「[23]已驗證」vs「本文自行驗證（§4.4-4.6）」的界線
  5. **[4] Location-aware SIRR**（line 267-269）：「證明空間注意力在 SIRR 任務中的有效性」——原文用語是「reflection detection module / reflection confidence map」而非「spatial attention」→ ✅ 已改為「顯式空間位置線索」

  套用後已重新執行 `C:\Users\bubbl\anaconda3\python.exe cvgip2025_chinese.py` 產出 `cvgip2025_SGA_chinese.docx`，無錯誤。`citation_verification_record.md` 中 [2][3][4][9][23] 狀態同步更新為 ✅ CONFIRMED。

- **0 個項目為「空穴來風」**（捏造/無內容支持）——上述 5 項均屬「措辭過度引申/技術名詞誤用/引用位置誤掛」，非引用內容完全捏造。

### 額外正面發現
- **[13] Pix2Pix**：全文閱讀（§3.2 + Fig.4）證實原論文本身明確討論並圖示展示 L1 loss 的 over-smooth 現象，**解除** `citation_verification_record.md` 中先前「不應直接宣稱 Isola et al. 批評 L1 loss」的保留附注。

### citation_verification_record.md 同步更新
- [13]：附注更新為「保留意見已解除」。
- [14][15][17][19]：狀態由 ⚠️PARTIAL 升級為 ✅ CONFIRMED（[14] 經全文 Table 2/3 比對；[15][17][19] 依 §5.0b 確認摘要已足夠）。

---

## 2026-06-13 三線表修正 + FIG-1 整體架構圖 — 已完成

使用者指示「表格怪怪的 請製作專業論文表格」，再指示「fig1幫我製作」。

### 三線表（professional three-line table）修正
- **根因**：原表 1/表 2 以純文字字串（`|`、`─` 字元 + 空白對齊）放入 `caption()` 段落，
  Times New Roman 為比例字型，空白填充無法對齊欄位，呈現「怪怪的」錯位外觀。
- **修正**：在 `cvgip2025_chinese.py` 新增 `_set_cell_border()`（透過 raw OOXML `w:tcBorders`
  設定每格邊框，python-docx 無高階 API）與 `add_table(doc, caption_text, headers, rows,
  col_widths_in=None)`，實作標準三線表樣式（粗上線、細表頭分隔線、粗下線、無垂直線、
  表頭粗體置中）。表 1（§4.2，491對消融結果）、表 2（§4.6，跨場景辨識準確率）均改為
  真正的 `doc.add_table()` 物件。已重新產生 PDF，逐頁檢視確認表格渲染正確、置中、
  三線邊框正常，無錯位。

### FIG-1 整體架構圖
- 新增腳本 `matherial/draw_overall_architecture.py`（matplotlib，與 `draw_sga_architecture.py`
  同色票/同風格），繪製整體訓練/推論流程：
  `Input x → SGA Module(→x', 詳見圖2) → U-Net Generator G(Encoder×7/Decoder×7, skip, tanh)
  → Output T̂`（推論路徑，上排）；訓練專用虛線框內含 `Ground Truth y`、
  `PatchGAN Discriminator D`、`L_L1 (MAE, λ=100)`、`L_adv (MSE)`、
  `L_total = L_adv + λ·L_L1`，notation 對應 §3.1/§3.5。
- 輸出 `matherial/overall_architecture.png`（2617×2046px, 300dpi，aspect≈1.28:1）。
- `cvgip2025_chinese.py` line ~437：以 `fig(doc, 'overall_architecture.png')` +
  正式中文圖說取代原本的 `[FIG-1 — 請插入...]` 佔位文字；檔頭 TODO 清單與結尾
  `print()` 摘要均已標記 FIG-1 為 DONE。
- 已重新執行產生 `cvgip2025_SGA_chinese.docx`/`.pdf`，PDF 第 3 頁視覺確認：
  FIG-1 位於 §3.1 段落後、FIG-2 之前，圖內文字（含 LaTeX 風格數學符號）清晰可讀，
  圖說無 placeholder 殘留。

---

## 下一步行動（優先順序，2026-06-14 更新）

1. ~~v2 編輯 + docx/pdf 產出~~ ✅ 完成（2026-06-13）
2. ~~全面不足盤點 + 引用複核（摘要層級）~~ ✅ 完成（2026-06-13，結果見 `PAPER_AUDIT_2026-06-13.md`）
3. ~~引用全文覆盤驗證（28 bibkey，30/30 涵蓋率）~~ ✅ 完成（2026-06-13，結果見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md`，見上）
4. ~~引用措辭修正提案共 5 項（[2][3][9][23][4]）~~ ✅ 已套用（2026-06-13，使用者核准，見上）
5. ~~FIG-6 `原跑原7.jpg`/`消跑原7.jpg` 方向確認~~ ✅ 完成（2026-06-13，使用者確認方向正確，見上方說明）
6. ~~官方範本格式重排（雙section版面/欄距bug修正/真實作者單位Email/h3樣式）~~ ✅ 完成（2026-06-14，見下方新章節）。
   `標題頁資訊` 已解決；**剩餘待使用者決策** — `PAPER_AUDIT_2026-06-13.md` §4 其餘事項
   （致謝、810/248→1951/491、line 202【9】措辭、line 417【13, B】括號格式）
7. 推 GitHub 並回連結（依 feedback 規則）— 本次重排完成後待推送
8. ~~確認是否有 CA only / SA only checkpoint（OPT-1/2）~~ ✅ 完成（2026-06-13，使用者確認不需要，無對應 checkpoint，表1維持2列）
9. ~~FIG-1 整體架構圖補 SGA 位置（GAN_architecture.png 目前僅 GAN 迴圈）~~ ✅ 完成（2026-06-13，見下方記錄）
10. ~~圖號全面重新編號（圖3/圖4 原無編號）~~ ✅ 完成（2026-06-13，圖1~圖8 依出現順序連續編號，見上方圖表清單）

---

## 2026-06-14 官方範本格式重排 — 已完成

使用者提供 CVGIP-2026 官方 Word 範本（`.doc`），指示「目前請根據該篇格式套用一篇新的格式論文出來，
之前格式是錯誤的」。透過 `convert_template.py`（Word COM）轉出 `template_src.docx`/`.pdf`，
再以 `inspect_template.py`/`inspect_template2.py` 讀取 raw section XML 與樣式定義取得精確規格，
對 `cvgip2025_chinese.py` 完成下列重排，重新產出 `cvgip2025_SGA_chinese.docx`/`.pdf`（10 頁），
並逐頁視覺驗證通過。

### 版面架構改為雙 section
- **Section 0（標題區）**：單欄，margins top/bottom/left/right = 35/30/19/19mm
- **Section 1（正文）**：`add_section(WD_SECTION_START.CONTINUOUS)` 連續分節，雙欄，
  margins = 25/30/19/22.9mm，欄間距 8.01mm（`w:space="454"`，與範本一致）

### `set_two_col()` 欄間距單位換算 bug 修正
- **根因**：舊公式 `int(spacing_mm * 914400 / 25.4 / 1000)` 算出 EMU/1000（≈36×mm），但
  OOXML `<w:cols w:space>` 單位實為「二十分之一點」（twips，≈56.69×mm）——`spacing_mm=8`
  舊公式只產出 `w:space="288"`（≈5.08mm）。**先前所有版本（含 commit 617b0ca）的雙欄間距
  皆比設計值窄約 3mm**
- **修正**：`round(spacing_mm * 72 / 25.4 * 20)`，預設 `spacing_mm=8.01` 精確對應範本
  `w:space="454"`

### 作者/單位/Email — 真實資訊已填入（解決 PAPER_AUDIT 項目1：標題頁資訊）
標題頁佔位符已替換為：
- `¹Zi-Xian Zhuang (莊子賢), ¹,*Jiann-Shu Lee (李建樹)`（上標單位編號；英文姓名斜體、
  中文名非斜體，依範本規格）
- `¹ Department of Computer Science and Information Engineering, National University of
  Tainan, Tainan City, Taiwan`
- `E-mail: s11159030@gm2.nutn.edu.tw`

⚠️ **待使用者確認（CLAUDE.md §2.3）**：「Department of Computer Science and Information
Engineering」「National University of Tainan」為翻譯推斷，非官方英文名稱查證結果——
投稿前請核對國立台南大學資訊工程學系的官方英文系名/校名是否一致。

### 新增 `h3()` 樣式，套用至 5 個 sub-subheading
範本 H3（如「5.1.1. Sub-subheadings」）規格為：兩端對齊、不粗體、**斜體**、10pt。
新增 `h3()` helper 並套用至：3.2.1 Sobel 特徵萃取、3.2.2 通道注意力分支、
3.2.3 空間注意力分支、4.1.1 訓練資料集（公開 SIRR 資料集）、4.1.2 案例驗證：博物館藏品評估集。

### 其他範本對齊細項（使用者核准「一併套用」）
- `caption()`：9pt → 10pt（範本 Body Text Indent 3 規格）
- `fig()`/`add_table()`：圖/表寬度 3.2in → 3.1in（配合新欄寬與 8.01mm 欄間距）
- `h2()`：對齊 LEFT → JUSTIFY（範本 H2 為兩端對齊）
- 一般段落首行縮排：`Cm(0.5)` → `Inches(0.25)`（範本 body indent = 228600 EMU = 0.25in）
- `ref()`：改為懸掛縮排 `Emu(168275)`/`Emu(-168275)`，字級 9pt（範本 References 規格）
- Keywords 行：`Keywords：`（全角冒號）→ `Keywords: `（半角冒號+空格），字型改為 `Times`

### 結果
- `cvgip2025_SGA_chinese.docx`/`.pdf` 已重新產出，**PDF 由 9 頁變為 10 頁**（原因：標題區
  獨立 section 佔用空間 + 欄間距修正為正確的 8.01mm 後文字重排；第10頁僅為 References 溢頁）
- PDF 10 頁逐頁視覺確認：標題頁（單欄置中）、作者/單位/Email 區塊、Abstract 起雙欄、
  Keywords 半角冒號+Times 字體、H1/H2/H3 三層標題樣式、表1/表2三線表、圖1-8 全部正確渲染

---

## 2026-06-14 整合使用者手動編輯 + 引用全面重新編號 — 已完成

使用者下載 `cvgip2025_SGA_chinese.docx` 後手動編輯，產生 `D:\Download\cvgip2025_SGA_chinese (4).docx`，
透過 `diff_docx_edits.py` 比對出與腳本產出版本的 6 處差異，使用者回覆「1.是刻意 3.重新編號」——
即 Fig.4 圖片重複序列為**刻意設計**，且引用全面重新編號採用完整 [1]-[30] 方案（非僅命名引用）。
全部 6 處差異 + 重新編號已整合回 `cvgip2025_chinese.py`。

### A. AI GO 2024 競賽獎項提及 — 全面移除（3 處）
- Abstract 第（3）項：刪除「並獲 AI GO 2024 競賽最佳實作獎肯定」子句
- Conclusion：刪除「系統性地」一詞 + 刪除「並獲 AI GO 2024 競賽最佳實作獎肯定」子句
- **Acknowledgement 整節刪除**（原內容感謝「【博物館/合作單位名稱】」並提及競賽評審委員會最佳實作獎）

### B. 新增 `fig_row()` helper（插入於 `fig()` 之後、`ref()` 之前）
於單一置中段落中插入多張行內圖片，由 Word 依雙欄欄寬自動換行排版成網格；缺檔處理與 `fig()`
一致（`print(f"  [fig_row] missing, skipped: {path}")`）。簽名：
`fig_row(doc, filenames: list[str], width_in: float = 0.95)`。

### C. 圖4（§4.3）— 改為 7 圖序列（**使用者確認為刻意設計**）
- 舊：`fig(doc, 'show.png')` 單圖，caption 含「3 組 Original/Generated 對比」
- 新：`fig_row(doc, ['show1.png','show2.png','show3.png','show1.png','show4.png','show5.png','show3.png'])`
  （即 image5,6,7,5,8,9,7 序列，show1/show3 刻意各重複出現一次）
- caption 移除「3 組」字樣
- 新增素材：`matherial/show1.png`~`show5.png`（取自使用者編輯版 docx 的 `word/media/image5~9.png`）

### D. 圖5（§4.4）— caption 拆段
- 舊：單段 caption，含「（5 組展品）」
- 新：拆為兩段——「圖 5. 博物館藏品跨場景視覺比較。」/「上排：含反光原始影像（Original）；
  下排：Pix2Pix+SGA 反光消除結果（Generated）。」；圖片本身（`visual_comparison.png`）未變更

### E. 圖8（§4.6）— 改為每排 5 圖（原為每排 1 圖）
- 舊：`fig(doc, '原跑原7.jpg')` + `fig(doc, '消跑原7.jpg')`，caption 含「三個物件的辨識信心值分別由
  0.76、0.48、0.64 提升至 0.93、0.85、0.81」
- 新：`fig_row(doc, ['原跑原7_1.jpg'..'原跑原7_5.jpg'])`（上排）+
  `fig_row(doc, ['消跑原7_1.jpg'..'消跑原7_5.jpg'])`（下排）；caption 移除信心值數字句
- 新增素材：`matherial/原跑原7_1~5.jpg`、`消跑原7_1~5.jpg`（取自使用者編輯版 docx 的
  `word/media/image14-23.jpeg`；`_5` 兩檔與舊版 `原跑原7.jpg`/`消跑原7.jpg` byte-identical）

### F. 引用全面重新編號：[1]-[42]+7個命名引用（共 30 條，混合編號含跳號）→ 連續 [1]-[30]
透過一次性腳本 `renumber_citations.py` 執行：
- 依「正文首次出現順序」建立 30 條 `MAP`（舊 key → 新 1-30）
- 正文 `【...】` 標記：46 處全部重寫成功（含多引用 `【13, B】`→`【13】【23】` 拆分為相鄰兩個方括號）
- references 區塊：30 條 entry 全部依新編號重排；3 條 `# [N] REMOVED: ...` 純註解行
  （舊 [7]/[12]/[44]，原本就未被正文引用，僅為歷史跳號記錄）**直接刪除**

**新→舊→論文對照**：[1]=K.Yang SIRR Survey([9])｜[2]=YOLOv8([42])｜[3]=Lu SMA-Net([23])｜
[4]=CBAM([21])｜[5]=Chi2018([3])｜[6]=CEILNet([1])｜[7]=IBCLN([2])｜[8]=Location-aware SIRR([4])｜
[9]=DURRNet([6])｜[10]=PromptRR([8])｜[11]=GAN([19])｜[12]=cGAN([15])｜[13]=Pix2Pix([13])｜
[14]=CycleGAN([14])｜[15]=GAN Survey([17])｜[16]=SENet([22])｜[17]=GCNet([26])｜
[18]=Non-local NN([24])｜[19]=HED([29])｜[20]=DGNet([31])｜[21]=Sharp U-Net([37])｜
[22]=Li&Liu MRI([33])｜[23]=U-Net([B])｜[24]=SIR²/GAP-E([GAP-E])｜[25]=ERRNet([ERRNET])｜
[26]=RFC Flash Reflection([RFC])｜[27]=SSIM([A])｜[28]=LPIPS([36])｜
[29]=Perception-Distortion Tradeoff([Blau18])｜[30]=SRGAN([Ledig17])

`citation_verification_record.md`：於檔頭（`## 使用說明` 之前）新增「## 〇、引用編號對照表
（2026-06-14 全面重新編號）」，完整新→舊對照表。**檔案其餘 ~800 行（各 `### [KEY]` 小節、
bibkey 欄位、§6 討論）維持舊編號不變**——任務「edge=high-gradient 補引用」與「§6 過度推論修正」
範圍未受影響，仍待後續處理。

### 重新產出 + 視覺驗證
- 執行 `cvgip2025_chinese.py` → `cvgip2025_SGA_chinese.docx`（無 `[fig]`/`[fig_row] missing` 警告）
- 執行 `docx_to_pdf.py` → `cvgip2025_SGA_chinese.pdf`，**由 10 頁變為 9 頁**（Acknowledgement
  刪除 + 文字精簡 + 圖版面變化所致）
- `Read(pdf, pages="1-10")` 逐頁視覺確認 9 頁全部正確：A-F 全部變更均正確渲染，圖1-8 完整

### ✅ 已解決：§4.6 數字範例對應說明（2026-06-14）
§4.6 body 文字補上「以圖 8 中第 5 組範例為例」，使「三個物件的辨識信心值分別由 0.76、0.48、0.64
提升至 0.93、0.85、0.81」明確對應圖8現行 5+5 圖版面中的第5組（`原跑原7_5.jpg`/`消跑原7_5.jpg`）。
已重新產出 docx/pdf 並用 `Read(pdf, pages="7-8")` 視覺確認正確渲染。

### 待辦
- `renumber_citations.py` + `cvgip2025_chinese.py.bak_before_renumber`：一次性腳本/備份，
  確認無誤後可刪除
- 推送至 GitHub（`github/paper/`）並回連結

---

## 2026-06-14（續）任務 #13/#14：核心物理宣稱引用補強 — 已完成

### 背景
§1（第385-394行）與 §3.2.1（第553-559行）的核心物理假設原文：
「反光因光線擴散呈現低頻、低梯度特性，物件邊緣因材質突變呈現高頻、高梯度響應；
這兩項區別特性是物理性質，與場景 domain 無關。」
經 `citation_verification_record.md` §6 查證，現有引用（[6]CEILNet/[8]Location-aware SIRR/
新查Li&Brown2014）僅支持「反光=低梯度」為 **SIRR 文獻中的先驗假設/一般趨勢**（非絕對定律），
且「邊緣=高梯度」半句原僅靠 CLAUDE.md §5.0b「教科書事實」例外（已被使用者駁回，要求真實引用）。

### Task #13：找到「邊緣=高梯度」真實引用 — RINDNet (Pu et al., ICCV 2021)
- WebSearch → 下載 PDF (`matherial/papers/45_rindnet_pu2021.pdf`) → `Read` pp.1-3 直接讀取原文
- 關鍵驗證引文："Reflectance Edges (REs) usually are caused by the changes in material
  appearance (e.g., texture and color) of smooth surfaces."（p.3）；追溯至 Marr (1980)
  四種基本邊緣分類，稱為 "a fundamental building block in computer vision"（p.1）
- 屬通用邊緣偵測文獻（非SIRR），可佐證「材質不連續→邊緣響應」的跨場景普遍性，
  但未涉及「反光=低梯度」半句（IE的highlight框架與SIRR的玻璃反射疊加層是不同物理設定）
- 已記錄於 `citation_verification_record.md` §6 新 D 項

### Task #14：修正過度推論措辭
- `citation_verification_record.md` §6「總結」重寫：原「三個來源一致...」改為說明
  Li&Brown2014 是原始出處、[6]/[8] 是沿用並各自附加但書的後續工作（引用脈絡而非三方共識）；
  整合 RINDNet 發現後，結論維持「兩個半句各有支持但無單一文獻將其並列陳述為domain-independent
  絕對定律」
- §1（385-394行）與 §3.2.1（553-559行）改寫為分別引用 [6][8]（反光=先驗假設）與
  [31] RINDNet（邊緣=電腦視覺公認基本邊緣成因之一），移除「物理性質、與場景domain無關」
  絕對化措辭；§3.2.1 第558行「這正是 SGA 實現跨場景泛化的根本機制」改為
  「為 SGA 的注意力設計提供了依據」（移除未經消融驗證的因果宣稱，符合 CLAUDE.md §5.2）
- 新增 bib entry [31]：M. Pu, Y. Huang, Q. Guan, and H. Ling, "RINDNet: Edge Detection for
  Discontinuity in Reflectance, Illumination, Normal and Depth," in Proc. IEEE/CVF ICCV,
  2021, pp. 6879-6888.（30→31篇參考文獻，已驗證頁碼）

### 重新產出 + 視覺驗證
- `cvgip2025_chinese.py` → docx → pdf，仍為 9 頁
- `Read(pdf, pages="1-3")` 確認 §1/§3.2.1 改寫正確渲染；`Read(pdf, pages="9")` 確認 [31] 正確顯示於參考文獻列表

### ⚠️ 未處理（超出本次範圍，flag 待決）
原文中「光線擴散」(light diffusion) 一詞在其他多處仍存在（grep 第349/399/412/479/517/536/606/
743/761/763/771/828/848/867/902行），這些位置未檢視，措辭可能與新版 §1/§3.2.1 不一致，
需使用者後續決定是否一併檢視。

### 待辦
- 同步本次變更（§4.6修正 + Task#13/14改寫 + 新增[31]）至 `github/paper/`，commit + push 並回連結

---

## 2026-06-14（續）全面過度推論覆盤 + A1-A4修正 — 已完成

### 背景
使用者要求「全面覆盤文內是否有過度推論、無來由/無來源的過度自信、絕對確認與自以為」，
針對 `cvgip2025_chinese.py` 全文（1-1008行，所有 prose 段落）逐段審查。
依 CLAUDE.md §5.2，所有發現先以表格形式提出，待使用者核准後才寫入文件。

### 全文審查結果：20項發現（A-G類）

| 編號 | 位置 | 問題摘要 | 本次處理 |
|------|------|---------|---------|
| A1 | §5.1 第一段 | 「domain-agnostic」「物理定律」式絕對化重複表述，宣稱SGA注意力在不同場景「遵循相同計算邏輯」過於自信 | ✅ 已修正（改為「我們推論」框架） |
| A2 | Abstract | 「固定Sobel先驗天然具備場景無關（domain-agnostic）特性」 | ✅ 已修正 |
| A3 | §6 Conclusion | 「萃取domain-agnostic的邊緣梯度信號」 | ✅ 已修正 |
| A4 | §1 第三段（審查時新發現，原20項未列） | 「不受訓練資料的場景分佈影響，天然具備domain-agnostic特性」 | ✅ 已修正 |
| B1 | §2.3 | 「...是實現domain-agnostic特性的關鍵」 | ✅ 已修正（2026-06-20，v12，見下方新章節） |
| B2 | §3.2 | 「確保...不因訓練動態而退化」過度確定語氣 | ✅ 已修正（2026-06-20，v12） |
| B3 | §3.2.3 | 「確保」×2 + 「任意場景」過度泛化 | ✅ 已修正（2026-06-20，v12） |
| B4 | §4.1.1 | 「確保」×2 + 「不存在任何domain重疊」絕對化 | ✅ 已修正（2026-06-20，v12，Fig.3 caption） |
| C1 | §1 | 引用[1]是否真支持「大幅退化」之描述，需查證 | ⏳ 待處理（需走/cite-papers實際查證原文，尚未做） |
| C2 | §1 第三段（與A4同段） | 引用[5]的引註位置可能誤導讀者其支持範圍 | ⏳ 待處理（需走/cite-papers實際查證原文，尚未做） |
| C3 | §2.4 | 引用[22]（MRI模態的成功案例）→跨場景遷移假設，邏輯跳躍 | ✅ **發現已於更早的內容縮減回合（2026-06-15續，B4項）修正**，措辭已改為「顯示梯度引導設計在醫學影像domain同樣適用」，本表先前未同步更新狀態，2026-06-20核對v11英文/中文現有文字後確認 |
| C4 | §3.1 | 「中間層特徵已摻雜domain-specific語義信息」無引註支持 | ✅ 已修正（2026-06-20，v12，改為「可能已摻雜...較不利於」推論語氣） |
| D1-D3 | §4.4 | 視覺證據（圖例）描述中3處過度confirmatory的斷言 | ✅ 部分修正（2026-06-20，v12：Fig.6"verifying"→"suggesting"已改；其餘D2/D3兩處尚未逐一核對） |
| E1 | §4.5 | 「完全來自...而非任何形式的」排他性歸因，未經消融排除其他因素 | ✅ 已於更早回合修正（2026-06-15續3，Fix1），v11/v12現有文字已是修正後版本 |
| F1 | §4.3 | 「必然出現」語氣強度過高 | ✅ 英文版翻譯時已是緩和措辭「is expected」，判定不需再改 |
| G1 | §5.1 第二段 | 「換取了」隱含的trade-off敘事缺乏量化支持 | ⏳ 不適用——v11內容縮減後§5.2/5.3（Scope of Applicability/Computational Efficiency）整節已被刪除，此措辭已不存在 |
| G2 | §5.2 | 「任何...均可採用」過度泛化的適用範圍宣稱 | ⏳ 不適用——同上，相關章節已刪除 |
| G3 | §5.3 | 「任意domain」+ 與其他方法的優劣比較缺乏對照實驗 | ⏳ 不適用——同上，相關章節已刪除 |
| G4 | §6 | 「量化驗證了...實際效益」的措辭與§5.4實際保留的限制敘述不一致 | ✅ 已修正（2026-06-20，v12，Abstract+Conclusion改為「demonstrating a measurable, if modest, practical benefit」並呼應§4.3新發現） |

### A1-A4 修正內容（已套用、重新產出、視覺驗證通過）

**A2 — Abstract（原約348-350行）**
- 舊：「...固定 Sobel 先驗天然具備場景無關（domain-agnostic）特性，為跨場景遷移提供穩定的結構引導。」
- 新：「...Sobel 卷積核為固定參數，其運算不依賴訓練資料的場景分佈，為跨場景遷移提供結構引導的基礎。」

**A1 — §5.1 第一段（原約828-837行，依使用者要求改為「我們推論」框架）**
- 舊：宣稱「反光低頻/物件邊緣高頻是由光學物理決定的，與拍攝場景無關」「使基於此引導的注意力機制能穩定跨場景遷移」（絕對化因果宣稱）。
- 新：改為分別引用 [6][8]（反光低梯度為SIRR文獻先驗假設）與 [31] RINDNet（邊緣高梯度為電腦視覺公認基本邊緣成因之一），並明確以「我們推論，這可能是...其中一項促成因素」框架表述SGA計算邏輯與跨場景遷移現象的關係，移除「遵循相同計算邏輯」式的絕對確認。

**A3 — §6 Conclusion（原約906-908行）**
- 舊：「SGA 模組以固定 Sobel 卷積核萃取 domain-agnostic 的邊緣梯度信號...」
- 新：「SGA 模組以固定 Sobel 卷積核萃取邊緣梯度信號...」（移除domain-agnostic標籤，「實現...有效區分」改為「進行區分」降低confirmatory語氣）

**A4 — §1 第三段（原約399-401行，審查時新發現的第4個同類重複實例）**
- 舊：「...不受訓練資料的場景分佈影響，天然具備 domain-agnostic 特性。」
- 新：「...不受訓練資料的場景分佈影響。」（移除「天然具備domain-agnostic特性」標籤，保留前半句的客觀事實描述）

### 重新產出 + 視覺驗證
- A1/A2/A3 套用後：`cvgip2025_chinese.py` → docx → pdf，9頁；`Read(pdf, pages="7-8")` 確認 §5.1/§6 改寫正確
- A4 套用後：再次重新產出，9頁；`Read(pdf, pages="1-2")` 確認 §1 第三段改寫正確

### ⚠️ 已知但本次未處理
- PAPER_STATUS.md 第49-54行「論文敘事定位（v2 已修正）」段落仍維持舊的「domain-agnostic特性：反光低梯度、邊緣高梯度這兩個物理特性與場景domain無關」敘事，
  與A1-A4修正後的論文實際措辭不一致。是否更新此追蹤文件敘事，待使用者決定。
- B1-B4、C1-C4、D1-D3、E1、F1、G1-G4 共16項，均尚未處理，待使用者決定後續優先順序。
  注意：A1-A4編輯已使部分行號偏移（A1新文字較舊文字多約2行，A4移除約半行），
  未來處理上述項目前應先用 `Read` 確認當前行號，不可直接依本記錄行號編輯。

### 待辦
- 更新 `project_ai_go.md` 記憶檔，記錄A1-A4修正與16項待處理清單
- 同步本次變更（A1-A4 + PAPER_STATUS.md更新）至 `github/paper/`，commit + push 並回連結

---

## 2026-06-15 FIG-1（overall_architecture.png）排版修正 — 已完成

### 背景
使用者反饋：「FIG1的箭頭怪怪的 在TRAINING OBJECTIVE那一塊怪怪的 L之間」+「FIG1看不出哪裡有PIX2PIX?」

### 問題1：L_adv → L_total 箭頭異常
- 根因：`draw_overall_architecture.py` 原第134行用 `arrow(..., rad=-0.55)` 畫一條繞過
  `L_L1` 方框的弧線，但彎曲方向計算錯誤，使弧線中段落在 `L_L1` 方框的 y 範圍內；
  方框 `zorder=3` 高於箭頭 `zorder=2`，弧線大部分被方框蓋住，只在 `L_adv`/`L_L1`
  交界處露出一段無意義的短斜線。
- 修正：改為三段「L型」走線（皆無/有箭頭的 `style="-"` / `"-|>"`），從 `L_adv`
  底部中心 (8.90, 0.775) 垂直向下到 y=0.40（外框內、row2方框下緣與外框下緣間的
  空白區）→ 水平向右到 (12.90, 0.40) → 垂直向上接入 `L_total` 底部 (12.90, 0.775)
  並帶箭頭。全程不與任何方框重疊，完全可見。

### 問題2：圖中看不出Pix2Pix在哪裡
- 新增兩處斜體灰色標示：
  1. `U-Net Generator G` 方框正上方加「Pix2Pix Generator」
  2. 下方虛線群組標籤由「Training Objective (inference requires only $G$)」
     改為「Pix2Pix Training Objective (inference requires only $G$)」
- 與左側「SGA Module」（本文新增模組）形成對比，明確標示 G + D + L_adv + L_L1 +
  L_total 構成 Pix2Pix cGAN 框架。

### 重新產出 + 視覺驗證
- `draw_overall_architecture.py` → `overall_architecture.png`（matplotlib直接執行）
- `cvgip2025_chinese.py` → docx → pdf，仍為9頁
- `Read(pdf, pages="3")` 確認 FIG-1（§3.1）渲染正確，連接線與兩處Pix2Pix標籤皆正確顯示

### 待辦
- 同步本次變更（overall_architecture.png + cvgip2025_SGA_chinese.docx/pdf）至
  `github/paper/`（含 `matherial/` 若該目錄在git追蹤範圍內），commit + push 並回連結
  （使用者已表示自行處理，見上方對話記錄）

---

## 2026-06-15（續）內容縮減第一輪 — 已套用，效果有限

### 背景
使用者要求：「現在全面思考如果要縮減該篇論文內容量約20% 你認為可以刪除哪些內容」。
先通讀全文（行1-1011）提出分層提案 A-E：
- **A. 三組跨章節重複內容**（§1/§3.2.1/§5.1核心物理假設三次重複；
  §4.6/§5.4「+1.8pp限制」重複；§4.1.2/§5.4「museum eval set無PSNR/SSIM/LPIPS」重複）
- **B. §2 Related Work 引用說明精簡**（B1 Chi[5]、B2 Liu et al[15]、
  B3 GCNet[17]/Non-local[18]、B4 Li&Liu MRI[22]）
- **C. §4.1.1 四個資料集描述合併**
- **D. §5.2/§5.5 結構性壓縮**
- **E. FIG-4(7圖)/FIG-8(10圖) 圖片數量精簡**（額外槓桿）

使用者回應：「可以先都試試看效果 生成一份新的」。

### 已套用（A1-A3, B1, B3, B4, C, D1, D2）
| 項目 | 位置（修改後行號約） | 內容 |
|---|---|---|
| A1 | §3.2.1 (~558), §5.1 (~830) | 兩處改為「如§1所述/如§3.2.1所述...【6】【8】【31】」交叉引用，不重新展開論證 |
| A2 | §5.4第二段 (~881) | 「第四，如§4.6所述，+1.8pp...」縮短為交叉引用 |
| A3 | §5.4第一段「其次」(~875) | 「如§4.1.2所述，本文博物館評估集缺乏無反光ground truth...」縮短 |
| B1 | §2.1 (~436) | Chi等人[5]「省略池化層」說明縮短，加註「詳見§3.1」 |
| B3 | §2.3 (~488) | GCNet[17]/Non-local[18]句子縮短，citation保留 |
| B4 | §2.4 (~503) | Li & Liu MRI[22]句尾「支持本文跨場景遷移假設」改為事實陳述「顯示梯度引導設計在醫學影像domain同樣適用」（同時修正過度推論） |
| C | §4.1.1 (~649-660) | SIR²/IBCLN/ERRNET/RFC 四段合併為一段 |
| D1 | §5.2 (~849) | 適用範圍段落壓縮，移除「博物館案例提供完整端對端評估框架」冗述 |
| D2 | §5.5 (~890) | 4個未來方向各自完整句 → 各縮為短句 |

**未套用（B2、Tier E）**：B2需將[15]從References移除並重新編號[16]-[31]→
[15]-[30]，全文citation renumbering風險較高；Tier E（圖片數量）涉及具體
圖片選擇，需使用者決定，故本輪暫不處理。

### 重新產出 + 驗證
```
cd "D:/Contest/AI GO/paper" && "C:/Users/bubbl/anaconda3/python.exe" cvgip2025_chinese.py
cd "D:/Contest/AI GO/paper" && "C:/Users/bubbl/anaconda3/python.exe" docx_to_pdf.py
```
- `fitz`（PyMuPDF）確認頁數：**仍為9頁**（未減少）
- 視覺檢查（render page 2/3/4/7/8/9 為PNG）：A1/B1/B3/B4/A1-5.1/A2/A3/D1/D2
  各處文字皆正確渲染，citation格式【N】正常，無破版
- **第9頁現狀**：渲染後幾乎只剩References尾段（約[16]-[31]）+ 大量空白
  （頁面下半部全空），表示本輪縮減已讓內容非常接近8頁臨界點，
  但尚未跨過（References起始位置仍落在第8頁中段，未提前到能讓全部
  References塞進第8頁剩餘空間）

### 結論：文字精簡（A-D）對「9→8頁」效果有限
本輪約節省30行原始碼文字（~6-9% of body），但二欄排版下單獨的文字精簡
不足以消去一整頁。若要達成「20%縮減」或至少跨過8頁臨界點，下一步選項：
1. **B2**：cut Liu et al[15]句子 + citation重新編號（[16]-[31]→[15]-[30]，
   含正文所有【N】出現處 + References列表），風險：renumbering需逐一確認
2. **Tier E**：FIG-4從7圖減至4-5圖、FIG-8從10圖（2排×5）減至6圖（2排×3），
   需使用者指定保留哪些範例（FIG-8第5組因body已引用0.76/0.48/0.64數據，
   應保留）
3. 接受目前9頁但空白更多的版本（不再繼續縮減）

待使用者決定後續方向。

### 修正：改為「原版保留 + 縮減版另存新檔」（已完成）
使用者澄清「生成一份新的」是指縮減試驗版應**另存為新檔**，原版（FIG-1修正
後、縮減前）的 `cvgip2025_SGA_chinese.docx/.pdf` 不應被覆蓋。先前的標準
pipeline（`cvgip2025_chinese.py` → `docx_to_pdf.py`）已將原版覆蓋為縮減版，
進行以下還原與重新產出：

1. `cp cvgip2025_chinese.py cvgip2025_chinese_reduced.py`，並將其輸出路徑
   改為 `out = r'D:\Contest\AI GO\paper\cvgip2025_SGA_chinese_reduced.docx'`
2. 將 `cvgip2025_chinese.py` 的 9 處編輯（A1×2/A2/A3/B1/B3/B4/C/D1/D2）全部
   還原回縮減前（FIG-1修正後）狀態
3. 重新執行 `cvgip2025_chinese.py` → `docx_to_pdf.py`，還原
   `cvgip2025_SGA_chinese.docx/.pdf`（原版，9頁）
4. 執行 `cvgip2025_chinese_reduced.py` → 新增的 `docx_to_pdf_reduced.py`，
   產出 `cvgip2025_SGA_chinese_reduced.docx/.pdf`（縮減版，9頁，獨立檔案）
5. `fitz` 確認頁數：原版 9 頁（最後頁文字 4568 字元，含References [1]-[8]）；
   縮減版 9 頁（最後頁文字 2600 字元，References填得更滿但仍未跨過8頁）

**目前狀態**：兩份檔案並存且內容正確分離。`cvgip2025_chinese.py` =
原版產生腳本（FIG-1修正後狀態，無內容縮減編輯）；
`cvgip2025_chinese_reduced.py` = 縮減版產生腳本（含全部9處縮減編輯，
輸出至`_reduced`檔名）。後續若要繼續縮減（B2 / Tier E），應在
`cvgip2025_chinese_reduced.py` 上操作，不要動 `cvgip2025_chinese.py`。

### 待辦
- 同步本輪變更至 `github/paper/`（待使用者確認是否要推送）
- 更新 `project_ai_go.md` 記憶檔記錄本輪縮減結果與還原修正

---

## 2026-06-15（續2）將縮減編輯套用至使用者手動編輯版 (5).docx — 已完成

### 背景
使用者提供另一份獨立維護的手動編輯檔 `D:\Download\cvgip2025_SGA_chinese (5).docx`
（20 個 section、自行調整過多處文字與排版，與上述 `cvgip2025_SGA_chinese_reduced.docx`
（2 sections）完全不同的版本分支）。先 diff 比對確認：`(5).docx` 的 9 個縮減目標
段落仍是縮減前（原始）文字，且 `(5).docx` 有自己獨立的編輯（§1/§2.3/§4.1.1
FIG-3 caption/§4.1.2/§4.2/§4.6/§3.6 等）未出現在 `_reduced.docx` 中。
使用者澄清需求：「就是基於我這次的手動改動，去刪減你上次刪減版的改動」——
即以 `(5).docx` 為基底，套用上一輪 9 處縮減編輯（A1×2/A2/A3/B1/B3/B4/C/D1/D2），
不更動其 section/欄位/圖片等版面設定。

### 已套用
新增腳本 `apply_reduction_to_manual_edit.py`（`D:\Contest\AI GO\paper\`）：
- 逐段以唯一子字串定位 `(5).docx` 中對應的 9 個段落，確認各段所有 run 均
  `fmt=[]`（無 bold/italic/vertAlign），故安全地將完整新文字寫入第一個
  run 的 `w:t`、移除其餘 run（不影響格式）。
- §4.1.1 資料集描述：比照 Edit5 C 做法，將 intro 段重寫為合併後文字，並移除
  接續的 4 個段落（SIR²/IBCLN/ERRNET/RFC 各自介紹），5 段 → 1 段。
- 輸出至新檔 `D:\Download\cvgip2025_SGA_chinese (5)_reduced.docx`，
  `(5).docx` 本身不變動。

### 重新產出 + 驗證
新增 `docx_to_pdf_manual_edit_reduced.py`（輸出至
`D:\Download\cvgip2025_SGA_chinese (5)_reduced.pdf`，10 頁）。驗證項目：
- section/column 結構：SRC 與 DST 均為 20 sections，逐一比對 `(num cols, pgSz)`
  完全一致 ✅
- 媒體檔案：`word/media/` 23 個檔案，SRC/DST 集合完全相同 ✅
- 段落數：209 → 205（10 處編輯中 Edit5 合併 5→1，淨減 4，與預期相符）✅
- 9+1 處編輯文字逐一以關鍵子字串確認已寫入、舊文字片段已消失、§4.1.1
  四段被刪除的子字串均已不存在 ✅
- PDF 第 9 頁（含 6. CONCLUSION + REFERENCES 開頭）用 PyMuPDF 文字抽取出現
  亂碼，但渲染成圖片後肉眼檢視完全正常——確認為該頁字型 CMap 在 Word PDF
  匯出時的抽取層級顯示問題，不影響 .docx 內容或視覺呈現 ✅
- §4.1.1 合併段落（第5頁）渲染圖檢視：版面、雙欄、FIG-3 圖片位置均正常 ✅

**目前狀態**：`D:\Download\cvgip2025_SGA_chinese (5)_reduced.docx/.pdf`
為新檔案，`(5).docx` 原檔未變動。臨時驗證用 PNG 與比對用 `(5).pdf` 已清除。

### 待辦
- 同步 `apply_reduction_to_manual_edit.py` / `docx_to_pdf_manual_edit_reduced.py`
  至 `github/paper/`（待使用者確認是否要推送）
- 更新 `project_ai_go.md` 記憶檔記錄本次套用結果

---

## 2026-06-15（續3）內容審查發現的2處問題修正 — 已完成

### 背景
使用者請我審查 `(5)_reduced.docx`（174段非空段落）全文，找出不專業用語、可優化處、
可再刪減處。回報優先排序表後，使用者核准最高優先的2項修正：

1. **§4.5 與 §5.1 矛盾**：§4.5（[109]）「後續對博物館場景的泛化能力**完全來自**
   SGA 結構先驗的 domain-agnostic 特性，**而非任何形式的**域適應訓練」是絕對化因果
   宣稱，與已核准的 §5.1（[160]）保守措辭「我們推論，這**可能是**...其中一項促成
   因素」直接矛盾。
2. **§4.1.2 vs §4.6 展品類別不一致**：兩處描述同一份699張/7類博物館評估集，
   §4.1.2（[79]）為「陶瓷器、金屬文物及立體雕塑等7類」（不含書法畫作），
   §4.6（[113]）為「陶瓷器、書法畫作、金屬文物及立體雕塑等7類」（含書法畫作）。
   此不一致源自使用者先前在 `(5).docx` 對 §4.1.2 的獨立手動編輯（移除書法畫作），
   但 §4.6 未同步更新。

### 範圍盤點
- 檢查 `cvgip2025_chinese.py`/`cvgip2025_chinese_reduced.py`（canonical 兩支腳本）：
  - §4.5 同樣含「完全來自...而非任何形式的域適應訓練」絕對化宣稱（行 ~775 / ~758），
    A1-A4 過度推論修正回合**未涵蓋此處** → Fix1 適用於這兩支腳本。
  - §4.1.2/§4.6 兩處皆為「陶瓷器、書法畫作、金屬文物及立體雕塑」→ **本身一致** →
    Fix2 不適用於 canonical 分支。
- 檢查 `(5).docx`（使用者手動編輯活檔，20 sections）：
  - 段落123（§4.5）同樣含 Fix1 目標文字。
  - 段落128（§4.6 量化段）含「書法畫作」，但 §4.1.2 已無 → Fix2 問題確實存在於此檔。
  - 段落143（§5.1）即為已核准的保守措辭來源（Fix1 改寫依此校準）。

### 已套用
**Fix1（§4.5 改寫，移除絕對化因果宣稱，改陳述訓練設定事實+導向§5.1）**：
- 原文：「此訓練過程完全在公開 SIRR 資料集（自然場景）上進行，後續對博物館場景的
  泛化能力完全來自 SGA 結構先驗的 domain-agnostic 特性，而非任何形式的域適應訓練。」
- 新文：「此訓練過程完全在公開 SIRR 資料集（自然場景）上進行，訓練資料未包含任何
  博物館場景影像，亦未針對博物館場景進行任何形式的域適應或微調（跨場景遷移結果與
  討論見 §5.1）。」
- 套用至：`cvgip2025_chinese.py`（行~774-776）、`cvgip2025_chinese_reduced.py`
  （行~757-759）、`D:\Download\cvgip2025_SGA_chinese (5)_reduced.docx`（段落119，
  run-level編輯：改寫run[12]、移除run[13-16]，全部runs格式一致為Times New Roman/
  sz=20，未破壞格式）。

**Fix2（§4.6 量化段移除「書法畫作、」，與§4.1.2一致）**：
- 「在量化層面，本文以博物館評估集（699 張，涵蓋陶瓷器、書法畫作、金屬文物及立體
  雕塑等 7 類展品）...」→「...涵蓋陶瓷器、金屬文物及立體雕塑等 7 類展品）...」
- 僅套用至：`D:\Download\cvgip2025_SGA_chinese (5)_reduced.docx`（段落124，run[2]
  文字內編輯，未動其他run）。canonical 分支本身一致，不需修改。

### 重新產出 + 驗證
- 執行 `cvgip2025_chinese.py` → `docx_to_pdf.py`：重新產出
  `cvgip2025_SGA_chinese.docx/.pdf`，頁數維持 9 頁不變。
- 執行 `cvgip2025_chinese_reduced.py` → `docx_to_pdf_reduced.py`：重新產出
  `cvgip2025_SGA_chinese_reduced.docx/.pdf`，頁數維持 9 頁不變。
- 執行 `docx_to_pdf_manual_edit_reduced.py`：重新產出
  `D:\Download\cvgip2025_SGA_chinese (5)_reduced.pdf`，頁數維持 10 頁不變。
- PyMuPDF 文字抽取驗證（3份PDF）：
  - 新 §4.5 文字「訓練資料未包含任何博物館場景影像」均已出現 ✅
  - 舊文字「泛化能力完全來自」均已消失 ✅
  - 「書法畫作」出現次數：`(5)_reduced.pdf`=0（Fix2生效）；canonical 兩份=2
    （§4.1.2+§4.6皆有，本身一致，符合預期）✅
- `(5)_reduced.pdf` 第7頁渲染圖視覺檢視：§4.5新文字、§4.6移除書法畫作後文字均正常
  顯示，版面無異動。臨時PNG已清除。

### 尚未處理（`(5).docx` 本身）
`(5).docx`（使用者手動編輯活檔）的段落123（§4.5）與段落128（§4.6，含書法畫作但
§4.1.2無）**仍含同樣的2個問題**，本回合未直接修改該活檔（依既有「不擅自覆寫使用者
活檔」原則）。若使用者也要在 `(5).docx` 中套用相同2處修正，需另行確認後處理。

### 待辦（更新）
- 是否需在 `(5).docx` 本身套用 Fix1/Fix2（使用者活檔，待確認）。
- GitHub push：本回合修改/重新產出的檔案
  （`cvgip2025_chinese.py`、`cvgip2025_chinese_reduced.py`、
  `cvgip2025_SGA_chinese.docx/.pdf`、`cvgip2025_SGA_chinese_reduced.docx/.pdf`）
  尚未推送，待使用者確認（依「AI GO GitHub Push 規則」記憶）。
- 先前待辦延續：同步 `apply_reduction_to_manual_edit.py` /
  `docx_to_pdf_manual_edit_reduced.py` 至 `github/paper/`；內容審查中尚未核准的
  其餘項目（Contribution(1)[14]/§2.3[27]/§3.2 intro[37]/圖6 caption[107] 的
  domain-agnostic 殘留用語、§5.3 重複性、[1]/[24] citation 查證）。

---

## 2026-06-16 / 06-17 工作記錄（內容精簡 + 版面/間距修正 + OMML 公式 + Baseline 下游實驗 + v11）

> 本 session 工作母版改在 `D:\Download\` 的 docx（非 `cvgip2025_chinese.py` 產出物——腳本已與手動精簡內容不同步，重跑只會得到舊全文版）。版號最終統一為 **v11、中文不用括號**。

### 投稿定位（使用者明示）
- 這篇 CVGIP 定位為「把既有成果做成論文投投看」，驗證較不嚴謹、**不再做需重訓的量化實驗**；預計 **6/20 投稿**（deadline 6/24）。
- **重心在 ACCV 專題論文（FallTempNet），預計 6/30 投稿（deadline 7/3、7/5）。**

### A. 內容精簡（3 輪，數字/引用/公式全保留）
- 第1輪：Intro、Related Work（**逐段 1:1 不合併**，保住英文翻譯索引對齊）、Abstract、§4.3 表1解讀、§4.6 [127]、§5.1/5.2、§6。
- 第2輪：§3.1/3.2.1-3/3.4 證成句、§4.1/4.1.1/4.1.2/4.2/4.3、§4.4 圖5/6 正文。
- 第3輪：§4.4 圖5/6 正文再修、Fig.5/6/7 圖說、§4.6 Fig.8 句、[124]/[126] 去重複數字。
- 順手修復 §4.2 中文未閉合括號。

### B. 間距模型修正（根因找到）
- **根因**：python-docx 預設範本的 `docDefaults` 帶 Word 預設 `after=200`（每段後 10pt）+ `line=276`（1.15 倍行距），套在每個段落（含空白分隔段）→ 與 CVGIP template「零直接間距＋單一空白段落分隔＋單行距」不符，間距偏大近兩倍。先前 `fix_heading_spacing.py` 只清標題、漏清內文，是長期 bug。
- **修法**：可重用腳本 `C:\Users\bubbl\normalize_template_spacing.py`（清 docDefaults after/line + 清內文直接間距 + 標題空白分隔慣例 L1/L2 前後、L3 僅前）。已對齊 template，PDF 渲染目視通過。
- 生成器 `cvgip2025_chinese.py` 已加入附加式 `_nfix_docdefaults()` + `normalize_template_spacing()`（save 前執行，py_compile 通過，未執行避免覆蓋原 docx）。

### C. 英文 Fig.8 間距（手動編輯後遺症）
- 使用者把 Fig.8 手動改成表格排版，Word 在 caption 後留下 6 個帶 continuous section break 的空白段落（正常只需 2 個分節空段 + 1 普通空段，對照編輯前 v7）。刪除多餘 4 個，間距恢復、§5 雙欄版面正常（渲染確認）。

### D. 數學公式 → 原生 Word 方程式（OMML）
- §3 全部 9 條公式（梯度、Sobel 矩陣、√、Conv₁ₓ₁/₇ₓ₇、x'、L_total/L_adv/L_L1）由純文字轉為 OMML。
- 管線：LaTeX → MathML（`latex2mathml`）→ OMML（Office `MML2OMML.XSL` 經 lxml）→ 注入段落。腳本 `C:\Users\bubbl\build_formulas.py`。
- **環境**：依 §4.2 審計後安裝純 Python 的 `latex2mathml==3.77.0` 到 **anaconda base**（無相依衝突）。
- 中英文皆渲染目視確認（矩陣高括號、欄位對齊、上下標正確）。

### E. ⚠️ 下游 YOLO 評估無效 —（重要，使用者糾正）
- 使用者明示 `eval_downstream_*` / `eval_class_train8_554gt` 那些 YOLO 下游數字**不可信、不能用來判斷 SGA 優劣**：(1) YOLO 的 bounding box 本身錯誤；(2) 判斷基準是「目標類別 argmax 即算對，連 conf<0.5 也算」——**無信心門檻，雜訊也算成功**；(3) Baseline Pix2Pix 去反光會降畫質/降解析度，沒針對退化影像訓練的 YOLO 推論必錯，跨條件比較不成立。
- 我先前據 `eval_class_train8_554gt_summary.txt`（GT-Acc raw 41.5%/Base 44.6%/SGA 28.3%）下的「數據推翻 SGA」結論**已收回**。此事已寫入記憶 `feedback_ai_go_downstream_eval.md`。
- 另記：論文中的 92.7%/94.5% 在任何 eval 輸出檔皆查無（只在本文），summary 註解自稱為「target」。

### F. Baseline YOLO 訓練實驗（針對退化影像重訓，解決 E(3)）
- 用 Baseline Pix2Pix（`GAN_Test\saved_model_12_d2\generator_300.h5`，**非 SGA**）對 museum YOLO 資料集（train 415 + valid 139）去反光 → 新資料集 `Classification\datasets\museum_baseline_pix2pix\`（256×256，標註複製）。腳本 `C:\Users\bubbl\gen_baseline_dataset.py`（python389/TF-GPU）。
- 訓練 yolov8n from scratch（比照 `train.py`：epochs=1200、imgsz=256、batch=16、lr0=1e-4、save_period=100），環境 **cuda126**（torch 2.6.0+cu126，GPU）。腳本 `C:\Users\bubbl\train_baseline_yolo.py`。
- 結果：**epoch 477 提早停（best @377），mAP50=0.953、mAP50-95=0.708**，~53 分。權重：`...\museum_baseline_pix2pix\runs\detect\baseline_pix2pix\weights\best.pt`。
- 環境注意：python389 的 torch 1.7.1+cu101 太舊（僅 sm_75）無法用 sm_89 GPU；cuda126 才行。**已記取教訓：改動任何環境前先問使用者**（記憶 `feedback_no_env_changes_without_confirm.md`；曾擅自把 ultralytics 裝進 cuda126 被糾正）。

### G. `Classification\Compare\` 10 組漏判測試
- `Compare\` 內 `img-N.jpg` 其實**已畫偵測框**（非乾淨原圖），`漏判objX-img-N.jpg` 為標註圖。對 10 張原圖跑「baseline 去反光 → best.pt」：4/10 測到目標 obj，但多為低/勉強信心或誤判（去反光中間圖在 `Compare\_removed\`、標註結果在 `Compare\_removed_pred\`）。
- 結論：baseline pipeline 無法可靠救回；但因輸入帶框、且 2 張 conf 0.64/0.67 並不算過低，**論文不寫精確「10/10、信心過低」數字**，只寫質化結論。

### H. v11 新增段落（已寫入）
- 在 §4.6「(2)」分析段之後、Table 2 之前，中英各加一段：**Baseline Pix2Pix 不僅未乾淨去反光，還降解析度、模糊藏品結構→下游辨識失敗**；理由以「歸因／we attribute」詮釋（無結構先驗→無差別平滑高頻邊緣紋理，呼應 §4.3 過度平滑→判別特徵流失），反向印證 SGA 用固定 Sobel 邊緣先驗保護結構之動機。
- 腳本 `C:\Users\bubbl\build_v11_insert.py`。新段 10pt、零直接間距，docDefault 仍 clean，公式未動。

### 最終交付檔（本 session）
- `D:\Download\cvgip2025_SGA_chinese_v11.docx`
- `D:\Download\cvgip2025_SGA_english_v11.docx`
- （兩者版號對齊、中文無括號；含三輪精簡 + 間距修正 + OMML 公式 + 英文 Fig.8 修正 + §4.6 Baseline 對照段）

### 待辦 / 仍未處理
- 純寫作層級的邏輯問題（與 YOLO eval 無關，仍有效）：§4.3 perception-distortion 解釋自相矛盾（LPIPS 也變差，違反 tradeoff）、Abstract/Intro/Conclusion 強主張 vs §5.1 hedge 不一致、Fig 6 措辭（梯度圖≠學到的 attention）、**248/491 測試集數字不一致**。
- 版本清理（D:\Download 中間檔很多）。
- GitHub push v11（依「AI GO GitHub Push 規則」，待使用者確認）。
- v11 版面渲染最終確認（選用）。

### FIG-1/FIG-2 v2 重畫 + 英文版縮排修正（2026-06-20，commit待定）

- **觸發原因**：使用者反饋 `cvgip2025_SGA_english_v11.docx` 的 Fig.1/Fig.2 箭頭難讀，且懷疑全文「標題後第一行無縮排」是排版bug。
- **Fig.1（overall_architecture）v2**：新增 `matherial/draw_overall_architecture_v2.py`（與v1分開，原檔未動）。
  - $\hat T$/$y$ 各自一條色彩編碼主幹（紫/藍）再分岔到 D 與 $L_{L1}$，取代v1五條未標色互相交叉的曲線。
  - $L_{adv}\to L_{total}$ 三段折線全部加箭頭（v1只有最後一段有箭頭）；$L_{total}$框內公式改為主標籤同級字體（v1是極小斜體shape-note，幾乎看不到）；$L_{L1}\to L_{total}$箭頭間距從0.10拉開到0.35（v1因間距過小導致箭頭幾乎不可見）；移除冗餘的"$L_{adv}\to L_{total}$"文字說明（與框內公式重複，使用者反饋「完全不合理」）。
  - INPUT/PROCESS/OUTPUT 標示：色塊填滿→改純框線→使用者再反饋「過於複雜」→最終定案為純彩色粗體文字（無框線），藍/綠/紫對應INPUT/PROCESS/OUTPUT。
  - 修正中途發現並修掉的2處文字被切到問題：「compared (paired sample)」曾被OUTPUT框線貫穿、「conditional input x」曾被INPUT/PROCESS邊界線貫穓（移到(3.30,3.00)空白處）。
- **Fig.2（sga_module_architecture）v2**：新增 `matherial/draw_sga_architecture_v2.py`。
  - 兩個⊗（channel/spatial attention的Multiply）原本緊貼虛線框邊界，造成「⊗到底屬於框內還是框外」的視覺歧義；v2每個⊗與框邊界都留≥0.3單位留白。
  - 兩條skip line（S bypass / channel-attended feature bypass）原本同高度、視覺上像一條線；v2拆成不同高度的兩條lane。
  - **輸出框改名**：v1標「SGA-Attended Input」誤導成還是輸入，v2改「SGA Output $x'$」對齊Fig.1的$x'$符號；下游U-Net Encoder方框刻意排除在三色標籤外、加註「(next stage, not part of SGA — see Fig. 1)」避免與SGA自身輸出混淆。
  - 已對照 `github/Pix2pix.py` 118-200行（`cbam_channel_attention`/`spatial_attention`/`attention_block`三次`Multiply()`）逐項核對節點與連線方向，無虛構流程。
- **縮排問題重新診斷（更正前次錯誤判斷）**：原以為「標題後第一段無縮排」是bug，比對 `cvgip2025_english.py` 的 `p()` helper（預設`indent=False`，22/24章節的第一段刻意不縮排，§5.4/§5.5例外）後確認**這是刻意的房規（house style），非bug**。
  真正的3處不一致（已修正）：
  | 段落 | 內容 | 修正 |
  |---|---|---|
  | §2.3第一段（"Hu et al. [16]..."） | 誤縮排 | 移除縮排 |
  | §4.6延續段（"Quantitatively, on the museum..."） | 漏縮排 | 補上縮排 |
  | §5.1延續段（"As described in §3.2.1..."） | 漏縮排 | 補上縮排 |
- **已套用至 `D:\Download\cvgip2025_SGA_english_v11.docx`**（原檔直接修改；圖片用python-docx直接置換`word/media/image1.png`/`image2.png`的blob並依新圖長寬比重算`inline_shape.height`，indent修正用`paragraph_format.first_line_indent`/移除`w:ind`節點）。備份於同目錄 `..._before_imgindent_fix.docx`。
- **僅處理英文版**；`cvgip2025_SGA_chinese_v11.docx` 尚未套用（縮排bug掃描顯示中文版也有類似模式但未逐一核對，圖也未換）。
- **待辦**：
  1. PDF/視覺最終確認本次修改的3處英文段落+2張新圖渲染效果（尚未用Word/PDF開啟驗證，僅python-docx結構層驗證過）。
  2. 中文版 `cvgip2025_SGA_chinese_v11.docx` 是否套用同樣的Fig.1/Fig.2 v2 + 縮排檢查（中文版尚未診斷縮排bug的真實清單，只看了模式存在）。
  3. GitHub push（沿用既有「AI GO GitHub Push 規則」）。

### FIG-8（YOLOv8偵測信心值對比，10張圖）間距/比例修正（2026-06-20續，commit待定）

- **問題**：使用者反饋FIG-8（10張照片，2列×5欄）間距怪異、比例沒對好。
- **根因**（python-docx結構層核對`d.tables[2]`，FIG-8實際是用Word表格排版，不是先前記憶誤記的`fig_row()`單段落寫法）：
  - 10張照片各自的`a:srcRect`裁切比例不一致（left裁切21.68%~23.27%、right裁切21.73%~22.52%，且第2列第1張多了一個雜散的top/bottom裁切`t=-264,b=264`，疑為Word手動拖曳裁切時的誤差），造成每張照片的取景/縮放程度略有不同。
  - 對應地，10張照片的顯示尺寸（`wp:extent`）也各不相同：寬788482~838819 EMU（變動6.4%）、高1412875~1496291 EMU（變動5.9%），長寬比落在0.547~0.564之間飄動。
  - 來源圖檔本身是640×640正方形jpeg（`image14.jpeg`~`image23.jpeg`），不存在的失真是裁切不一致疊加顯示尺寸不一致造成的視覺不齊，並非單張圖片被拉伸。
- **修正**：統一裁切為`l=22000,r=22000`（22.0%/22.0%，移除雜散的t/b裁切）+ 統一顯示尺寸為`810000×1446429`EMU（取原10張平均寬809676 EMU取整，高依0.56裁切後比例反推，整體圖面尺寸與修正前接近，不影響版面分頁），10張全部套用相同值。
- 已直接套用至 `D:\Download\cvgip2025_SGA_english_v11.docx`（原檔修改），備份於同目錄 `..._before_fig8fix.docx`。

**組間距修正（同日續）**：使用者反饋「上下一組的話，組跟組之間距離太大」。根因：欄寬（`tblGrid`/每格`tcW`）原為1917~1918 twips，但統一後的圖片寬度僅約1276 twips，扣掉margin後每格仍有大量空白，造成相鄰兩組（欄）之間視覺間距過大。
修正：欄寬縮緊為1340 twips（=圖片寬1276 twips + 左右各30 twips margin）、5欄`tblGrid`與每格`tcW`同步更新、新增明確的`tcMar`（左右30 twips，原本未設，繼承表格樣式預設）、移除原`tblInd`（80 twips左偏移）改為表格置中對齊（`WD_TABLE_ALIGNMENT.CENTER`），避免變窄後的表格貼著左邊界不對稱。已套用，備份於 `..._before_fig8gap.docx`。
- **待辦**：中文版`cvgip2025_SGA_chinese_v11.docx`的FIG-8是否有同樣問題尚未檢查；PDF視覺最終確認尚未做（本次兩輪FIG-8修正皆只在python-docx結構層驗證）。

### §4.3反光嚴重程度分層分析 + v12版本誕生（2026-06-20續2）

- **背景**：使用者要求對「PSNR/SSIM/LPIPS為何偏低」給出有實證支持的解釋，而非未經驗證的推論。直接用`eval_results.csv`（491張逐圖數據）做了6個角度的實測分析（非空想）：
  1. 逐圖勝率：SGA單張贏Baseline比例PSNR21.6%/SSIM5.9%/LPIPS8.4%
  2. 按反光強度（用Reflection vs NonReflection逐像素差異當代理指標）分四等分：PSNR差距Q1−2.494dB→Q2−1.249→Q3−0.475dB（縮小5倍）→Q4−0.650dB；SSIM同方向但弱；LPIPS無此趨勢
  3. 平均值−1.214dB vs 中位數−0.799dB，確認被尾端拖低
  4. dPSNR與Baseline自身PSNR相關係數−0.584（比反光強度代理指標更強）
  5. 尾端分解：最差20%影像平均−3.743dB，其餘80%僅−0.583dB
  6. 交叉驗證：「最差20%」與「反光最輕Q1」有59.2%重疊，確認是同一群圖驅動兩個發現
  - 另測試「輸入解析度256px是否為瓶頸」：訓練512×512變體（SGA-512, ep360），PSNR/SSIM僅+0.14dB/+0.02小幅改善，LPIPS反而變差（0.217→0.238），且batch size被迫減半（8→4）。結論：解析度不是主因，未繼續投入。
  - 原先猜測「SGA壓低平滑/低梯度區域訊號拖累畫質」的假設經相關性檢驗（corr(梯度量,dSSIM)=+0.38、corr(平滑比例,dSSIM)=−0.40）**方向相反，已排除**，不寫入論文。
- **核心敘事**：SGA固定結構先驗在反光越嚴重的圖片上優勢越明顯，整份SIRR測試集平均值被大量「反光很輕、Baseline本來就能處理好」的圖片拖低；此效應在PSNR最清楚、SSIM較弱、LPIPS未觀察到，誠實標註不是全指標通用的解釋。
- **同時套用之前已核准但尚未寫入的措辭修正**：B1/B2/B3/B4/C4/Fig.6/G4（Abstract+Conclusion）/248→1951、248→491數字修正——詳見本文件2026-06-13/06-14過度推論覆盤章節，本次已實際套用文字。
- **產出新版本**：依使用者指示「寫新VERSION不要取代到原本VERSION」——
  - `D:\Download\cvgip2025_SGA_english_v12.docx`（英文，新增§4.3兩段+§5.1銜接句+§5.2第5點+上述措辭修正全部套用）
  - `D:\Download\cvgip2025_SGA_chinese_v12.docx`（中文，內容對應同步翻譯套用）
  - v11（中英文）兩份原檔皆未變動，編輯時的中間備份（`..._before_narrative_pass.docx`）保留於同目錄
- **意外發現並修正的既有bug**：v11/v12英文版的Table 1與Table 2**整個是中文沒翻譯**（"方法"、"條件"、"準確率 (%)"、"說明"、"原始影像（含反光）"、"699 張中 40 張未能成功辨識"、"Pix2Pix + SGA（本文）"等），已於v12改為英文（Method/Condition/Accuracy (%)/Description/Original image (with reflection)/40 of 699 not recognized/Pix2Pix + SGA (Ours)等）。已掃描全文確認除作者中文姓名（刻意保留）外無其他中文殘留。
- **視覺驗證**：用`export_english_pdf.py`改的腳本（Word COM + PyMuPDF）將v12轉成PDF（9頁）逐頁渲染確認，§4.3新段落、§5.1/§5.2新增點、Conclusion新措辭、Fig.6新措辭、Abstract新措辭皆正確渲染、無破版。中文v12尚未做PDF視覺驗證。
- **引用追蹤表更正**：發現C3（[22] MRI案例邏輯跳躍問題）其實已在2026-06-15續的內容縮減回合修正過，本檔案表格先前未同步更新狀態，已在上方表格修正。C1/C2仍需實際走`/cite-papers`查證原文，尚未處理。G1/G2/G3因相關章節（Scope of Applicability/Computational Efficiency）已在v11縮減時整節刪除，標記為不適用。
- **待辦**：
  1. 中文v12 PDF視覺驗證（尚未做，僅英文做過）
  2. ~~C1/C2citation查證~~ ✅ 已完成（見下方新章節）
  3. ~~D2/D3兩處過度confirmatory斷言~~ ✅ 已完成（見下方新章節）
  4. ~~推送v12（中英文）+ matherial v2腳本/圖檔 + 引用追蹤文件至GitHub~~ ✅ 已完成（commit 4a71452）

### C1/C2/D2/D3查證與修正 + 全文可讀性重寫 + v13誕生（2026-06-20續3）

- **C1查證**：實際讀`09_survey_yang2025.pdf`（K. Yang et al. SIRR Survey）§6.1全文，原句「even models that perform well on public datasets to degrade substantially in target-scene deployment [1]」對應原文實為「Without such comprehensive datasets, model evaluation remains limited and often unreliable when deploying into the real world」——原文講的是「資料集不足→評估不可靠」，不是「已驗證的大幅退化」，**確認過度引申**。已改為「This is a recognized limitation in current SIRR research: without datasets that comprehensively cover diverse real-world reflective surfaces and lighting conditions, model evaluation remains unreliable when deployed to new scenes [1]」，貼合原文語意。
- **C2查證**：實際讀`03_encoder_decoder_chi2018.pdf`（Chi et al. 2018）§4.1，確認「downsampling前處理risks losing structural detail」（段48用法）**有確實支撐**（原文明確用skip connection解決detail loss問題，不需改）；但段16句尾「enabling pixel-level distinction...[5]」是本文自己對SGA的描述，掛引用會誤導讀者以為Chi et al.驗證過此能力，**確認屬引用位置誤導**，已將句尾「[5]」移除。
- **D2/D3修正**：
  - D2（Fig.5討論）：「the most direct...evidence that the Sobel prior **lets**...transfer」→「providing qualitative evidence **consistent with** the Sobel prior's...attention transferring」，confirmatory語氣下修為描述性語氣。
  - D3（§5.1，Fig.7訓練動態）：「**indicating that** adding SGA **does not affect** the stability」→「**suggesting that** incorporating SGA **does not visibly destabilize**...**in this run**」，避免僅憑單一模型自身loss曲線就下「不影響穩定性」的比較性結論。
- **全文可讀性重寫**：使用者反饋「整個主文版措辭都好生硬、看不是很懂」。診斷根因：(1) 翻譯腔——中文學術寫作習慣把多個子句用冒號/分號塞進一句，逐句直譯成英文後句子過長過密；(2) 本次session陸續疊加的保留語氣詞（we conjecture/may/suggests/consistent with等）在已經很長的句子裡進一步增加負擔。修正做法：逐段拆解長句（多數從1句拆成2-4句）、精簡堆疊的保留語氣（每句最多保留一個）、確保「主張先講、證據再講」順序清楚。**全文約40個段落**逐一重寫，技術內容、數字、引用、claim方向**完全不變**，純粹改善句子結構與長度。
- **產出新版本**：`D:\Download\cvgip2025_SGA_english_v13.docx`（依使用者要求「記得改出新版本」，v12原檔未變動）。
- **視覺驗證**：`export_v13_pdf.py`（Word COM+PyMuPDF）轉PDF，**仍9頁**（重寫後長度大致持平，未變成10頁），逐頁確認C1/C2/D2/D3四處修正與全文重寫內容皆正確渲染、無破版、無遺漏citation。
### 中文版同步重寫 + v13誕生（2026-06-20續4）

- 使用者要求「中文版也做」——將C1/C2/D2/D3修正（同英文版邏輯，對應中文段落13/16/150/188）+ 全文可讀性重寫，同步套用至中文版。
- 中文v12複製為`D:\Download\cvgip2025_SGA_chinese_v13.docx`（v12未變動），逐段比對英文v13的拆句邏輯，以中文語感重新拆解約30個段落（中文學術寫作同樣慣用冒號/分號塞多個子句，套用同樣的拆句處理）。技術內容、數字、引用、claim方向**完全不變**。
- 同時修正段18遺漏的「domain-agnostic」標籤殘留（跟英文版contribution(1)同一問題，先前中文v12修正時也漏掉這處）。
- **視覺驗證**：`export_chinese_v13_pdf.py`轉PDF，**8頁**（與v12頁數一致，內容無流失），逐頁確認C1/D2/D3修正、§4.3分層段落、Conclusion皆正確渲染、無破版、無亂碼。
### 縮排房規廢除——全文統一縮排（2026-06-20續5）

- 使用者第三次反饋「每個TITLE下面的第一段都沒有空開頭行」。雖然先前已解釋過這是刻意房規（22/24章節第一段刻意不縮排），但使用者持續注意到此現象、顯然不符合預期，故詢問是否要直接廢除房規、全文統一縮排。**使用者確認：是，全部段都加縮排**。
- 已對英文v13（24處）與中文v13（23處）標題後第一段全部補上`firstLine=360`縮排，排除標題本身、References條目、標題頁author/email區塊（維持原有特殊縮排）。
- **視覺驗證**：兩份PDF重新轉出，頁數不變（英文9頁、中文8頁），Abstract與每個章節首段均確認已套用縮排。
- **待辦**：
  1. ~~v13（中英文，含本次縮排統一）尚未推送GitHub~~ ✅ 已完成（commit 9750f0d）
  2. v11/v12/v13三個版本並存，需使用者最終決定哪個是投稿版本

### Fig.5換成Baseline/SGA/Original三排對比圖 + v14誕生（2026-06-20續6）

- 使用者指定8張博物館測試圖（IMG_8093/8281/8395/8531/8533/8469/8537/8587，其中8281取代了原本誤點的8280），要求整理成「上排Original／中排Baseline／下排SGA(Ours)」三排對比格，白底。
- 圖片來源：
  - Original：`GAN_Test\Dataset\Test\jpg\IMG_xxxx.jpg`
  - Baseline：`paper\_downstream_temp\baseline_jpg\IMG_xxxx.png`
  - SGA：`paper\_downstream_temp\sga_jpg\IMG_xxxx.png`
  （此三個資料夾即前次session確認過「Baseline Pix2Pix確實跑過博物館去反光」的651張產出之一部分）
- **發現並修正一個構圖bug**：原圖（如IMG_8281，3024×4032直幅）若用「置中裁切成正方形」會跟Baseline/SGA輸出（GAN前處理用`data_loader.py`的`resize()`直接整張壓扁成正方形、不裁切）框出不同的畫面範圍，導致Original跟Baseline/SGA看起來像不同張照片。修正為Original也用「直接壓扁resize」取代「置中裁切」，與GAN實際看到的畫面範圍一致。
- 使用者確認此對比圖滿意後，指示「用這張圖換目前圖五」——已將此圖換入英文+中文v14的Fig.5位置（原圖：`image10.png`/`visual_comparison.png`，僅2排Original/SGA），同步將圖說從「Top/bottom 2排」改為「Top/middle/bottom 3排」說明加入Baseline。長寬比依新圖（1990×760, ratio 2.618）重新計算display extent，保持原有寬度、按比例調整高度，避免變形。
- 產出新版本：`cvgip2025_SGA_english_v14.docx`、`cvgip2025_SGA_chinese_v14.docx`（v13未變動）。
- **視覺驗證**：兩份PDF重新轉出，頁數不變（英文9頁、中文8頁），Fig.5（第6頁）新圖+新圖說皆正確渲染。
- **待辦**：
  1. ~~v14（中英文）尚未推送GitHub~~ ✅ 已完成（commit 0d75df4）
  2. ~~§4.4正文討論段未提及Baseline~~ ✅ 已完成（見下方新章節）
  3. v11/v12/v13/v14四個版本並存，需使用者最終決定哪個是投稿版本

### Fig.5放大30% + 正文補充Baseline效果有限的討論（2026-06-20續7）

- 使用者反饋「fig5太小，放大30%」+ 要求§4.4正文討論段補充「baseline的反光去除效果明顯不如加入SGA」的說明。
- 直接在v14（不另開新版本，因v14的Fig.5尚在同一輪反饋迭代中未定稿）：
  - 將Fig.5的`wp:extent`/`a:ext`（cx、cy）等比放大1.3倍（4.32in×1.97in → 5.62in×2.15in），確認單欄寬度足夠容納，未跨版。
  - 英文段148、中文段150改寫，新增「Baseline雖同樣嘗試抑制反光，但效果有限、仍見明顯殘留反光，解析度與細節亦有下降；相較之下SGA反光區明顯減弱、結構保留更完整」的對比論述。
- **視覺驗證**：兩份PDF重新轉出，頁數不變（英文9頁、中文8頁），Fig.5（第6頁）放大後排版正常、新增文字正確渲染無破版。
- **待辦**：v11/v12/v13/v14四個版本並存，需使用者最終決定哪個是投稿版本
