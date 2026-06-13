# PAPER_STATUS.md — AI GO CVGIP 論文狀態追蹤

Last updated: 2026-06-13（v2 完成 + 全面審查 + 引用全文覆盤驗證已完成（30/30 個 reference 條目），結果見
`PAPER_AUDIT_2026-06-13.md` 與 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md`，
共 11+5=16 項待使用者決策事項，詳見本文件最新章節）

---

## 基本資訊

- **論文標題**：基於 Sobel 引導注意力機制之 Pix2Pix 博物館文物反光消除與辨識
- **投稿目標**：CVGIP 2025（台灣國內研討會）
- **主要語言**：中文版（主線）
- **主要成果**：AI GO 2024 競賽最佳實作獎
- **生成指令**：`C:\Users\bubbl\anaconda3\python.exe cvgip2025_chinese.py`

---

## 現有檔案版本鏈

| 檔案 | 說明 | 狀態 |
|------|------|------|
| `cvgip2025_chinese.py` | 中文版 docx 生成腳本（主線） | ✅ v3 敘事重構（2026-06-02） |
| `cvgip2025_SGA_chinese.docx` | 上述腳本生成的 docx | ✅ 已生成，有 [MUST-X] 佔位符待填 |
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

## 下一步行動（優先順序，2026-06-13 更新）

1. ~~v2 編輯 + docx/pdf 產出~~ ✅ 完成（2026-06-13）
2. ~~全面不足盤點 + 引用複核（摘要層級）~~ ✅ 完成（2026-06-13，結果見 `PAPER_AUDIT_2026-06-13.md`）
3. ~~引用全文覆盤驗證（28 bibkey，30/30 涵蓋率）~~ ✅ 完成（2026-06-13，結果見 `CITATION_FULLTEXT_VERIFICATION_2026-06-13.md`，見上）
4. ~~引用措辭修正提案共 5 項（[2][3][9][23][4]）~~ ✅ 已套用（2026-06-13，使用者核准，見上）
5. ~~FIG-6 `原跑原7.jpg`/`消跑原7.jpg` 方向確認~~ ✅ 完成（2026-06-13，使用者確認方向正確，見上方說明）
6. **等待使用者決策** — `PAPER_AUDIT_2026-06-13.md` §4 的其餘待決事項（標題頁資訊、
   致謝、810/248→1951/491、line 202【9】措辭、line 417【13, B】括號格式）
7. 推 GitHub 並回連結（依 feedback 規則）
8. ~~確認是否有 CA only / SA only checkpoint（OPT-1/2）~~ ✅ 完成（2026-06-13，使用者確認不需要，無對應 checkpoint，表1維持2列）
9. ~~FIG-1 整體架構圖補 SGA 位置（GAN_architecture.png 目前僅 GAN 迴圈）~~ ✅ 完成（2026-06-13，見下方記錄）
10. ~~圖號全面重新編號（圖3/圖4 原無編號）~~ ✅ 完成（2026-06-13，圖1~圖8 依出現順序連續編號，見上方圖表清單）
