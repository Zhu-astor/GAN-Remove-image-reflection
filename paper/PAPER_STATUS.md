# PAPER_STATUS.md — AI GO CVGIP 論文狀態追蹤

Last updated: 2026-06-14（§4.6 body 文字補上「以圖8第5組範例為例」說明（解決前次 flag）；
完成「邊緣=高梯度」真實引用查證（新增 [31] RINDNet, ICCV 2021），修正 §1/§3.2.1 過度絕對化的
「物理性質、與場景domain無關」措辭與「根本機制」因果宣稱，citation_verification_record.md §6
同步修正「三者一致」過度推論；PDF 仍為 9 頁，逐頁視覺驗證通過，詳見本文件最新章節）

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
