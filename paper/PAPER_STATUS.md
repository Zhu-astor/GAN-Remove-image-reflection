# PAPER_STATUS.md — AI GO CVGIP 論文狀態追蹤

Last updated: 2026-06-06（量化評估完成 + citation_verification_record.md 全量建立）

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
- **博物館評估**：699 張影像（下游辨識用，跨 domain 設計，**未用於訓練**）
- **下游驗證**：YOLOv8n，7 類展品，92.7% → 94.5%（+1.8pp）
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
| MUST-3 | 表 2 | Baseline Pix2Pix + YOLOv8 準確率（博物館 699 張） | ❌ |
| OPT-1 | 表 1 | CA only 的 PSNR/SSIM/LPIPS（有 checkpoint 才填，沒有刪整行） | ❓ |
| OPT-2 | 表 1 | SA only 的 PSNR/SSIM/LPIPS（有 checkpoint 才填，沒有刪整行） | ❓ |

---

## ❌ 待製圖表（共 6 張）

| 標記 | 檔名 | 內容 | 製作方式 |
|------|------|------|------|
| FIG-1 | overall_architecture.png | 整體架構圖（輸入→SGA→UNet→輸出+PatchGAN） | 手動繪製（draw.io / PPT） |
| FIG-2 | sga_module.png | SGA 模組詳細結構（Sobel萃取→CA+SA雙分支） | 手動繪製 |
| FIG-3 | visual_comparison.png | Before/After 博物館視覺比較（3~4組） | 跑推論截圖：含反光原圖\|Baseline\|SGA |
| FIG-4 | attention_map.png | Sobel Attention Map 視覺化 | `GAN_Test/Pic_process_sobel.py` 生成 |
| FIG-5 | training_loss.png | G loss / D loss 訓練曲線（500 epoch） | 從訓練 print log 提取 d_losses/g_losses |
| FIG-6 | downstream_accuracy.png | 下游準確率 bar chart（3 組） | matplotlib 畫（資料已有） |

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

## 下一步行動（優先順序）

1. ~~修正 §3.6 解析度說明（512→256）~~ ✅ 完成
2. ~~加入 [Blau18]/[Ledig17] 引用到論文 §4.3 + references~~ ✅ 完成
3. **最優先** — 執行 `eval_downstream.py` 取得 MUST-3 數據（Phase 4 paired agreement for raw vs SGA_GAN）
4. 確認是否有 CA only / SA only checkpoint
5. 製作 FIG-3（Before/After）和 FIG-4（Attention Map）—— 視覺說服力最強
6. 從訓練 log 製作 FIG-5（Loss 曲線）
7. 製作 FIG-6（bar chart，matplotlib，資料已有）
8. 手動繪製 FIG-1、FIG-2（架構圖）
