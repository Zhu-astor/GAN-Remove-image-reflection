# CVGIP 2025 論文全面審查報告 — 2026-06-13

> 範圍：`cvgip2025_chinese.py`（827 行，v2 版本，commit `34c5547` 之後）
> 對應使用者指示：「幫我分析與盤點該篇論文的不足，另外請再次全面檢查論文引用是否正確，
> 記得寫引用處所、引用內容、引用證據的檔案下來」
> 方法：依 CLAUDE.md §5.0b / §5.2 — 每條引用對照 `citation_verification_record.md`
> （2026-06-06 建立）之既有驗證結果，並對「具體數值/方向性主張」的引用標記是否需要
> 進一步讀取原文全文才能達到驗證標準。本報告**不直接修改 `cvgip2025_chinese.py`**，
> 所有建議修正均列為「待使用者決策事項」（§4），符合 §5.2 對學術文件修改前需先提案的要求。

---

## 1. 不足盤點（Shortcomings Inventory）

### 1.1 標題頁佔位符（lines 165-167）

```python
p(doc, '1學生姓名（中文名），以及 1,*指導教授姓名（中文名）\n'
       '1 [系所名稱]，[學校名稱]，[城市]，臺灣\n'
       'E-mail：[email@university.edu.tw]',
  align=WD_ALIGN_PARAGRAPH.CENTER, size=10, after=6)
```

- **問題**：學生姓名、指導教授姓名、系所、學校、城市、Email 均為佔位符文字，尚未填入真實資訊。
- **影響**：若直接以目前 docx/pdf 投稿，標題頁會直接顯示 `[系所名稱]`、`[email@university.edu.tw]` 等字樣。
- **所需資訊**（須使用者提供）：學生姓名、指導教授姓名、系所名稱、學校名稱、城市、聯絡 Email。

### 1.2 FIG-1 缺失 + 正文懸空引用（lines 342, 350）

```python
# line 342（§3.1 正文）
'...整體流程如圖 1 所示：給定含反光的輸入影像 ...'

# line 350（caption，無對應 fig() 呼叫）
caption(doc, '圖 1. 整體架構示意圖。[FIG-1 — 請插入 overall_architecture.png]')
```

- **問題**：
  1. §3.1 正文（line 342）明確寫「如圖 1 所示」，但 line 350 只有 caption 佔位文字，**沒有對應的 `fig()` 呼叫**——目前輸出的 docx/pdf 中「圖 1」實際上不存在任何圖片，只有一行說明文字。
  2. 檢查 `D:\Contest\AI GO\matherial\` 目錄，確認 `overall_architecture.png` **不存在**。目錄中現有 `GAN_architecture.png`（37,785 bytes，2026-06-10 產生）以及 `draw_sga_architecture.py`、`draw_downstream_accuracy.py` 兩支繪圖腳本，但沒有「整體架構示意圖（含 SGA 模組位置）」的成品圖。
- **影響**：讀者讀到「如圖 1 所示」卻找不到圖 1，是審稿時容易被指出的明顯缺陷。
- **所需資訊/動作**（須使用者決策）：
  - 是否使用現有 `GAN_architecture.png`（但 PAPER_STATUS.md 記錄此圖「僅畫 GAN 迴圈，未含 SGA」，與正文描述「SGA 插入於 Encoder Block 0 之前」不完全對應）？
  - 或需另外繪製一張「Pix2Pix U-Net + SGA 插入位置」的整體架構圖（可能需要新的繪圖腳本）？

### 1.3 致謝佔位符（line 735）

```python
p(doc, '感謝【博物館/合作單位名稱】提供展品影像供跨場景評估使用。'
       '本研究於 AI GO 2024 競賽期間完成，獲競賽評審委員會最佳實作獎肯定。')
```

- **問題**：`【博物館/合作單位名稱】` 為佔位符，目前文中已多次提到「國立歷史博物館」（見 PAPER_STATUS.md 引用之原始 PDF：「我使用國立歷史博物館的展品數據集...」）。
- **所需資訊**（須使用者確認）：致謝對象是否即為「國立歷史博物館」？是否需要列出合作單位的正式全名？

### 1.4 訓練/測試集數量不一致：「810 對／248 對」vs.「491 對」

依 CLAUDE.md §4.1 六步驟流程（COLLECT → STRUCTURE → HYPOTHESIZE → VERIFY → FIX → VALIDATE）整理如下：

#### Step 1-2：COLLECT + STRUCTURE — 證據列表

| # | 位置 | 原文片段 | 數字 |
|---|------|---------|------|
| E1 | `cvgip2025_chinese.py` line 486-487（§4.1.1） | 「本文將四個資料集合併後進行隨機分割，80% 作為訓練集（**810 對**），20% 作為測試集（**248 對**）。」 | 810 / 248 |
| E2 | `cvgip2025_chinese.py` line 518（§4.2） | 「（1）影像復原指標（在公開 SIRR 測試集上，**248 對**，含 ground truth）」 | 248 |
| E3 | `cvgip2025_chinese.py` line 528（§4.3） | 「為驗證 SGA 模組的有效性，本文在公開 SIRR 測試集（**491 對**）上比較兩種設定」 | 491 |
| E4 | `cvgip2025_chinese.py` line 534（表 1 caption） | 「表 1. 公開 SIRR 測試集（**491 對**）消融實驗結果。」 | 491 |
| E5 | `cvgip2025_chinese.py` line 679（§5.3） | 「在訓練資料規模受限（**810 對**）的情境下...」 | 810 |
| E6 | `cvgip2025_chinese.py` line 727（§6 Conclusion） | 「實驗結果顯示，以公開 SIRR 資料集（**810 對**）訓練的模型...」 | 810 |
| E7 | `PAPER_STATUS.md` line 81-82（MUST-1/MUST-2 記錄） | 「表1｜Baseline Pix2Pix 的 PSNR/SSIM/LPIPS（公開 SIRR 測試集，**491 對**）｜✅ 23.896/0.8706/0.1630」「同上 Pix2Pix+SGA｜✅ 22.682/0.8192/0.2178」 | 491 |
| E8 | `eval_metrics.py` line 66-67 | `REFLECTION_DIR = r"D:\Contest\AI GO\github\Dataset2\Test\Reflection"`<br>`GT_DIR = r"D:\Contest\AI GO\github\Dataset2\Test\NonReflection"`（`n_total = len(reflection_paths)`，用於表 1 數值來源） | （讀取實際資料夾） |
| E9 | 實測：`D:\Contest\AI GO\github\Dataset2\Train\Reflection` | 檔案數量 = **1951** | 1951 |
| E10 | 實測：`D:\Contest\AI GO\github\Dataset2\Test\Reflection` / `NonReflection` | 檔案數量 = **491 / 491** | 491 |

#### Step 3：HYPOTHESIZE

| 假設 | 機制 | 支持證據 | 反對證據 |
|------|------|---------|---------|
| H1：「491」為正確數字，「810/248」為舊版殘留 | §4.2/§4.3/Table1 的「491」與 eval_metrics.py 實際讀取的 `Dataset2/Test/Reflection` 資料夾（491 個檔案）完全吻合；PAPER_STATUS.md 的 MUST-1/2 記錄（2026-06-06，當時已執行 eval_metrics.py）也記為 491 | E3, E4, E7, E8, E10 | — |
| H2：「810/248」描述的是「80%/20% 隨機分割」這一框架本身，框架正確但**絕對數字**未隨資料集更新而修正 | 1951 / (1951+491) = **79.9% ≈ 80%**；491 / (1951+491) = **20.1% ≈ 20%**——「80/20 分割」的描述方向是對的，只是 810/248 這兩個具體數字與目前 `Dataset2` 實際檔案數（1951 訓練 / 491 測試）不符 | E9, E10（80/20 框架在 1951/491 上幾乎精確成立） | — |
| H3：810/248 來自另一個與目前 `Dataset2` 不同的舊資料集版本 | 810+248=1058，與 1951+491=2442 差距甚大，不是同一批資料的子集關係 | E1 與 E9/E10 數字差距過大 | 無法排除，但即使如此，**目前 `Dataset2` 資料夾才是 eval_metrics.py 實際使用、產出表1數據的資料來源**，因此 810/248 即便曾經正確，對「目前這篇論文所報告的實驗」而言已是過時數字 |

#### Step 4：VERIFY（已執行）

- 直接列出 `D:\Contest\AI GO\github\Dataset2\Train\Reflection`（1951）與 `Dataset2\Test\Reflection`／`NonReflection`（491/491）的檔案數量 → 證據 E9/E10 已確認。
- `eval_metrics.py` 的 `REFLECTION_DIR`/`GT_DIR` 指向的正是 `Dataset2\Test\Reflection`／`NonReflection`，且其 `n_total = len(reflection_paths)` 即為表 1 數值的樣本數來源 → 證據 E8 已確認「491」是表1數據實際對應的樣本數。

#### Step 5：結論（FIX 建議，尚未套用，列為提案）

- **「491」（E3/E4/E7/E8/E10）為正確、且與實際資料夾及 eval_metrics.py 輸出一致，不需修改。**
- **「810/248」（E1/E2/E5/E6）與目前 `Dataset2` 資料夾的實際數量（1951/491）不符**，但其「80%/20% 隨機分割」框架描述（line 486-487：「80% 作為訓練集...20% 作為測試集...」）與 1951/491 的實際比例（79.9%/20.1%）幾乎完全吻合。
- **建議修正**：將 E1、E2、E5、E6 四處的「810」→「**1951**」、「248」→「**491**」，使全文數字一致，且與實際資料、eval_metrics.py 輸出、PAPER_STATUS.md 記錄三方吻合。「80%/20% 隨機分割」的敘述本身**不需更動**（比例描述仍然正確）。

#### Step 6：VALIDATE（修正後應確認）

- 全文 grep `810|248|491`，確認「810/248」不再出現、「1951/491」在 E1/E2/E5/E6/E3/E4 六處一致。
- 重新生成 docx/pdf 後，人工檢視 §4.1.1、§4.2、§4.3、Table 1、§5.3、§6 六處數字是否前後一致。

> ⚠️ **本項為提案，未經使用者確認前不會修改 `cvgip2025_chinese.py`**（CLAUDE.md §5.2）。

### 1.5 引用括號風格不一致（line 417）

```python
'SGA 模組之後接標準 Pix2Pix U-Net 生成器【13, B】。'
```

- **問題**：全文其餘多重引用均以「相鄰獨立全角括號」表示（例如 line 317「GCNet【26】統一 Non-local Networks【24】」），僅此處以「逗號合併於同一組全角括號內」（【13, B】）且混用數字鍵與字母鍵。
- **影響**：純格式一致性問題，不影響學術正確性，但建議統一為「【13】【B】」以符合全文風格。

---

## 2. 引用全面審查表（Citation Audit Table）

> 「驗證狀態」欄取自 `citation_verification_record.md`（2026-06-06 建立，本次審查逐筆核對方向是否與目前 .py 正文一致）。
> 「本次複核」欄為 2026-06-13 新增結果：✅ = 與既有記錄方向一致、未發現新問題；⚠️ = 發現需進一步處理的問題（詳見 §3）。

| 位置 (file:line) | Bibkey | 本文論述（節錄） | citation_verification_record.md 狀態 | 本次複核 |
|---|---|---|---|---|
| 202 | 【9】 | 「此一 domain gap 問題導致...在目標場景的實際部署中往往大幅退化【9】」 | ⚠️PARTIAL（arXiv摘要已讀，[9] SIRR Survey 2025） | ⚠️ 見 §3.4(a) |
| 210 | 【42】 | 「YOLOv8【42】在含反光影像上的辨識準確率僅為 92.7%」 | ⚠️PARTIAL（[42] YOLOv8第三方論文，官方無正式paper） | ✅ 此處僅作為「YOLOv8」方法名稱引用，與既有記錄一致 |
| 230 | 【23】 | 「Lu 等人【23】在醫學影像分割中展示了 Sobel 引導注意力的有效性」 | ⚠️PARTIAL（搜尋結果確認） | ✅ 與 [23] 角色描述一致（COVID病灶分割+Sobel邊緣特徵融合+CBAM-style attention） |
| 231 | 【21】 | 「CBAM【21】確立了通道—空間雙維注意力的互補優勢」 | ⚠️PARTIAL（[21] CBAM原始論文） | ✅ CBAM 論文核心主張，well-established |
| 235 | 【3】 | 「使模型在像素層面即對「展品結構」與「反光干擾」進行空間上的區分【3】」 | ⚠️PARTIAL（[3] Chi 2018, abstract-only） | ⚠️ 見 §3.2 |
| 258 | 【1】 | 「Fan 等人【1】提出 CEILNet，首次以級聯CNN架構在SIRR中引入邊緣資訊：邊緣預測網路（E-CNN）先估計物件邊緣圖，再由重建網路（I-CNN）以邊緣圖為輔助恢復傳輸層」 | ⚠️PARTIAL（[1] abstract: "exploits edge information through cascaded convolutional layers"） | ✅ 摘要確認「cascaded + edge information」方向一致；E-CNN/I-CNN 命名細節未在摘要bullet中逐字出現，但與「cascaded CNN + edge」方向相符，未發現矛盾 |
| 263 | 【2】 | 「Li 等人【2】提出 IBCLN...並建立 SIR² 真實場景配對資料集，是本文訓練所用資料集之一」 | ⚠️PARTIAL（[2] abstract bullet: 「建立真實場景配對資料集」） | ⚠️ 見 §3.1 |
| 265 | 【3】 | 「Chi 等人【3】深入分析了編碼器—解碼器架構的根本缺陷：連續下採樣操作不可逆地削弱高頻邊緣響應，使後續解碼器無法精確復原物件邊界，直接支撐了本文在編碼器前插入邊緣注意力的設計動機」 | ⚠️PARTIAL（[3] abstract-only，bullet 未提及此具體主張） | ⚠️ 見 §3.2（高優先） |
| 267 | 【4】 | 「Dong 等人【4】提出位置感知反光消除...證明空間注意力在 SIRR 任務中的有效性」 | ⚠️PARTIAL（[4] abstract: "probabilistic confidence map", "multi-scale Laplacian features"） | ✅ 方向一致 |
| 272 | 【6】 | 「DURRNet【6】採用演算法展開（algorithm unrolling）將迭代優化轉化為深度網路，具備理論可解釋性」 | ⚠️PARTIAL（[6] abstract: "model-based optimization using transform-based exclusion priors" + "deep unrolling architecture"） | ✅ 方向一致 |
| 274 | 【8】 | 「PromptRR【8】以擴散模型作為頻域提示生成器驅動 Transformer 網路，達最新 SOTA 水準」 | ⚠️PARTIAL（[8] abstract: "frequency prompt encoder + diffusion model", "outperforms SOTA"） | ✅ 方向一致 |
| 283 | 【19】 | 「生成對抗網路（GAN）【19】以生成器與判別器的對抗訓練學習數據分佈」 | ⚠️PARTIAL（[19] Goodfellow 原始GAN） | ✅ 教科書級事實，§5.0b「well-known fact」例外適用 |
| 284 | 【15】 | 「Mirza 與 Osindero【15】提出條件 GAN（cGAN），在生成器與判別器中同時加入條件向量」 | ⚠️PARTIAL（[15] cGAN原始論文） | ✅ 教科書級事實 |
| 286 | 【13】 | 「Isola 等人【13】實例化此範式為 Pix2Pix：U-Net 生成器搭配 PatchGAN 判別器，以 L1+cGAN 損失組合訓練」 | ✅CONFIRMED（[13] 核心架構基礎，已多次驗證） | ✅ |
| 290-291 | 【14】 | 「CycleGAN【14】引入循環一致性損失...然而 Zhu 等人【14】的原始比較實驗顯示，在配對資料充足的條件下，Pix2Pix 優於 CycleGAN 及其他無監督方法」 | ⚠️PARTIAL（[14] arXiv摘要已讀） | ⚠️ 見 §3.5 |
| 295 | 【17】 | 「Liu 等人【17】對 GAN 圖像合成的全面綜述確立了對抗訓練的廣泛有效性」 | ⚠️PARTIAL（[17] GAN綜述，背景性宣稱支撐） | ✅ 方向一致（綜述類引用，泛化敘述） |
| 300 | 【22】 | 「Hu 等人【22】提出 Squeeze-and-Excitation Network（SENet），以全局平均池化壓縮空間維度後...奠定了通道注意力在 CNN 中的地位」 | ⚠️PARTIAL（[22] SENet原始論文） | ✅ well-established |
| 305 | 【21】 | 「Woo 等人【21】在 SENet 的基礎上提出卷積塊注意力模組（CBAM），依序在通道和空間兩個維度推斷注意力圖...在分類與偵測任務的廣泛實驗中均優於 SENet」 | ⚠️PARTIAL（[21] CBAM原始論文） | ✅ CBAM論文核心主張 |
| 312-316 | 【23】 | 「Lu 等人【23】...以梯度幅度作為結構先驗驅動注意力機制，顯著提升分割邊界精度...此設計的成功**驗證了**固定 Sobel 梯度引導注意力在跨場景設定下的穩定性，是本文方法在 SIRR 場景中應用的最直接依據」 | ⚠️PARTIAL（[23] role: 「直接支撐Sobel+attention設計思路」，未涉及「跨場景穩定性」） | ⚠️ 見 §3.3（高優先） |
| 317 | 【26】【24】 | 「GCNet【26】統一 Non-local Networks【24】與 SENet 的結構分析，說明結合全局語境與通道校準優於任一單一機制，進一步支持本文雙分支設計」 | ⚠️PARTIAL（[26] abstract: "unifies NL + SE, GC block lightweight & effective global context"; [24] abstract: "weighted sum of features at all positions"） | ✅ 方向一致 |
| 323-324 | 【29】 | 「Xie 與 Tu【29】提出整體嵌套邊緣偵測（HED），以多尺度深度監督邊緣學習展示不同層次的邊緣特徵攜帶互補結構資訊，成為後續邊緣引導方法的重要基準」 | ⚠️PARTIAL（[29] abstract: "multi-scale and multi-level feature learning within unified framework", SOTA on BSD500/NYU Depth） | ✅ 方向一致 |
| 325-326 | 【31】 | 「Ji 等人【31】提出 DGNet，以物件梯度監督解耦紋理與語義特徵，其梯度引導特徵提煉的思路與本文類比」 | ⚠️PARTIAL（搜尋結果確認；[31] abstract: "object gradient supervision", "decouples context/texture encoder via gradient-induced transition"） | ✅ 方向一致 |
| 328-330 | 【37】 | 「Sharp U-Net【37】在 U-Net 的 skip connection 前加入銳化核...減少編碼器與解碼器特徵的語義不相似性」 | ⚠️PARTIAL（[37] Sharp U-Net原始論文） | ✅ 方向一致（依角色描述） |
| 331-333 | 【33】 | 「Li 與 Liu【33】在 MRI 超解析任務中引入梯度圖邊緣品質損失，強制模型學習邊緣結構細節——此類邊緣引導設計在醫學影像這一特殊 domain 中的成功，進一步支持本文以固定梯度先驗跨場景遷移的假設」 | ⚠️PARTIAL（[33] abstract: "L1 loss of SSIM and gradient map edge quality loss could force model to focus on edge/structure details"） | ⚠️ 見 §3.6（低優先，措辭已較緩和） |
| 341 | 【13】 | 「本文方法以 Pix2Pix【13】為骨幹架構，核心創新為在 U-Net 生成器的首個編碼器塊前插入 SGA 模組」 | ✅CONFIRMED | ✅ |
| 348 | 【3】 | 「...若讓網路先執行下採樣再補救，高頻邊緣資訊已不可逆損失【3】，且中間層特徵已摻雜 domain-specific 的語義信息，不利於跨場景遷移」 | ⚠️PARTIAL（同 line 265，abstract-only） | ⚠️ 見 §3.2（與 265 同一問題的第二次出現） |
| 417 | 【13, B】 | 「SGA 模組之後接標準 Pix2Pix U-Net 生成器【13, B】」 | ✅CONFIRMED（[13]）/ ⚠️PARTIAL（[B] U-Net原始論文，abstract已讀） | ✅ 方向正確；括號風格問題見 §1.5 |
| 426 | 【13】 | 「PatchGAN 判別器【13】對影像中的重疊 N×N 圖像塊分別判斷真偽」 | ✅CONFIRMED | ✅ |
| 443 | 【13】 | 「此配置與 Pix2Pix 原始設定一致【13】」（L1權重100、對抗損失MSE） | ✅CONFIRMED | ✅ |
| 470 | 【GAP-E】 | 「SIR²【GAP-E】：大規模真實場景配對反光資料集，涵蓋物件（Objects）、野外（Wild）與後處理合成（Postcard）三個子集」 | ⚠️PARTIAL（[GAP-E] Wan et al. 2017, "first captured SIRR dataset SIR2", 40 controlled+100 wild scenes） | ✅ 與 [GAP-E] 角色描述一致，是 SIR² 的正確出處 |
| 477 | 【ERRNET】 | 「ERRNET【ERRNET】：Wei 等人提出的配對資料集，資料涵蓋多種材質表面與光線條件下的反光場景」 | ⚠️PARTIAL（[ERRNET] 已更正為 Wei et al. CVPR 2019, github.com/Vandermode/ERRNet） | ✅ |
| 481-483 | 【RFC】 | 「RFC（Flash Reflection Removal）【RFC】：Lei 與 Chen 所提供的以閃光燈輔助拍攝的配對資料集」 | ⚠️PARTIAL（[RFC] abstract已讀） | ✅ |
| 519 | 【A】 | 「SSIM【A】（結構相似性，越高越好）」 | ✅CONFIRMED（[A] 標準SSIM論文） | ✅ |
| 520 | 【36】 | 「LPIPS【36】（學習感知距離，以預訓練 VGG 特徵計算，越低越好）」 | ✅CONFIRMED（[36] LPIPS，方向明確） | ✅ |
| 522 | 【42】 | 「（2）跨場景下游效益...YOLOv8【42】展品分類準確率」 | ⚠️PARTIAL | ✅ 同 line 210，方法名稱引用 |
| 544 | 【Blau18】 | 「Blau 與 Michaeli【Blau18】從理論層面證明，感知品質與失真指標之間存在根本性的取捨關係（perception-distortion tradeoff）」 | ✅CONFIRMED（2026-06-06 新增，ar5iv全文已讀，含直接引文） | ✅ |
| 547-548 | 【Ledig17】 | 「Ledig 等人【Ledig17】在影像超解析度任務中也實驗確認：「最小化 MSE 鼓勵模型輸出所有合理解的像素均值，導致結果趨於過度平滑」」 | ✅CONFIRMED（2026-06-06 新增，ar5iv全文已讀，含直接引文） | ✅ |
| 549 | 【13】 | 「本文所採用的 Pix2Pix L1 損失【13】具有相同的機制特性」 | ✅CONFIRMED | ✅ |
| 654-656 | 【29】 | 「若採用可學習的邊緣偵測器（如 HED【29】）作為注意力驅動信號...在跨場景應用時**可能**出現邊緣偵測器對目標場景紋理的語義誤判」 | ⚠️PARTIAL | ⚠️ 見 §3.6（低優先，已用「可能」措辭緩和） |
| 692 | 【9】 | 「第三，本文尚未系統驗證強烈動態反光（如戶外強日照）的消除效果【9】」 | ⚠️PARTIAL | ⚠️ 見 §3.4(b) |
| 707 | 【29】 | 「（2）以可學習邊緣偵測器（如 HED【29】）在多尺度補充固定 Sobel 核，研究其對跨場景泛化的影響與取捨」 | ⚠️PARTIAL | ✅ 未來工作方向描述，僅作方法名稱引用，風險低 |
| 711 | 【8】 | 「（4）將 SGA 整合至 Transformer 架構（如 PromptRR【8】），結合全局注意力與 Sobel 局部先驗的互補優勢」 | ⚠️PARTIAL | ✅ 未來工作方向描述，僅作方法名稱引用，風險低 |

**統計**：全文共 30 個 reference 條目，body text 中實際引用 24 個 bibkey（[1][2][3][4][6][8][9][13][14][15][17][19][21][22][23][24][26][29][31][33][36][37][42][A][B][GAP-E][RFC][ERRNET][Blau18][Ledig17]，共 47 處標記）。其中：
- ✅ 與既有記錄方向一致、本次未發現新問題：約 39 處
- ⚠️ 需進一步處理：8 處，對應 §3 的 6 個議題（[2]/[GAP-E] 歸屬、[3]×2、[23]、[9]×2、[14]、[33]、[29]×1）

---

## 3. 重點問題深入分析

### 3.1 【高優先】[2] IBCLN 與 SIR² 資料集歸屬疑似錯置（line 263）

**本文宣稱**（line 263）：
> 「Li 等人【2】提出 IBCLN，以卷積 LSTM 實現迭代漸進式的傳輸層與反射層交替提煉，並**建立 SIR² 真實場景配對資料集**，是本文訓練所用資料集之一。」

**證據**：

| 來源 | 內容 |
|---|---|
| `citation_verification_record.md` [2] IBCLN | 作者：C. Li, Y. Yang, K. He, S. Lin, J. E. Hopcroft，"Single Image Reflection Removal through Cascaded Refinement"，CVPR 2020。摘要bullet：「Iterative Boost Convolutional LSTM Network (IBCLN)」「iteratively refines transmission and reflection layer estimates」「建立真實場景配對資料集」 |
| `citation_verification_record.md` [GAP-E] | 作者：R. Wan, B. Shi, L.-Y. Duan, A.-H. Tan, A. C. Kot，"Benchmarking Single-Image Reflection Removal Algorithms"，ICCV 2017。摘要bullet：**"first captured Single-image Reflection Removal dataset 'SIR2'"**，40 controlled + 100 wild scenes，廣泛被後續 SIRR 研究引用為標準評估基準 |
| `cvgip2025_chinese.py` line 470 | 「SIR²【GAP-E】：大規模真實場景配對反光資料集...」——本文**已經**將「SIR²」正確歸屬給 [GAP-E]（Wan et al. 2017） |

**矛盾**：本文同時在兩處將「SIR²資料集」與兩個不同論文關聯——
- line 263：[2]（Li et al., IBCLN, 2020）「建立 SIR² 真實場景配對資料集」
- line 470：[GAP-E]（Wan et al., 2017）才是「first captured...SIR2」資料集的來源

[2] 摘要中確實提到「建立真實場景配對資料集」，但根據 [GAP-E] 的驗證記錄，「SIR²」這個**資料集名稱與出處**屬於 Wan et al. 2017（時間也早於 IBCLN 2020），IBCLN 論文所建立的應是**另一個（非SIR²命名）的真實場景資料集**。

**建議修正方向（提案，待使用者確認）**：
- 將 line 263 的「並建立 SIR² 真實場景配對資料集」改為較中性的描述，例如「並提供額外的真實場景配對訓練資料」，**移除「SIR²」這個具體資料集名稱**，避免與 line 470（[GAP-E]）的 SIR² 歸屬產生衝突。
- 若使用者希望保留「[2] 建立了某個資料集」這一描述，建議的更精確寫法需要讀取 [2] 全文確認其資料集的實際名稱（非 SIR²）。

### 3.2 【高優先】[3] 「編碼器-解碼器架構根本缺陷」具體技術宣稱缺乏全文驗證（lines 265, 235, 348）

**本文宣稱**（line 265，是本文 SGA 模組「插入位置」這一核心設計決策的主要論證來源）：
> 「Chi 等人【3】深入分析了編碼器—解碼器架構的根本缺陷：**連續下採樣操作不可逆地削弱高頻邊緣響應，使後續解碼器無法精確復原物件邊界**，直接支撐了本文在編碼器前插入邊緣注意力的設計動機。」

此宣稱在 line 348（§3.1）以幾乎相同文字再次出現：「...若讓網路先執行下採樣再補救，高頻邊緣資訊已不可逆損失【3】...」；line 235（§1 contributions）亦以此為前提。

**證據**：

| 來源 | 內容 |
|---|---|
| `citation_verification_record.md` [3] | 作者：Z. Chi, X. Wu, X. Shu, J. Gu，"Single Image Reflection Removal Using Deep Encoder-Decoder Network"，arXiv:1802.00094, 2018。驗證狀態：⚠️PARTIAL（arXiv 摘要已讀）。摘要 bullet 僅有：「deep convolutional encoder-decoder method to remove reflection」「synthetic training dataset by modeling physical reflection formation」「significantly outperforms the other tested SOTA techniques」 |

**問題**：[3] 的摘要層級驗證內容中，**沒有任何一條 bullet 直接對應「連續下採樣不可逆削弱高頻邊緣響應、解碼器無法精確復原物件邊界」這一具體技術論斷**。本文卻將此具體論斷作為「SGA 為何插入於 Encoder Block 0 之前（而非中間層）」這一核心架構設計決策的**直接依據**，且重複出現 2 次（line 265, 348）。

依 CLAUDE.md §5.0b：「任何數值性、方向性、或領域特定的具體主張」都需要讀取原文確認，不能僅憑摘要或記憶推論。此處的「下採樣不可逆損失高頻邊緣資訊」雖然在影像處理領域是一個**普遍被接受的現象**（多尺度/U-Net文獻中常見），但「[3] 對此進行了深入分析」這一**歸屬到特定論文**的具體陳述，目前僅有摘要層級的 ⚠️PARTIAL 驗證，未達到 §5.0b 對「歸因到特定論文的具體技術論斷」的驗證標準。

**建議處理方向（提案，待使用者確認）**，二選一：
1. **讀取 [3] 全文**（若 `D:\Contest\AI GO\paper\NewRefs\` 或 arXiv 1802.00094 PDF 可取得），確認其是否確實包含「連續下採樣不可逆損失高頻邊緣資訊→解碼器無法復原物件邊界」這一分析，補充至 `citation_verification_record.md`。
2. 若不執行全文驗證，則將此論斷改為**作者自身的設計推論**（移除對 [3] 的具體歸因，或將「[3] 深入分析了...」改為更弱的措辭，如「此現象與 U-Net 類架構中下採樣對高頻資訊的影響已被廣泛討論【3】相關」），降低對 [3] 的具體歸因強度。

### 3.3 【高優先】[23] "驗證了...跨場景設定下的穩定性" 過度推論（lines 312-316）

**本文宣稱**：
> 「Lu 等人【23】在醫學影像分割任務中提出以 Sobel 算子引導的多尺度注意力網路，以梯度幅度作為結構先驗驅動注意力機制，顯著提升分割邊界精度。醫學影像與自然場景之間同樣存在顯著的 domain gap，**此設計的成功驗證了固定 Sobel 梯度引導注意力在跨場景設定下的穩定性**，是本文方法在 SIRR 場景中應用的最直接依據。」

**證據**：

| 來源 | 內容 |
|---|---|
| `citation_verification_record.md` [23] | F. Lu et al., "Multi-Attention Segmentation Networks Combined with the Sobel Operator for Medical Images", Sensors 2023。角色：「直接支撐『Sobel 運算子結合注意力機制』這一設計思路」。摘要bullet：「edge feature fusion module with Sobel operator」「self-attention channel attention + spatial linear attention」「應用於 COVID-19 病灶分割」。**[23] 本身並未做任何「跨場景/跨domain穩定性」的測試或宣稱**——它是在單一醫療影像 domain 內的分割任務。 |

**問題**：「此設計的成功**驗證了**...在跨場景設定下的穩定性」——「驗證了」（proven/validated）是強烈的因果/實證用語，但 [23] 本身：
1. 沒有做跨 domain（例如「同一模型訓練於A domain、測試於B domain」）的實驗；
2. 沒有對「Sobel 引導注意力的跨場景穩定性」做出任何宣稱。

本文這裡的論證實際上是**作者自己的類比推論**（「醫學影像也有 domain gap，[23] 在醫學影像中用 Sobel+attention 成功了，所以類似設計在 SIRR 跨場景中也應該穩定」），但用詞「驗證了」把「作者的類比推論」包裝成「[23] 已經證明的事實」，這正是 CLAUDE.md §5.2 所指「將邏輯推論/領域直覺替代引用證據」的典型模式。

**建議修正方向（提案，待使用者確認）**：將「此設計的成功**驗證了**固定 Sobel 梯度引導注意力在跨場景設定下的穩定性」改為作者自身推論的措辭，例如：
> 「醫學影像與自然場景之間同樣存在顯著的 domain gap，**【23】在此 domain gap 下仍能以 Sobel 引導注意力取得分割邊界精度提升，為本文「固定 Sobel 先驗具跨場景穩定性」的設計假設提供了類比性的支持**，是本文方法在 SIRR 場景中應用的重要參考依據。」

（將「驗證了...穩定性（事實陳述）」→「為...假設提供類比性支持（推論陳述）」，移除「最直接依據」的絕對化用語。）

### 3.4 [9] SIRR Survey 2025 — 兩處引用方向問題

**(a) line 202（§1 Introduction，中優先）**：
> 「此一 domain gap 問題導致即便在公開資料集上訓練效果良好的模型，**在目標場景的實際部署中往往大幅退化**【9】。」

`citation_verification_record.md` 中 [9] 的摘要 bullet 為：「reflection is quite common in digital images, posing significant challenges」「涵蓋 ICCV/ECCV/CVPR/NeurIPS 頂會方法」「包含 single-stage 與 two-stage 方法比較」「rapidly evolving research area」——這些 bullet **均未直接對應「跨域部署時模型大幅退化」這一具體方向性主張**。

此一「domain gap → 部署時退化」的說法在深度學習文獻中相當常見（屬於泛化能力討論的常見主題），但依 §5.0b，**具體方向性主張仍需驗證**而非僅憑摘要層級的一般性背景描述。

**(b) line 692（§5.4 限制分析，中優先，引用位置邏輯問題）**：
> 「第三，本文尚未系統驗證強烈動態反光（如戶外強日照）的消除效果【9】。」

此處引用 [9] 是用來支撐「**本文**尚未驗證...」這一**關於本文自身的限制陳述**——但 [9]（一篇 SIRR 綜述）顯然無法對「本文是否驗證了某事」提供證據。合理的引用方式應該是：[9] 若確實將「強烈動態反光/戶外強日照」列為 SIRR 領域中已知的挑戰，則應寫成「...強烈動態反光（如戶外強日照）的消除效果，此為 SIRR 文獻中已知的挑戰之一【9】」，引用支撐的是「這是一個已知挑戰」而非「本文沒做」。

**建議處理方向（提案，待使用者確認）**：
- (a)：若不執行 [9] 全文驗證，建議將「往往大幅退化」軟化為更一般的敘述（例如移除具體的退化程度描述，僅保留「domain gap 是 SIRR 實務部署的核心挑戰之一【9】」），或讀取 [9] 全文確認是否有對應章節討論跨域部署退化。
- (b)：建議調整句子結構，讓【9】支撐「這是已知挑戰」而非「本文未驗證」，或在確認 [9] 確實討論該主題後再保留引用。

### 3.5 【中優先】[14] CycleGAN「原始比較實驗」具體宣稱（lines 290-291）

> 「CycleGAN【14】引入循環一致性損失，無需配對資料即可學習域間映射，理論上適合配對資料難以取得的場景。然而 **Zhu 等人【14】的原始比較實驗顯示，在配對資料充足的條件下，Pix2Pix 優於 CycleGAN 及其他無監督方法**。」

`citation_verification_record.md` 中 [14] 僅有「⚠️PARTIAL（arXiv 摘要已讀）」與角色描述「GAN-based 影像轉換的代表性工作」，**沒有列出與「Pix2Pix 優於 CycleGAN（作為 paired-data oracle 比較）」相關的摘要 bullet**。

這一說法在 GAN 文獻中**廣為人知**（CycleGAN 原論文確實將 pix2pix 作為「upper bound/oracle」對照組），但既然本文將其表述為「Zhu 等人的原始比較實驗顯示」——這是對 [14] 論文內部一個具體實驗結果的歸因，依 §5.0b 仍建議以原文確認（CycleGAN 論文 Figure/Table 中與 pix2pix 的比較段落）為佳，或在 `citation_verification_record.md` 補充對應 bullet。

### 3.6 【低優先】[33] 與 [29] 的「進一步支持/可能」式推論（lines 331-333, 654-656）

這兩處与 §3.3（[23]）屬於同一類「以類比支持本文假設」的論證模式，但措辭已經比 [23] 處緩和：

- line 331-333（[33]）：「此類邊緣引導設計在醫學影像這一特殊 domain 中的成功，**進一步支持**本文以固定梯度先驗跨場景遷移的假設」——使用「進一步支持」而非「驗證了」，且本身定位為「假設」，符合 §5.2 對「以類比/推論作為輔助論證、但不包裝為已證實事實」的可接受範圍。
- line 654-656（[29] HED）：「在跨場景應用時**可能**出現邊緣偵測器對目標場景紋理的語義誤判」——使用「可能」表明這是作者的**假設性風險評估**，而非引用 [29] 報告的實際失敗案例，[29] 僅作為「可學習邊緣偵測器」的方法名稱引用。

**建議**：這兩處風險較低，**可不修改**；若使用者希望進一步強化學術嚴謹度，可考慮在 [33] 處同樣加上「（類比推論，非 [33] 直接宣稱）」的限定語氣詞，但非必要。

---

## 4. 待使用者決策事項總覽

| # | 項目 | 類型 | 建議優先序 |
|---|------|------|-----------|
| 1 | 標題頁：學生姓名/指導教授/系所/學校/城市/Email（§1.1） | 需使用者提供資訊 | 高（投稿前必填） |
| 2 | FIG-1 整體架構圖缺失（§1.2） | 需使用者決策：用現有圖或重繪 | 高（正文有懸空引用） |
| 3 | 致謝對象名稱確認（§1.3） | 需使用者確認 | 中 |
| 4 | 810/248 → 1951/491 數字修正（§1.4） | 提案，待核准後套用 | 高（內部數字矛盾，審稿易發現） |
| 5 | 【13, B】括號風格（§1.5） | 提案，純格式 | 低 |
| 6 | [2]/[GAP-E] SIR² 資料集歸屬修正（§3.1） | 提案，待核准後套用 | 高 |
| 7 | [3] 「編碼器-解碼器根本缺陷」措辭軟化或全文驗證（§3.2） | 提案：軟化措辭 vs. 讀取全文 | 高 |
| 8 | [23] 「驗證了...穩定性」改為推論措辭（§3.3） | 提案，待核准後套用 | 高 |
| 9 | [9] 兩處引用方向調整（§3.4） | 提案：軟化措辭 vs. 讀取全文 | 中 |
| 10 | [14] CycleGAN 比較實驗宣稱（§3.5） | 提案：補充驗證或保留現狀 | 中 |
| 11 | [33]/[29] 推論措辭（§3.6） | 可不處理 | 低 |

> 以上 #4, #6, #8 三項屬於「修正即可解決、不需額外資料」的問題，若使用者核准，可直接套用至
> `cvgip2025_chinese.py` 並重新生成 docx/pdf。#1, #2, #3 需使用者提供額外資訊。
> #7, #9, #10 需使用者決定「軟化措辭」或「啟動 `/cite-papers` 讀取全文驗證」。
