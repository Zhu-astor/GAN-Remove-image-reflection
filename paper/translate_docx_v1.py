"""
translate_docx_v1.py

In-place translation of:
    D:\Download\cvgip2025_SGA_chinese (6)_reduced.docx
Output:
    D:\Download\cvgip2025_SGA_english_v1.docx

Strategy:
  1. Copy the source docx verbatim (preserves all Word styles, images,
     tables, page layout, section breaks, column settings).
  2. Walk body paragraphs by index; for each paragraph whose index
     appears in TRANSLATIONS, replace its run text with the English
     translation while preserving the paragraph's existing run properties
     (font, size, bold/italic, colour, spacing).
  3. Chinese citation brackets 【n】 are converted to [n] inside every
     translation string before insertion.

Changelog
---------
v1.0 — 2026-06-16
    Initial English translation of (6)_reduced.docx.
    Source: cvgip2025_SGA_chinese (6)_reduced.docx (181 paragraphs, 2 tables).
    All natural-language Chinese text translated; math formula lines
    (para 45, 47, 49, 53, 57, 59, 72, 74, 76), reference entries
    (para 150-180), and already-English headers/author lines left intact.
"""

import sys
import shutil
from copy import deepcopy
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

SRC = r"D:\Download\cvgip2025_SGA_chinese (6)_reduced.docx"
DST = r"D:\Download\cvgip2025_SGA_english_v1.docx"

# ---------------------------------------------------------------------------
# TRANSLATIONS — keyed by paragraph index (0-based, body paragraphs only).
# Only paragraphs with Chinese content are listed; all others are left intact.
# Chinese citation brackets 【n】 → [n] is handled automatically at runtime.
# ---------------------------------------------------------------------------
TRANSLATIONS = {
    # Title
    0: (
        "Sobel-Guided Attention for Pix2Pix-Based Cross-Scene "
        "Single Image Reflection Removal\n"
        ": A Case Study on Museum Artifact Recognition"
    ),

    # Abstract body
    6: (
        "The fundamental bottleneck of supervised single image reflection removal "
        "(SIRR) lies in the near-impossibility of obtaining pixel-aligned paired "
        "training data in real-world application scenes. Motivated by cross-scene "
        "generalization, this paper proposes a Sobel-Guided Attention (SGA) module "
        "integrated into the Pix2Pix image-to-image translation framework, "
        "investigating how a model trained on public SIRR datasets can be deployed "
        "to unseen target scenes without fine-tuning. The SGA module extracts "
        "per-channel edge gradient magnitudes using fixed Sobel convolution kernels "
        "to drive a CBAM-style channel-and-spatial dual-branch attention; since the "
        "Sobel kernels are fixed parameters whose computation is independent of the "
        "training data distribution, they provide a structurally guided foundation "
        "for cross-scene transfer. We validate the approach on museum artifact "
        "reflection removal: museums represent a real-world case where pixel-aligned "
        "paired data are completely unobtainable at scale due to immovable exhibits "
        "and protective glass enclosures. Experiments show that SGA enables the "
        "model to generalize successfully to museum artifacts across scenes, "
        "improving downstream YOLOv8 exhibit recognition accuracy from 92.7% to "
        "94.5%, demonstrating the practical benefit of cross-scene reflection removal."
    ),

    # Keywords
    7: (
        "Keywords: Single image reflection removal, cross-scene generalization, "
        "Sobel structural prior, conditional generative adversarial network, Pix2Pix."
    ),

    # §1 Introduction — paragraph 1
    9: (
        "Single image reflection removal (SIRR) is a persistently studied research "
        "problem in computer vision, with applications spanning museum exhibit "
        "photography, automotive windshield imaging, and numerous other real-world "
        "scenarios. However, existing supervised deep-learning methods face a "
        "fundamental training dilemma: models require pixel-aligned paired images "
        "(with/without reflection) for training, yet in the vast majority of real "
        "deployment scenarios, such paired data are nearly impossible to obtain at "
        "scale. In factory glass inspection, for example, replacing glass or "
        "controlling light sources entails prohibitive costs; in automotive scenes, "
        "continuously changing ambient lighting precludes acquiring static aligned "
        "pairs. This domain gap causes even models that perform well on public "
        "datasets to degrade substantially in actual target-scene deployment [1]."
    ),

    # §1 — paragraph 2
    10: (
        "Museum artifact reflection removal is a representative instance of this "
        "dilemma. Exhibits are protected by glass enclosures and cannot be moved "
        "or arbitrarily disassembled; ambient lighting is dictated by exhibition "
        "design and cannot be freely adjusted, making pixel-aligned paired data "
        "virtually impossible to obtain at scale in practice. Yet the demand for "
        "museum smart-guide systems is real: when visitors photograph exhibits with "
        "smartphones, glass reflections substantially reduce the recognition accuracy "
        "of AI systems — in our experiments, YOLOv8 [2] achieved only 92.7% "
        "recognition accuracy on images with reflections. The museum setting thus "
        "provides an ideal case study with quantifiable downstream metrics, "
        "well-suited for validating the practical efficacy of cross-scene SIRR "
        "generalization methods."
    ),

    # §1 — paragraph 3
    11: (
        "The core research question of this paper is: can a model trained on public "
        "SIRR datasets be deployed directly in completely unseen target scenes "
        "without any fine-tuning? What architectural design can support such "
        "cross-scene transfer? The key insight is that if the feature representations "
        "a model relies on are scene-agnostic, then the reflection removal capability "
        "learned on one scene can naturally transfer. SIRR literature widely treats "
        "the low-gradient characteristic of reflection layers as a commonly used "
        "prior assumption [6][8], while object edges exhibit high-gradient responses "
        "due to material discontinuity — a foundational edge cause universally "
        "recognized in the computer vision edge detection literature [31]; this paper "
        "takes these two properties as the starting point for designing feature "
        "representations that capture this gradient difference."
    ),

    # §1 — paragraph 4
    12: (
        "Based on this observation, we propose a design strategy using fixed Sobel "
        "gradients as attention-driving signals. The Sobel convolution kernels are "
        "fixed parameters; their computed gradient magnitudes reflect only the local "
        "structural properties of the image, independent of the scene distribution "
        "of the training data. Lu et al. [3] demonstrated the effectiveness of "
        "Sobel-guided attention in medical image segmentation, while CBAM [4] "
        "established the complementary advantage of dual-branch channel-and-spatial "
        "attention — together providing the theoretical basis for our Sobel-Guided "
        "Attention (SGA) module design. SGA drives a CBAM-style dual-branch "
        "attention using fixed Sobel gradient magnitudes, performing edge-aware "
        "feature recalibration ahead of the first encoder block of the Pix2Pix "
        "U-Net generator, enabling the model to distinguish between 'exhibit "
        "structure' and 'reflection interference' at the pixel level [5]."
    ),

    # §1 contributions intro
    13: "The main contributions of this paper are as follows:",

    # Contribution (1)
    14: (
        "(1) We propose the SGA module, which uses fixed Sobel edge gradients to "
        "guide CBAM-style channel-and-spatial dual-branch attention. Without "
        "introducing any additional trainable parameters, it provides domain-agnostic "
        "structural prior guidance for Pix2Pix reflection removal."
    ),

    # Contribution (2)
    15: (
        "(2) We adopt a 'train on public SIRR datasets, deploy directly to target "
        "scene' research framework to investigate the role of fixed structural priors "
        "in cross-scene SIRR generalization, providing a viable methodological path "
        "for real-world applications where paired data are difficult to obtain."
    ),

    # Contribution (3)
    16: (
        "(3) We conduct a case study on museum artifact reflection removal, training "
        "on public SIRR datasets and applying across scenes to museum images, with "
        "quantitative validation via YOLOv8 recognition accuracy (92.7% -> 94.5%)."
    ),

    # §2.1 heading
    18: "2.1. Single Image Reflection Removal",

    # §2.1 — paragraph 1
    19: (
        "SIRR research has evolved from optimization-based traditional methods to "
        "deep learning. Fan et al. [6] proposed CEILNet, the first cascaded CNN "
        "architecture to introduce edge information into SIRR: an E-CNN first "
        "estimates object edge maps, then an I-CNN uses edge maps as auxiliary "
        "input to recover the transmission layer. This design established the "
        "'edge-guided restoration' technical paradigm, a key predecessor of our "
        "SGA module design."
    ),

    # §2.1 — paragraph 2
    20: (
        "Li et al. [7] proposed IBCLN, using convolutional LSTM to achieve "
        "iterative progressive alternating refinement of transmission and reflection "
        "layers, and built a real-scene paired dataset with dense ground-truth "
        "annotations — one of the datasets used in our training. Chi et al. [5] "
        "noted that information loss due to downsampling increases decoder "
        "restoration difficulty, an observation that echoes our design motivation "
        "for injecting edge attention before the encoder (see §3.1). Dong et al. [8] "
        "proposed location-aware reflection removal, using an explicit reflection "
        "location detection module to regress a reflection probability map to guide "
        "feature flow, demonstrating the effectiveness of 'explicit spatial location "
        "cues' in SIRR tasks."
    ),

    # §2.1 — paragraph 3
    21: (
        "Recent methods have advanced in different directions: DURRNet [9] endows "
        "the network with theoretical interpretability through algorithm unrolling; "
        "PromptRR [10] uses a diffusion model as a frequency-domain prompt generator "
        "to drive a Transformer, achieving the latest SOTA but with high inference "
        "cost due to multi-step sampling. Although these methods have achieved "
        "significant progress on public benchmarks, cross-scene generalization "
        "remains the core challenge for real-world deployment of deep-learning SIRR."
    ),

    # §2.2 heading
    22: "2.2. Image-to-Image Translation and Conditional Generative Adversarial Networks",

    # §2.2 — paragraph 1
    23: (
        "Generative adversarial networks [11][15] learn data distributions via "
        "adversarial training between a generator and a discriminator, providing a "
        "powerful framework for image synthesis. Mirza and Osindero [12] proposed "
        "cGAN, incorporating condition vectors into both generator and discriminator, "
        "enabling the network to learn deterministic input-to-output mappings and "
        "establishing the theoretical foundation for Pix2Pix. Isola et al. [13] "
        "instantiated this paradigm as Pix2Pix: a U-Net generator paired with a "
        "PatchGAN discriminator, trained with an L1+cGAN loss combination, achieving "
        "breakthrough results on multiple paired translation tasks and establishing "
        "the effectiveness of paired supervised training in image restoration tasks."
    ),

    # §2.2 — paragraph 2
    24: (
        "CycleGAN [14] introduced cycle-consistency loss to learn cross-domain "
        "mappings without paired data, theoretically suitable for scenarios where "
        "paired data are difficult to obtain. Comparative experiments show that "
        "Pix2Pix outperforms CycleGAN and other unsupervised methods when sufficient "
        "paired data are available. Our choice to train Pix2Pix on public paired "
        "SIRR datasets and then deploy cross-scene is a design strategy that balances "
        "'leveraging the supervisory strength of available paired data' with 'zero "
        "paired data in the target scene'."
    ),

    # §2.3 heading
    25: "2.3. Attention Mechanisms",

    # §2.3 — paragraph 1
    26: (
        "Hu et al. [16] proposed SENet, which compresses spatial dimensions via "
        "global average pooling and then learns inter-channel dependencies through "
        "fully connected layers for channel feature re-weighting, establishing the "
        "position of channel attention in CNNs. Woo et al. [4] proposed CBAM, "
        "sequentially inferring attention maps along channel and spatial dimensions, "
        "outperforming SENet across extensive classification and detection "
        "experiments, demonstrating the complementarity of dual-branch attention. "
        "The dual-branch design of our SGA module derives directly from the CBAM "
        "framework, replacing pure learned statistics with fixed Sobel gradients as "
        "the driving signal — this substitution is the key to achieving "
        "domain-agnostic characteristics."
    ),

    # §2.3 — paragraph 2
    27: (
        "Lu et al. [3] proposed a multi-scale attention network guided by the Sobel "
        "operator for medical image segmentation, using gradient magnitude as a "
        "structural prior to drive the attention mechanism, significantly improving "
        "segmentation boundary precision. A significant domain gap also exists "
        "between medical images and natural scenes; that method achieved good "
        "segmentation results within a single medical domain using fixed Sobel "
        "gradients to drive attention, inspiring us to extend this design principle "
        "further to cross-scene scenarios for validation. The combination of global "
        "context modeling and channel calibration outperforms either mechanism alone, "
        "further supporting our dual-branch design [17][18]."
    ),

    # §2.4 heading
    28: "2.4. Edge-Guided Image Processing",

    # §2.4 — paragraph 1
    29: (
        "Using edge information to guide image restoration is a mature technical "
        "paradigm: Xie and Tu [19] proposed holistically nested edge detection, "
        "demonstrating through multi-scale deep supervision that edge features at "
        "different levels carry complementary structural information — an important "
        "benchmark for subsequent edge-guided methods; Ji et al. [20] proposed "
        "DGNet, which decouples texture and semantic features via object gradient "
        "supervision; our work is deeply inspired by its gradient-guided feature "
        "refinement approach."
    ),

    # §2.4 — paragraph 2
    30: (
        "Sharp U-Net [21] incorporates sharpening kernels before the skip connections "
        "of a U-Net, reducing the semantic dissimilarity between encoder and decoder "
        "features — echoing our design philosophy of injecting Sobel structural "
        "guidance before skip connections. Li and Liu [22] introduced a gradient-map "
        "edge quality loss in MRI super-resolution tasks, compelling the model to "
        "learn edge structural details, demonstrating that gradient-guided design is "
        "also applicable in the medical image domain."
    ),

    # §3.1 heading
    33: "3.1. Overall Architecture",

    # §3.1 body
    34: (
        "The proposed method uses Pix2Pix [13] as the backbone architecture, with "
        "the core innovation of inserting the SGA module before the first encoder "
        "block of the U-Net generator. The overall pipeline is shown in Fig. 1: "
        "given a reflection-contaminated input image Ir ∈ R(H×W×3), the SGA module "
        "first performs edge-aware feature recalibration, outputting a structurally "
        "enhanced feature map x', which is then mapped by the U-Net generator to a "
        "reflection-free output T̂ ∈ R(H×W×3). The PatchGAN discriminator evaluates "
        "the authenticity of local image patches during training, forcing the "
        "generator to produce outputs with realistic high-frequency details. SGA is "
        "inserted at the front rather than at an intermediate layer because reflection "
        "interference already exists at the pixel level; allowing the network to "
        "first perform downsampling before remediation risks losing structural detail "
        "information during the downsampling process [5], and intermediate-layer "
        "features are already mixed with domain-specific semantic information, which "
        "is detrimental to cross-scene transfer."
    ),

    # Fig. 1 caption
    39: (
        "Fig. 1. Overall architecture diagram. The input image x (with reflection) "
        "is recalibrated by the SGA module (structure detailed in Fig. 2) to x', "
        "then mapped by the U-Net generator G to the reflection-free output T̂. "
        "During training, the PatchGAN discriminator D compares (x, T̂) with (x, y) "
        "to compute the adversarial loss Ladv, and T̂ is compared with ground truth y "
        "to compute LL1, combined as Ltotal = Ladv + λ·LL1 (λ = 100); during "
        "inference, only the generator G is required."
    ),

    # §3.2 heading
    41: "3.2. Sobel-Guided Attention Module (SGA)",

    # §3.2 intro
    42: (
        "The SGA module accepts the raw input image x ∈ R(H×W×3) and outputs a "
        "recalibrated feature map x' of the same dimensions, comprising three "
        "sequentially executed stages: Sobel feature extraction, channel attention, "
        "and spatial attention. All computations in SGA are based on fixed "
        "parameters, introducing no additional trainable parameters, ensuring that "
        "the domain-agnostic properties of the attention guidance signal do not "
        "degrade through training dynamics."
    ),

    # §3.2.1 heading
    43: "3.2.1. Sobel Feature Extraction",

    # §3.2.1 intro sentence
    44: (
        "Fixed 3×3 Sobel horizontal convolution kernel Kx and vertical convolution "
        "kernel Ky are applied separately to each of the three input image channels "
        "(R, G, B) to compute gradient components:"
    ),

    # "where Kx and Ky are defined as follows:"
    46: "where Kx and Ky are defined as follows:",

    # "The gradient magnitude is defined as:"
    48: "The gradient magnitude is defined as:",

    # §3.2.1 Sobel map discussion
    50: (
        "This yields the Sobel feature map S ∈ R(H×W×3), capturing the edge "
        "intensity distribution of each channel. As described in §1, reflection "
        "regions exhibit low gradients [6][8] while exhibit edges exhibit high "
        "gradients [31]; S therefore simultaneously captures the difference in "
        "gradient intensity between 'object structure' and 'reflection interference', "
        "providing the basis for SGA's attention design."
    ),

    # §3.2.2 heading
    51: "3.2.2. Channel Attention Branch",

    # §3.2.2 intro
    52: (
        "Global average pooling is applied to the Sobel feature map S to obtain a "
        "channel descriptor v ∈ R3. Channel attention weights are then generated "
        "via a 1×1 convolution and Sigmoid activation:"
    ),

    # §3.2.2 discussion
    54: (
        "The channel-recalibrated feature map is Xc = x ⊙ wc, where ⊙ denotes "
        "element-wise multiplication broadcast along spatial dimensions. This "
        "operation adaptively adjusts the contribution ratio of R, G, B channels "
        "according to the relative importance of each channel's edge intensity, "
        "assigning higher weights to channels rich in edge information."
    ),

    # §3.2.3 heading
    55: "3.2.3. Spatial Attention Branch",

    # §3.2.3 intro
    56: (
        "Average pooling (AvgPool) and max pooling (MaxPool) are applied to the "
        "channel-recalibrated feature map Xc along the channel dimension separately, "
        "then concatenated and passed through a 7×7 convolution and Sigmoid "
        "activation to generate a spatial attention map:"
    ),

    # "The final output of the SGA module is:"
    58: "The final output of the SGA module is:",

    # §3.2.3 discussion
    60: (
        "This dual-branch design ensures that feature regions rich in object edges "
        "are enhanced along both channel and spatial dimensions, while diffuse "
        "reflection regions are suppressed due to their weak Sobel response. Since "
        "both wc and ws are driven by fixed Sobel gradients, their computation logic "
        "remains consistent across any scene, ensuring the stability of attention "
        "during cross-scene deployment."
    ),

    # Fig. 2 caption
    64: (
        "Fig. 2. Detailed structure of the SGA module. Sobel features are "
        "sequentially recalibrated by channel attention (1×1 convolution) and "
        "spatial attention (7×7 convolution), then element-wise multiplied with "
        "the original input and fed into the first encoder block of the U-Net."
    ),

    # §3.3 heading
    66: "3.3. U-Net Generator Architecture",

    # §3.3 body
    67: (
        "The SGA module is followed by the standard Pix2Pix U-Net generator [13][23]. "
        "The encoder consists of 7 downsampling blocks and the decoder consists of "
        "7 upsampling blocks, connected via skip connections that pass encoder feature "
        "maps at corresponding resolutions to preserve spatial details. The final "
        "layer outputs pixel values to [-1, 1] via tanh activation. All input and "
        "output images are normalized to [-1, 1] range via xnorm = x / 127.5 - 1."
    ),

    # §3.4 heading
    68: "3.4. PatchGAN Discriminator",

    # §3.4 body
    69: (
        "The PatchGAN discriminator [13] judges the authenticity of overlapping N×N "
        "image patches in the image individually, rather than outputting a single "
        "scalar for the entire image. This design focuses the discriminator on the "
        "authenticity of high-frequency local textures, which is particularly "
        "appropriate for reflection removal — reflections primarily affect local "
        "regions, and patch-level judgment provides more precise gradient signals. "
        "Our discriminator consists of 4 convolutional layers followed by a 1×1 "
        "output layer."
    ),

    # §3.5 heading
    70: "3.5. Loss Function",

    # §3.5 intro
    71: "The training objective combines adversarial loss with pixel reconstruction loss:",

    # "The adversarial loss Ladv uses mean squared error:"
    73: "The adversarial loss Ladv uses mean squared error:",

    # "The L1 pixel reconstruction loss..."
    75: (
        "The L1 pixel reconstruction loss spatially constrains the difference "
        "between the generated image and the ground-truth reflection-free image:"
    ),

    # §3.5 discussion
    77: (
        "The weight coefficient λ = 100 heavily favors the L1 term, ensuring "
        "pixel-level structural accuracy while the adversarial loss supplements "
        "perceptual realism. This configuration is consistent with the original "
        "Pix2Pix settings [13]."
    ),

    # §3.6 heading
    78: "3.6. Training Configuration",

    # §3.6 body
    79: (
        "Input and output images are uniformly resized to 256×256 pixels. The Adam "
        "optimizer is used with a learning rate of 5e-5, training for 500 epochs "
        "with a batch size of 8. Data augmentation employs random horizontal "
        "flipping applied synchronously to input-output pairs to maintain correct "
        "pixel correspondence."
    ),

    # §4.1 heading
    81: "4.1. Experimental Design",

    # §4.1 intro
    82: (
        "The reflection removal model is trained entirely on public SIRR datasets "
        "without using any images from the target scene; after training, the model "
        "is deployed directly to museum scenes to evaluate its reflection removal "
        "performance in 'never-seen target domains'."
    ),

    # §4.1.1 heading
    83: "4.1.1. Training Dataset (Public SIRR Datasets)",

    # §4.1.1 — paragraph 1
    84: (
        "The training data integrates four public single-image reflection removal "
        "datasets: SIR² [24] (large-scale real-scene paired reflection dataset with "
        "diverse reflection types), IBCLN [7] (real-scene reflection image pairs "
        "from multiple indoor environments), ERRNET [25] (reflection scenes under "
        "multiple material surfaces and lighting conditions with rich variation in "
        "reflection intensity gradients), and RFC [26] (paired data captured with "
        "flash assistance, covering multiple glass types and indoor lighting "
        "conditions)."
    ),

    # §4.1.1 — paragraph 2
    85: (
        "The four datasets are merged and randomly split, with 80% as the training "
        "set (810 pairs) and 20% as the test set (248 pairs). Splitting uses random "
        "sampling rather than dataset-based partitioning, ensuring that training and "
        "test sets have similar distributions of reflection types, scene diversity, "
        "and lighting conditions."
    ),

    # Fig. 3 caption
    87: (
        "Fig. 3. Sample examples from the training dataset (left: Original, source "
        "image with reflection; right: Generated, corresponding model output). The "
        "scenes involve ordinary architecture and vegetation, completely unrelated to "
        "museum artifacts, directly illustrating the absence of domain overlap "
        "between training data and evaluation scenes."
    ),

    # §4.1.2 heading
    88: "4.1.2. Case Study: Museum Collection Evaluation Set",

    # §4.1.2 — paragraph 1
    89: (
        "Museum artifact reflection removal is the selected cross-scene case for "
        "this paper, representing a typical situation where 'target-scene paired "
        "data are completely unobtainable at scale', while also providing "
        "quantifiable downstream recognition accuracy as an evaluation metric. We "
        "collected museum artifact images with reflections as an evaluation set, "
        "covering 7 exhibit categories including ceramics, metal artifacts, and "
        "three-dimensional sculptures, totaling 699 images — entirely excluded from "
        "model training and used only for cross-scene benefit verification of the "
        "downstream recognition task."
    ),

    # §4.1.2 — paragraph 2
    90: (
        "This evaluation set provides only downstream recognition accuracy and "
        "cannot provide PSNR/SSIM/LPIPS quantitative metrics."
    ),

    # §4.2 heading
    91: "4.2. Evaluation Metrics",

    # §4.2 — paragraph 1
    92: "This paper adopts a two-tier evaluation strategy:",

    # §4.2 — paragraph 2
    93: "(1) Image restoration metrics (on the public SIRR test set): PSNR, SSIM [27], LPIPS [28];",

    # §4.2 — paragraph 3
    94: (
        "(2) Cross-scene downstream benefit (on the museum evaluation set): "
        "YOLOv8 [2] exhibit classification accuracy, directly reflecting the impact "
        "of reflection removal on actual recognition tasks. The two-tier evaluation "
        "respectively quantifies the model's restoration quality within the training "
        "distribution and its practical benefit on the target domain, providing a "
        "complete picture of cross-scene generalization."
    ),

    # §4.3 heading
    95: "4.3. Ablation Study: Contribution of SGA to Generalization",

    # §4.3 intro
    96: (
        "To verify the effectiveness of the SGA module, this paper compares two "
        "configurations on the public SIRR test set: (a) Baseline Pix2Pix: standard "
        "Pix2Pix without any attention module; (b) Pix2Pix+SGA (ours): incorporating "
        "the complete dual-branch Sobel-guided attention module. Quantitative results "
        "are shown in Table 1."
    ),

    # Table 1 caption
    98: "Table 1. Ablation experiment results on the public SIRR test set (491 pairs).",

    # §4.3 discussion
    100: (
        "Table 1 shows that after adding SGA, PSNR (22.682 dB) and SSIM (0.8192) "
        "are slightly lower than baseline Pix2Pix (23.896 dB / 0.8706), while LPIPS "
        "(0.2178) is higher than baseline (0.1630). This does not indicate "
        "degradation in reflection removal capability, but rather a characteristic "
        "typically observed when GAN-based methods optimize for perceptual quality. "
        "Blau and Michaeli [29] theoretically proved that a fundamental tradeoff "
        "exists between perceptual quality and distortion metrics — methods with "
        "higher perceptual quality tend to yield lower PSNR/SSIM, and this "
        "phenomenon does not disappear with choice of metric. Ledig et al. [30] "
        "experimentally confirmed in image super-resolution that 'minimizing MSE "
        "encourages the model to output the pixel-averaged over all plausible "
        "solutions, leading to over-smoothed results'; the Pix2Pix L1 loss [13] "
        "used in this paper has the same mechanistic characteristics, and SGA's "
        "adversarial training moves the model toward the perceptual boundary, so "
        "lower PSNR/SSIM is an expected phenomenon."
    ),

    # Fig. 4 caption
    106: (
        "Fig. 4. Examples of reflection removal results on the training dataset "
        "(Original/Generated comparison). The reflection regions are visibly and "
        "substantially reduced, corroborating that the model has genuine reflection "
        "removal capability in its training domain — should be interpreted together "
        "with the quantitative metrics in Table 1 (see Perception-Distortion Tradeoff "
        "discussion above)."
    ),

    # §4.4 heading
    108: "4.4. Visual Comparison and Attention Map Analysis",

    # §4.4 — paragraph 1
    109: (
        "Fig. 5 presents a visual comparison of museum artifacts before and after "
        "Pix2Pix+SGA processing: reflection regions are visibly reduced, a difference "
        "that is quite intuitive to the human eye and constitutes the most direct "
        "visual evidence of this paper's effectiveness. This demonstrates that the "
        "Sobel prior indeed enables edge-aware attention to transfer effectively to "
        "museum scenes."
    ),

    # §4.4 — paragraph 2
    110: (
        "Fig. 6 presents a visualization of the Sobel Attention Map: attention is "
        "highly concentrated on exhibit edges and material detail regions, while "
        "attention weights in diffuse reflection regions are noticeably lower, "
        "directly verifying that SGA can still correctly identify the spatial "
        "distribution of 'object structure' versus 'reflection interference' in "
        "the museum domain."
    ),

    # Fig. 5 main caption
    114: "Fig. 5. Cross-scene visual comparison on museum collection items.",

    # Fig. 5 sub-caption
    115: (
        "Top row: original images with reflection (Original); "
        "bottom row: Pix2Pix+SGA reflection removal results (Generated)."
    ),

    # Fig. 6 caption
    119: (
        "Fig. 6. Sobel gradient magnitude visualization (left: original image; "
        "right: gradient magnitude map). Top: scene with reflection — the reflection "
        "region produces high gradients only at its boundary while remaining "
        "low-gradient internally; bottom: scene without reflection — gradients are "
        "concentrated along object structural edges. This contrast is the "
        "domain-agnostic physical foundation of the SGA attention guidance signal."
    ),

    # Fig. 7 caption
    121: (
        "Fig. 7. Training loss curves. Blue line: Generator loss; orange line: "
        "Discriminator loss; horizontal axis: training iterations."
    ),

    # §4.6 heading
    122: "4.6. Cross-Scene Downstream Validation",

    # §4.6 — paragraph 1
    123: (
        "Fig. 8 uses YOLOv8 recognition results as examples, presenting changes in "
        "detection bounding boxes and confidence values for the same exhibit images "
        "before and after reflection removal, demonstrating the direct gain of "
        "reflection removal on downstream recognition tasks."
    ),

    # §4.6 — paragraph 2
    124: (
        "At the quantitative level, YOLOv8 recognition results were tested on the "
        "museum evaluation set: the recognition accuracy of original "
        "reflection-containing images was 92.7% (40 of 699 images failed to be "
        "successfully recognized); after Pix2Pix+SGA reflection removal, 10 of "
        "these 40 (25%) were successfully recognized, raising overall accuracy to "
        "94.5% (+1.8 pp)."
    ),

    # §4.6 — paragraph 3
    125: (
        "The overall accuracy improvement (+1.8 pp) is relatively modest, attributed "
        "primarily to two constraining factors:"
    ),

    # §4.6 — paragraph 4
    126: (
        "(1) The base number of correctable failure samples is inherently limited — "
        "only 40 of 699 images (5.7%) originally failed recognition; even if all "
        "were recovered, the theoretical accuracy ceiling would only increase by "
        "5.7 percentage points. Measured by 'failure case recovery rate', this paper "
        "actually recovered 10/40 (25%), demonstrating that reflection removal "
        "provides substantial benefit to 'edge cases'."
    ),

    # §4.6 — paragraph 5
    127: (
        "(2) The recognition model itself already achieves a high baseline accuracy "
        "of 92.7% on original images, with inherently limited room for improvement, "
        "making the marginal contribution of reflection removal difficult to fully "
        "manifest in the overall metric. Furthermore, the scale of our museum "
        "evaluation set (699 images, 7 categories) is relatively small compared to "
        "large general-purpose detection benchmarks (such as COCO), with limited "
        "samples per category, making overall accuracy more sensitive to individual "
        "cases."
    ),

    # Table 2 caption
    128: (
        "Table 2. Cross-scene downstream recognition accuracy "
        "(museum evaluation set, 699 images, 7 exhibit categories)."
    ),

    # Fig. 8 caption
    137: (
        "Fig. 8. YOLOv8 recognition result examples: comparison of detection "
        "bounding boxes and confidence values between images with reflection (top) "
        "and images processed by Pix2Pix+SGA (bottom)."
    ),

    # §5.1 heading
    140: "5.1. Why a Fixed Sobel Prior Enables Cross-Scene Generalization",

    # §5.1 — paragraph 1
    141: (
        "The most central finding of this paper is that a model trained on a "
        "completely different domain can, after adding SGA, generalize directly to "
        "museum artifacts without any form of fine-tuning or domain adaptation "
        "(§4.6). From Fig. 7, we can observe the training dynamics on the public "
        "SIRR dataset: both D loss and G loss converge stably with no obvious mode "
        "collapse, indicating that the inclusion of the SGA module does not affect "
        "the stability of adversarial training."
    ),

    # §5.1 — paragraph 2
    142: (
        "As described in §3.2.1, SGA uses fixed Sobel gradients as "
        "attention-driving signals [6][8][31], with computation relying only on "
        "local pixel values of the input image, independent of the scene "
        "distribution of the training data. We conjecture that this may be one of "
        "the contributing factors to the cross-scene transfer phenomenon observed in "
        "this paper: SGA's computation logic does not adjust with the scene "
        "distribution of training data, so the attention guidance signals it "
        "generates are not subject to systematic shifts across different scenes due "
        "to differences in training distribution."
    ),

    # §5.1 — paragraph 3
    143: (
        "By contrast, if a learnable edge detector (such as HED [19]) is used as "
        "the attention-driving signal, its weights would be adjusted according to "
        "the scene distribution of training data, potentially resulting in semantic "
        "misidentification of target-scene textures by the edge detector during "
        "cross-scene application."
    ),

    # §5.2 heading
    144: "5.2. Limitation Analysis",

    # §5.2 — paragraph 1
    145: (
        "This method has several limitations. First, when objects themselves contain "
        "large areas of uniform low-gradient regions (such as plain-colored ceramics "
        "or solid-color backgrounds), the Sobel response is weak, SGA's edge "
        "guidance is limited, and the model degrades toward behavior approximating "
        "standard Pix2Pix. Second, as noted in §4.1.2, the museum evaluation set in "
        "this paper lacks reflection-free ground truth; it must be interpreted "
        "together with the ablation results on the public SIRR test set to present "
        "a complete picture of cross-scene generalization. Third, this paper has not "
        "yet systematically validated the removal of strongly dynamic reflections "
        "(such as intense outdoor sunlight)."
    ),

    # §5.2 — paragraph 2
    146: (
        "Fourth, as described in §4.6, the overall improvement of +1.8 pp is "
        "jointly constrained by the scale of the evaluation set and the baseline "
        "accuracy of the recognition model, with inherently limited theoretical "
        "improvement space; future plans to build larger-scale, cross-scene "
        "downstream evaluation datasets will be an important direction for "
        "subsequently validating the method's effectiveness — confirming the "
        "generalization capability of this paper (factory, automotive, etc.)."
    ),

    # §6 Conclusion body
    148: (
        "Supervised SIRR in real deployment scenarios universally faces the "
        "difficulty of obtaining target domain paired data. Motivated by cross-scene "
        "generalization, this paper proposes integrating the SGA module into the "
        "Pix2Pix architecture to investigate the mechanism of fixed structural "
        "priors in cross-scene SIRR. The SGA module extracts edge gradient signals "
        "using fixed Sobel convolution kernels to drive CBAM-style "
        "channel-and-spatial dual-branch attention, distinguishing object structure "
        "from reflection interference at the pixel level without introducing any "
        "additional trainable parameters. This paper conducts a case study on "
        "museum artifact reflection removal. Experimental results show that a model "
        "trained on public SIRR datasets can generalize to museum artifacts without "
        "any fine-tuning, improving downstream YOLOv8 exhibit recognition accuracy "
        "from 92.7% to 94.5% (+1.8 pp), quantitatively validating the practical "
        "benefit of fixed Sobel structural priors in cross-scene settings."
    ),
}


# ---------------------------------------------------------------------------
# Helper: replace paragraph text while preserving run properties
# ---------------------------------------------------------------------------

def replace_paragraph_text(para, new_text):
    """Replace all run content in `para` with `new_text`.

    Args:
        para     : docx Paragraph object.
        new_text : Replacement string. May contain '\\n' (kept as-is in the
                   paragraph text; Word does not render these as line breaks
                   in the same paragraph, but the string is stored correctly).
    Notes:
        Saves the first run's <w:rPr> element (font name, size, bold, italic,
        colour, etc.) and attaches it to a single new <w:r> element containing
        the translated text.  All other existing runs are removed so that
        sub-paragraph formatting inconsistencies (from the Chinese original) do
        not bleed through.
    """
    p_elem = para._p

    # Collect first run's properties before clearing
    rPr_copy = None
    runs = p_elem.findall(qn('w:r'))
    if runs:
        first_r = runs[0]
        rPr_elem = first_r.find(qn('w:rPr'))
        if rPr_elem is not None:
            rPr_copy = deepcopy(rPr_elem)

    # Remove every existing run element
    for r in p_elem.findall(qn('w:r')):
        p_elem.remove(r)

    # Build new run
    new_r = OxmlElement('w:r')
    if rPr_copy is not None:
        new_r.append(rPr_copy)

    new_t = OxmlElement('w:t')
    new_t.text = new_text
    # Preserve leading/trailing spaces
    if new_text != new_text.strip():
        new_t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    new_r.append(new_t)

    p_elem.append(new_r)


def convert_citations(text):
    """Convert Chinese citation brackets 【n】 → [n]."""
    import re
    return re.sub(r'【(\d+)】', r'[\1]', text)


# ---------------------------------------------------------------------------
# Main translation routine
# ---------------------------------------------------------------------------

def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print(f"Copying source docx ...")
    shutil.copy2(SRC, DST)
    print(f"  {SRC}")
    print(f"  -> {DST}")

    doc = Document(DST)
    paras = doc.paragraphs

    translated = 0
    skipped = []
    for idx, para in enumerate(paras):
        if idx not in TRANSLATIONS:
            continue
        raw_en = TRANSLATIONS[idx]
        en_text = convert_citations(raw_en)
        replace_paragraph_text(para, en_text)
        translated += 1
        preview = en_text[:60].replace('\n', '\\n')
        print(f"  [para {idx:3d}] -> {preview}...")

    doc.save(DST)
    print(f"\nDone. {translated} paragraphs translated.")
    print(f"Output: {DST}")


if __name__ == "__main__":
    main()
