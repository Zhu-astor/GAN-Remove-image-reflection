"""
Draw the SGA (Sobel-Guided Attention) module architecture diagram.

Purpose — Render a publication-quality block diagram of the SGA module exactly
          as implemented in github/Pix2pix.py (build_generator, lines 118-206):
          input d0 -> Sobel filter -> simplified channel attention (1x1 Conv +
          sigmoid + multiply) -> spatial attention (channel avg/max pool ->
          7x7 Conv + sigmoid + multiply) -> element-wise multiply with the
          original input -> fed into the first U-Net encoder layer.
Args    — None (paths are hard-coded; run from any cwd).
Returns — Saves 'sga_module_architecture.png' (300 dpi) into this folder.
Raises  — IOError if the output folder is not writable.
Notes   — The channel attention is the SIMPLIFIED variant actually active in
          the code (single 1x1 Conv + sigmoid), NOT the full CBAM avg/max-pool
          + shared-MLP variant, which is commented out in Pix2pix.py.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ---------------------------------------------------------------- palette ---
C_INPUT   = "#D6E4F0"   # light blue  — image tensors
C_SOBEL   = "#FCE4D6"   # light orange — fixed (non-learnable) operator
C_CONV    = "#E2EFDA"   # light green — learnable layers
C_POOL    = "#FFF2CC"   # light yellow — parameter-free ops
C_OUT     = "#E8DFF0"   # light purple — module output
C_EDGE    = "#404040"
C_GROUP   = "#7F7F7F"
C_ZONE_IN  = "#4F81BD"   # saturated blue   — INPUT zone outline (matches Fig.1)
C_ZONE_PR  = "#70AD47"   # saturated green  — PROCESS zone outline (matches Fig.1)
C_ZONE_OUT = "#8064A2"   # saturated purple — OUTPUT zone outline (matches Fig.1)

FS_TITLE  = 11.5
FS_BOX    = 9.0
FS_SHAPE  = 7.6
FS_OP     = 13


def box(ax, cx, cy, w, h, label, shape=None, fc=C_CONV, fs=FS_BOX):
    """Rounded box centered at (cx, cy) with a label and optional shape note."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.06",
                                fc=fc, ec=C_EDGE, lw=1.1, zorder=3))
    ax.text(cx, cy + (0.10 if shape else 0.0), label, ha="center", va="center",
            fontsize=fs, zorder=4, linespacing=1.25)
    if shape:
        ax.text(cx, cy - h / 2 + 0.16, shape, ha="center", va="center",
                fontsize=FS_SHAPE, color="#595959", style="italic", zorder=4)


def otimes(ax, cx, cy, r=0.16):
    """Circled element-wise multiplication operator."""
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=C_EDGE, lw=1.2, zorder=4))
    ax.text(cx, cy, r"$\otimes$", ha="center", va="center",
            fontsize=FS_OP, zorder=5)


def arrow(ax, x1, y1, x2, y2, style="-|>", color=C_EDGE, lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def zone(ax, x0, x1, y0, y1, label, color):
    """Bold INPUT/PROCESS/OUTPUT caption only -- no surrounding box.
    v2.2: the dashed outline box (v2.1) was removed per feedback that the
    extra frame made the diagram more cluttered, not clearer; the colored,
    bold text alone is enough to mark each region."""
    if label:
        ax.text((x0 + x1) / 2, y1, label, ha="center", va="top",
                fontsize=9.5, color=color, weight="bold")


def main():
    # v2 fix: v1 packed the channel- and spatial-attention multiplies (the
    # otimes circles) flush against their dashed group borders, with only a
    # 0.4-unit gap between the two groups -- the circles visually collided
    # with the boundary lines, making it look ambiguous whether each
    # multiply belongs inside or outside its group. v2 widens the canvas and
    # gives every otimes explicit padding from its own group's border, plus
    # a wider gap between groups, and separates the two bypass ("skip")
    # lines onto two different height lanes so they no longer read as one
    # continuous line.
    fig, ax = plt.subplots(figsize=(17.4, 6.6))
    ax.set_xlim(0, 17.4)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    Y_TOP = 4.9    # identity path of the original input
    Y_BOT = 2.35   # Sobel-attention path

    # ------------------------------------------------------------- input ---
    box(ax, 1.25, Y_TOP, 1.9, 1.0, "Input Image\n$d_0$",
        shape="H×W×3", fc=C_INPUT)

    # identity path: straight line to the final multiply
    # (must equal otimes_s computed in the spatial-attention block below)
    X_FINAL = 13.08
    arrow(ax, 2.20, Y_TOP, X_FINAL - 0.18, Y_TOP)
    ax.text((2.20 + X_FINAL) / 2, Y_TOP + 0.22, "identity path",
            ha="center", fontsize=FS_SHAPE, color="#595959", style="italic")

    # drop from input down to the Sobel branch
    arrow(ax, 1.25, Y_TOP - 0.50, 1.25, Y_BOT, style="-")
    arrow(ax, 1.25, Y_BOT, 2.30 - 0.05, Y_BOT)

    # ------------------------------------------------------- Sobel block ---
    box(ax, 3.15, Y_BOT, 1.7, 1.18,
        "Sobel Filter\n(fixed $G_x$, $G_y$)\nper-channel\n$\\sqrt{G_x^2+G_y^2}$",
        fc=C_SOBEL, fs=8.0)
    ax.text(3.15, Y_BOT - 0.82, "non-learnable", ha="center",
            fontsize=FS_SHAPE, color="#595959", style="italic")
    arrow(ax, 4.00, Y_BOT, 4.55, Y_BOT)

    box(ax, 5.30, Y_BOT, 1.5, 0.95, "Sobel Feature\n$S$", shape="H×W×3",
        fc=C_INPUT)
    arrow(ax, 6.05, Y_BOT, 6.55, Y_BOT)

    # -------------------------------------------- channel attention group ---
    # otimes_c sits 0.3 inside the group's right border (was 0.24 in v1,
    # visually flush against the boundary) so the multiply clearly reads as
    # belonging to this group, not floating on its edge.
    gx0 = 6.55
    otimes_c = 8.61
    gx1 = otimes_c + 0.16 + 0.30
    ax.add_patch(FancyBboxPatch((gx0, Y_BOT - 1.06), gx1 - gx0, 2.12,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="none", ec=C_GROUP, lw=1.0,
                                linestyle=(0, (4, 3)), zorder=1))
    ax.text((gx0 + gx1) / 2, Y_BOT + 1.24, "Channel Attention (simplified)",
            ha="center", fontsize=8.2, color=C_GROUP)

    box(ax, 7.45, Y_BOT, 1.3, 0.95, "1×1 Conv\n+ Sigmoid",
        shape="H×W×3", fc=C_CONV)
    otimes(ax, otimes_c, Y_BOT)
    arrow(ax, 8.10, Y_BOT, otimes_c - 0.16, Y_BOT)
    # skip line (lane 1): S bypasses the conv and multiplies the attention
    # map -- kept under the channel-attention group only.
    Y_LANE1 = Y_BOT - 1.30
    arrow(ax, 5.30, Y_BOT - 0.475, 5.30, Y_LANE1, style="-")
    arrow(ax, 5.30, Y_LANE1, otimes_c, Y_LANE1, style="-")
    arrow(ax, otimes_c, Y_LANE1, otimes_c, Y_BOT - 0.16)
    arrow(ax, otimes_c + 0.16, Y_BOT, gx1 + 0.62, Y_BOT)

    # -------------------------------------------- spatial attention group ---
    # 0.62 gap between the two group borders (was 0.4 in v1) so the
    # connecting arrow + both dashed edges are no longer crowded together.
    sx0 = gx1 + 0.62
    pool_cx = sx0 + 0.25 + 0.59
    conv2_cx = pool_cx + 0.59 + 0.35 + 0.55
    otimes_s = conv2_cx + 0.55 + 0.35 + 0.16   # == X_FINAL, set below
    sx1 = otimes_s + 0.16 + 0.30
    ax.add_patch(FancyBboxPatch((sx0, Y_BOT - 1.06), sx1 - sx0, 2.12,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="none", ec=C_GROUP, lw=1.0,
                                linestyle=(0, (4, 3)), zorder=1))
    ax.text((sx0 + sx1) / 2, Y_BOT + 1.24, "Spatial Attention",
            ha="center", fontsize=8.2, color=C_GROUP)

    box(ax, pool_cx, Y_BOT, 1.18, 1.0,
        "Channel\nAvg / Max Pool\n+ Concat", shape="H×W×2",
        fc=C_POOL, fs=7.6)
    arrow(ax, pool_cx + 0.59, Y_BOT, conv2_cx - 0.55, Y_BOT)
    box(ax, conv2_cx, Y_BOT, 1.1, 0.95, "7×7 Conv\n+ Sigmoid",
        shape="H×W×1", fc=C_CONV, fs=8.2)
    # the spatially-attended feature: multiply happens right after, drawn as
    # the vertical rise into the final multiply on the identity path
    otimes(ax, X_FINAL, Y_BOT)
    arrow(ax, conv2_cx + 0.55, Y_BOT, X_FINAL - 0.16, Y_BOT)
    # skip line (lane 2): channel-attended feature bypasses pooling/conv
    # into this multiply. Routed on a LOWER lane than lane 1 (with a clear
    # vertical gap between the two) so the two bypasses never look like one
    # continuous line, even though they run side by side.
    Y_LANE2 = Y_BOT - 1.55
    arrow(ax, otimes_c + 0.16, Y_LANE2, X_FINAL, Y_LANE2, style="-")
    arrow(ax, X_FINAL, Y_LANE2, X_FINAL, Y_BOT - 0.16)
    arrow(ax, otimes_c + 0.16, Y_BOT - 0.16, otimes_c + 0.16, Y_LANE2,
          style="-")
    ax.text((otimes_c + X_FINAL) / 2, Y_LANE2 - 0.22,
            "channel-attended feature", ha="center",
            fontsize=FS_SHAPE, color="#595959", style="italic")

    # rise to the final multiply on the identity path
    otimes(ax, X_FINAL, Y_TOP)
    arrow(ax, X_FINAL, Y_BOT + 0.16, X_FINAL, Y_TOP - 0.16)
    ax.text(X_FINAL + 0.08, (Y_BOT + Y_TOP) / 2, "attention map\nH×W×3",
            ha="left", va="center", fontsize=FS_SHAPE, color="#595959",
            style="italic")

    # ------------------------------------------------------------ output ---
    # OUT_CX must clear the spatial-attention group's right border (sx1)
    # with margin so the box doesn't clip the dashed boundary.
    OUT_CX = max(14.30, sx1 + 0.30 + 0.975)
    arrow(ax, X_FINAL + 0.16, Y_TOP, OUT_CX - 0.90, Y_TOP)
    # v2 rename: v1 called this "SGA-Attended Input", which reads as if it
    # were still an input -- it is in fact the module's OUTPUT (d0 recalibrated
    # by the attention map). Renamed to match the x' notation used in Fig. 1.
    box(ax, OUT_CX, Y_TOP, 1.65, 1.0, "SGA Output\n$x'$", shape="H×W×3",
        fc=C_OUT)
    arrow(ax, OUT_CX, Y_TOP - 0.50, OUT_CX, Y_TOP - 1.45)
    box(ax, OUT_CX, Y_TOP - 2.05, 1.95, 1.05,
        "U-Net Encoder $d_1$\n4×4 Conv, s=2\n+ LeakyReLU",
        shape="H/2×W/2×64", fc=C_CONV, fs=8.0)
    ax.text(OUT_CX, Y_TOP - 2.70, "(next stage, not part of SGA — see Fig. 1)",
            ha="center", fontsize=FS_SHAPE, color="#595959", style="italic")

    # ===================================================== zone banners ===
    # Background bands making the INPUT -> PROCESS -> OUTPUT flow explicit
    # at a glance. OUTPUT is kept to the top row only, so it does NOT tint
    # the U-Net Encoder box below it (that box is the next module, outside
    # the SGA boundary, and is intentionally left unzoned).
    # v2.1 fix: v2's band top (5.65) sat only ~0.1 above the Input/Output box
    # tops (5.4), so the bold INPUT/PROCESS/OUTPUT captions visually crowded
    # those boxes. Canvas grew taller (6.2->6.6) and the title moved up to
    # 6.35, freeing room to raise the band top to 6.05 -- now ~0.5 of
    # clearance on both sides of every caption.
    Y_ZONE_LO, Y_ZONE_HI = 0.55, 6.05
    zone(ax, 0.15, 2.30, Y_ZONE_LO, Y_ZONE_HI, "INPUT", C_ZONE_IN)
    zone(ax, 2.30, X_FINAL + 0.35, Y_ZONE_LO, Y_ZONE_HI, "PROCESS", C_ZONE_PR)
    zone(ax, X_FINAL + 0.35, OUT_CX + 1.10, Y_TOP - 0.70, Y_ZONE_HI,
         "OUTPUT", C_ZONE_OUT)

    # ------------------------------------------------------------- title ---
    ax.text(8.20, 6.35, "SGA Module — Sobel-Guided Attention (input stage of the Pix2Pix U-Net generator)",
            ha="center", fontsize=FS_TITLE, weight="bold")

    fig.savefig(r"D:\Contest\AI GO\matherial\sga_module_architecture_v2.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    print("saved: sga_module_architecture_v2.png")


if __name__ == "__main__":
    main()
