"""
Draw the overall architecture diagram for the CVGIP paper (Figure 1).

Purpose — Render a publication-quality block diagram showing the full
          training/inference pipeline: input image -> SGA module (see Fig. 2
          for internals) -> Pix2Pix U-Net generator -> output, plus the
          training-time supervision (PatchGAN discriminator -> L_adv, L1
          reconstruction loss -> L_L1, combined into L_total). Matches the
          notation used in cvgip2025_chinese.py SS3.1/3.5 (x, x', T_hat, y,
          L_adv, L_L1, L_total, lambda=100).
Args    — None (paths are hard-coded; run from any cwd).
Returns — Saves 'overall_architecture.png' (300 dpi) into this folder.
Raises  — IOError if the output folder is not writable.
Notes   — Uses the same color palette and box/arrow helpers as
          draw_sga_architecture.py for visual consistency between Fig. 1
          and Fig. 2. Layout is two rows (inference pipeline on top,
          training objective in the dashed box below) on a wide canvas so
          the diagram reads as a flat, landscape block diagram rather than
          a tall stack.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- palette ---
C_INPUT = "#D6E4F0"   # light blue   — image tensors / dataset
C_SOBEL = "#FCE4D6"   # light orange — SGA module / adversarial loss
C_CONV  = "#E2EFDA"   # light green  — learnable network components
C_POOL  = "#FFF2CC"   # light yellow — discriminator
C_OUT   = "#E8DFF0"   # light purple — outputs / final loss
C_EDGE  = "#404040"
C_GROUP = "#7F7F7F"
C_THAT  = "#8064A2"   # saturated purple — all lines carrying T_hat
C_YLINE = "#4F81BD"   # saturated blue   — all lines carrying y

FS_TITLE = 11.5
FS_BOX   = 9.0
FS_SHAPE = 7.6


def box(ax, cx, cy, w, h, label, shape=None, fc=C_CONV, fs=FS_BOX, ls="solid"):
    """Rounded box centered at (cx, cy) with a label and optional shape note."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                 boxstyle="round,pad=0.012,rounding_size=0.06",
                                 fc=fc, ec=C_EDGE, lw=1.1, zorder=3,
                                 linestyle=ls))
    ax.text(cx, cy + (0.13 if shape else 0.0), label, ha="center", va="center",
            fontsize=fs, zorder=4, linespacing=1.25)
    if shape:
        ax.text(cx, cy - h / 2 + 0.20, shape, ha="center", va="center",
                fontsize=FS_SHAPE, color="#595959", style="italic",
                linespacing=1.2, zorder=4)


def arrow(ax, x1, y1, x2, y2, style="-|>", color=C_EDGE, lw=1.3, rad=0.0,
          ls="solid"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle=style, mutation_scale=13,
                                  color=color, lw=lw, zorder=2, linestyle=ls,
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
    fig, ax = plt.subplots(figsize=(14.45, 6.6))
    ax.set_xlim(0, 14.55)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    Y_TOP = 4.65   # inference pipeline (runs at deployment)
    Y_BOT = 1.25   # training objective: discriminator + losses

    # ===================================================== zone banners ===
    # Background bands making the INPUT -> PROCESS -> OUTPUT flow explicit
    # at a glance, independent of (and behind) every box below.
    # (the training-objective area below already has its own dashed border
    # from the existing group box, so it does not need a 4th zone outline)
    # v2.1 fix: v2's zone top (5.55) sat only ~0.05 below the "compared
    # (paired sample)" annotation at 5.60, so that text rendered crossed
    # right through the OUTPUT zone's top border. Raised to 6.05 (title
    # moved up to 6.40 below) for real clearance, matching Fig. 2's bands.
    ZONE_TOP = 6.05
    zone(ax, 0.15, 2.30, 2.45, ZONE_TOP, "INPUT", C_YLINE)
    zone(ax, 2.30, 7.95, 2.45, ZONE_TOP, "PROCESS", "#70AD47")
    zone(ax, 7.95, 12.50, 2.45, ZONE_TOP, "OUTPUT", C_THAT)

    # ============================================================ Row 1 ===
    # Generation / inference pipeline: x -> SGA -> U-Net G -> T_hat
    box(ax, 1.10, Y_TOP, 1.6, 1.10, "Input Image\n$x$",
        shape="$H\\times W\\times3$\n(with reflection)", fc=C_INPUT)
    arrow(ax, 1.90, Y_TOP, 2.30, Y_TOP)

    box(ax, 3.30, Y_TOP, 2.0, 1.40, "SGA Module",
        shape="Sobel-Guided Attention\n$\\rightarrow x'$ (see Fig. 2)",
        fc=C_SOBEL)
    arrow(ax, 4.30, Y_TOP, 4.80, Y_TOP)

    box(ax, 6.15, Y_TOP, 2.7, 1.50, "U-Net Generator $G$",
        shape="Encoder $\\times7$ / Decoder $\\times7$\nskip connections, tanh",
        fc=C_CONV)
    ax.text(6.15, Y_TOP + 0.92, "Pix2Pix Generator", ha="center", va="center",
            fontsize=FS_SHAPE, color="#595959", style="italic")
    arrow(ax, 7.50, Y_TOP, 8.00, Y_TOP)

    box(ax, 8.85, Y_TOP, 1.7, 1.00, "Output $\\hat{T}$",
        shape="$H\\times W\\times3$\n(reflection-removed)", fc=C_OUT)

    # Ground Truth y: paired training-only input. Drawn with a dashed
    # outline (vs. the solid outlines of the inference-path boxes to its
    # left) so it reads as training-time-only without splitting the single
    # dashed "Training Objective" rectangle below into an L-shape.
    box(ax, 11.20, Y_TOP, 1.6, 0.95, "Ground Truth $y$",
        shape="$H\\times W\\times3$\n(paired sample)",
        fc=C_INPUT, ls=(0, (4, 3)))

    # "compared" connector between T_hat and y
    arrow(ax, 9.70, Y_TOP, 10.40, Y_TOP, style="<->", color=C_GROUP,
          ls=(0, (4, 3)))
    ax.text(10.05, Y_TOP + 0.95, "compared\n(paired sample)",
            ha="center", va="center", fontsize=FS_SHAPE, color="#595959",
            style="italic", linespacing=1.2)

    # =============================================== training-only group ===
    gx0, gx1 = 0.20, 14.50
    gy0, gy1 = 0.10, 2.30
    ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                 fc="none", ec=C_GROUP, lw=1.0,
                                 linestyle=(0, (4, 3)), zorder=1))
    ax.text((gx0 + gx1) / 2, gy1 - 0.18,
            "Pix2Pix Training Objective  (inference requires only $G$)",
            ha="center", fontsize=8.5, color=C_GROUP, style="italic")

    # ============================================================ Row 2 ===
    # PatchGAN discriminator (below G) and the two loss terms, left to right
    box(ax, 6.15, Y_BOT, 3.1, 1.45, "PatchGAN\nDiscriminator $D$",
        shape="$D(x,\\hat{T})$ vs $D(x,y)$\n$N\\times N$ patch realism",
        fc=C_POOL)

    box(ax, 8.90, Y_BOT, 2.0, 0.95, "$L_{adv}$ (MSE)",
        shape="$E[(D{-}1)^2]$\n$+\\,E[(D{-}0)^2]$", fc=C_SOBEL, fs=8.4)
    arrow(ax, 7.70, Y_BOT, 7.90, Y_BOT)

    box(ax, 10.85, Y_BOT, 1.6, 0.95, "$L_{L1}$ (MAE)",
        shape="$E[\\,\\|y-\\hat{T}\\|_1\\,]$\n$\\lambda = 100$", fc=C_CONV)

    # v2.1 fix: v2 put "$L_{total}$" as the (large) main label and the actual
    # formula as a tiny italic shape-note underneath -- the formula, which is
    # the whole point of this box, was nearly unreadable. Both lines are now
    # the main label at the same (larger) font size, no shape note.
    L_TOTAL_CX = 13.15
    box(ax, L_TOTAL_CX, Y_BOT, 2.5, 1.05,
        "$L_{total}$\n$=L_{adv}+\\lambda L_{L1}$", fc=C_OUT, fs=10.5)

    # L_adv -> L_total, routed below L_L1's box (in the gap between the
    # row-2 boxes and the outer dashed border) so the connector is never
    # hidden behind the L_L1 box, which sits at a higher zorder.
    # v2 fix: v1 only put an arrowhead on the final (upward) segment, so the
    # first two legs read as a static line rather than a directed flow.
    # Every leg now has its own arrowhead, and the line is thicker + colored
    # to match L_total's box so it reads as one continuous, directed wire.
    # v2.1: dropped the redundant "$L_{adv} \to L_{total}$" caption -- with
    # arrowheads on every leg, plus the formula now legible on the box
    # itself, the text label only repeated the same fact twice.
    Y_WIRE = 0.40
    C_WIRE = "#C0504D"   # warm red-orange, ties the wire back to L_adv's box
    arrow(ax, 8.90, Y_BOT - 0.475, 8.90, Y_WIRE, style="-|>", color=C_WIRE,
          lw=1.6)
    arrow(ax, 8.90, Y_WIRE, L_TOTAL_CX, Y_WIRE, style="-|>", color=C_WIRE,
          lw=1.6)
    arrow(ax, L_TOTAL_CX, Y_WIRE, L_TOTAL_CX, Y_BOT - 0.525, style="-|>",
          color=C_WIRE, lw=1.6)
    # L_L1 -> L_total: v2.1 fix -- v2's gap here was only 0.10 units (vs.
    # ~0.15-0.2 for every other connector in this row), so the arrow rendered
    # almost invisibly small. L_total moved right (12.90 -> 13.15) to open up
    # a normal-sized 0.35-unit gap for this arrow.
    arrow(ax, 11.65, Y_BOT, 11.90, Y_BOT)

    # --------------------------------------------------- routed connectors ---
    # v2 fix: the v1 diagram sent 5 separate curves (x, That->D, y->D,
    # That->L_L1, y->L_L1) straight through the same neutral zone, crossing
    # each other, with "That"/"y" labelled twice each -> unreadable. v2 bundles
    # each source into a single color-coded stem (one label each) that then
    # forks to its two destinations, so the eye only has to follow one
    # colored line per source instead of disentangling a 5-way crossing.
    D_TOP = Y_BOT + 0.725     # D box top edge
    L1_TOP = Y_BOT + 0.475    # L_L1 box top edge

    # x (conditional input) -> D : unchanged, dashed gray, leftmost entry
    arrow(ax, 1.10, Y_TOP - 0.55, 4.90, D_TOP, style="-|>",
          color=C_GROUP, ls=(0, (4, 3)), rad=-0.12)
    # v2.1 fix: v2's label sat at x=2.00, almost on top of the INPUT/PROCESS
    # zone boundary (x=2.30) -- the boundary line ran right through the
    # text. Moved further along the same dashed line, into open space well
    # inside the PROCESS zone.
    ax.text(3.30, 3.00, "conditional\ninput $x$", ha="center", va="center",
            fontsize=FS_SHAPE, color="#595959", style="italic",
            linespacing=1.2)

    # T_hat: one stem down from the Output box to a junction, labelled once,
    # then forks left to D and right to L_L1. Entry points on D/L_L1 are
    # ordered left-to-right to match source order, so this fork never
    # crosses the y fork below.
    JX_T, JY_T = 8.85, 3.15
    arrow(ax, 8.85, Y_TOP - 0.50, JX_T, JY_T, style="-", color=C_THAT, lw=1.5)
    ax.text(JX_T + 0.32, JY_T, "$\\hat{T}$", ha="left", va="center",
            fontsize=9.0, color=C_THAT, weight="bold")
    arrow(ax, JX_T, JY_T, 6.15, D_TOP, color=C_THAT, lw=1.3, rad=-0.18)
    arrow(ax, JX_T, JY_T, 10.40, L1_TOP, color=C_THAT, lw=1.3, rad=0.18)

    # y: same pattern, blue, junction sits closer to its source so the two
    # forks stay clear of T_hat's fork.
    JX_Y, JY_Y = 11.20, 3.40
    arrow(ax, 11.20, Y_TOP - 0.475, JX_Y, JY_Y, style="-", color=C_YLINE,
          lw=1.5)
    ax.text(JX_Y + 0.30, JY_Y, "$y$", ha="left", va="center",
            fontsize=9.0, color=C_YLINE, weight="bold")
    arrow(ax, JX_Y, JY_Y, 7.40, D_TOP, color=C_YLINE, lw=1.3, rad=-0.30)
    arrow(ax, JX_Y, JY_Y, 11.30, L1_TOP, color=C_YLINE, lw=1.3, rad=0.05)

    # ------------------------------------------------------------- title ---
    ax.text(7.275, 6.40,
            "Overall Architecture — Pix2Pix + SGA Reflection Removal Pipeline",
            ha="center", fontsize=FS_TITLE, weight="bold")

    fig.savefig(r"D:\Contest\AI GO\matherial\overall_architecture_v2.png",
                 dpi=300, bbox_inches="tight", facecolor="white")
    print("saved: overall_architecture_v2.png")


if __name__ == "__main__":
    main()
