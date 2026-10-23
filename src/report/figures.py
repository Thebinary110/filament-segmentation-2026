"""Report figures from the current best ensemble (b+a+c, 12 views; validation set).

Organizer-required (self-evaluation notebook, COMPETITION_INFO.md "Evaluation"):
  fig_iou_dist.png, fig_dice_dist.png, fig_m2n.png
Ours:
  fig_pq_vs_threshold.png    in-sample PQ and the spread over 10 split-half halves
  fig_recall_bins.png        recall by GT contrast, r/R and area
  fig_observatory_pq.png     per-observatory PQ
  fig_panels.png             qualitative: found / missed (low contrast) / annotator disagreement
Palette: reference categorical slots (validated: scripts/validate_palette.js, light surface).
"""
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import (BASE, SITES, W, baseline, decode, gt_for_tune, image_path, iou_matrix, load_val_gt,  # noqa: E402
                    pq_score)
import tune  # noqa: E402
from multiprocessing import Pool  # noqa: E402

OUT = W / "report_figs"
OUT.mkdir(exist_ok=True)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 10, "axes.titleweight": "bold"})


def pairs_and_m2n(gt, preds):
    ious, gt_deg, pred_deg = [], [], []
    for stem, recs in gt.items():
        p = preds.get(stem, [])
        for g in recs.values():
            if not p:
                gt_deg += [0] * len(g)
                continue
            m = iou_matrix(g, p)
            ious += m[m > 0].tolist()
            gt_deg += (m > 0).sum(1).tolist()
            pred_deg += (m > 0).sum(0).tolist()
    return np.array(ious), gt_deg, pred_deg


def hist(values, title, xlabel, fname):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bins = np.linspace(0, 1, 51)
    c, e = np.histogram(values, bins)
    ax.bar((e[:-1] + e[1:]) / 2, c / c.sum(), width=0.018, color=BLUE, linewidth=0)
    ax.set_ylim(0, c.max() / c.sum() * 1.25)
    top = ax.get_ylim()[1]
    ax.axvline(values.mean(), color=ORANGE, lw=2, ls="--")
    ax.text(values.mean() + 0.015, top * 0.93, f"mean {values.mean():.3f}", color=INK2, ha="left", va="top")
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.text(0.485, top * 0.93, "match\nthreshold 0.5", color=MUTED, fontsize=8, ha="right", va="top")
    ax.set(title=title, xlabel=xlabel, ylabel="share of overlapping GT-prediction pairs")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)


def m2n_plot(gt_deg, pred_deg):
    g, p = Counter(gt_deg), Counter(pred_deg)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    xs_g = sorted(g)
    xs_p = sorted(p)
    bg = ax.bar([x + 0.5 for x in xs_g], [g[x] for x in xs_g], width=0.8,
                color=[RED if x == 0 else AQUA for x in xs_g], linewidth=0)
    bp = ax.bar([-(x + 0.5) for x in xs_p], [p[x] for x in xs_p], width=0.8,
                color=[ORANGE if x == 0 else BLUE for x in xs_p], linewidth=0)
    for rect, n in list(zip(bg, [g[x] for x in xs_g])) + list(zip(bp, [p[x] for x in xs_p])):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), str(n), ha="center", va="bottom", fontsize=7, color=INK2)
    ticks = [-(x + 0.5) for x in xs_p] + [x + 0.5 for x in xs_g]
    ax.set_xticks(ticks, [f"{x}:1" for x in xs_p] + [f"1:{x}" for x in xs_g], fontsize=8)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set(title="Many-to-one and one-to-many relations (IoU > 0)", xlabel="left: predictions by GTs touched · right: GTs by predictions touched",
           ylabel="count")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AQUA, label="GT hit (1:n, n>0)"), Patch(color=RED, label="GT missed (1:0)"),
                       Patch(color=BLUE, label="prediction hit (n:1, n>0)"), Patch(color=ORANGE, label="prediction missed (0:1)")],
              fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig_m2n.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pq_vs_threshold(fused, gt):
    gtt = gt_for_tune(gt)
    stems = sorted(gtt)
    ths = [0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4, 0.45, 0.5]
    with Pool(12, initializer=tune._init, initargs=(fused, gtt)) as pool:
        per = {t: dict(zip(stems, pool.map(tune._score_image, [(s, dict(BASE, conf_thr=t)) for s in stems]))) for t in ths}

    def pq(t, sub):
        s, tp, fp, fn = (sum(per[t][x][i] for x in sub) for i in range(4))
        return s / (tp + 0.5 * fp + 0.5 * fn)
    full = [pq(t, stems) for t in ths]
    halves = []
    for seed in range(5):
        sh = stems[:]
        random.Random(seed).shuffle(sh)
        halves += [sh[:len(sh) // 2], sh[len(sh) // 2:]]
    H = np.array([[pq(t, h) for t in ths] for h in halves])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.fill_between(ths, H.min(0), H.max(0), color=BLUE, alpha=0.15, linewidth=0, label="range over 10 split-half halves")
    ax.plot(ths, full, color=BLUE, lw=2, marker="o", ms=4, label="all 106 validation images")
    i = int(np.argmax(full))
    ax.annotate(f"best {full[i]:.4f} at {ths[i]}", (ths[i], full[i]), xytext=(ths[i] + 0.03, full[i] + 0.012), color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set(title="PQ vs confidence threshold (3-model ensemble)", xlabel="fused confidence threshold", ylabel="local PQ")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pq_vs_threshold.png", dpi=200)
    plt.close(fig)
    return pd.DataFrame(dict(threshold=ths, PQ_all=full, PQ_half_min=H.min(0), PQ_half_max=H.max(0)))


def recall_bins():
    H = pd.read_csv(W / "analysis" / "c2_gt_hits.csv")
    col = "ensemble b+a+c (baseline)"
    specs = [("contrast", [-1, 0.05, 0.1, 0.15, 0.2, 1], ["<0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", ">0.20"], "GT contrast (ring - inside) / ring"),
             ("r_rel", [0, 0.5, 0.7, 0.85, 2], ["<0.5", "0.5-0.7", "0.7-0.85", ">0.85"], "GT distance from disk centre (r/R)"),
             ("area", [0, 500, 1000, 2000, 4000, 1e9], ["<500", "500-1k", "1k-2k", "2k-4k", ">4k"], "GT area (px)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, (c, b, l, xl) in zip(axes, specs):
        d = H.assign(bin=pd.cut(H[c], b, labels=l)).groupby("bin", observed=True).agg(n=(col, "size"), r=(col, "mean"))
        bars = ax.bar(range(len(d)), d.r, color=BLUE, width=0.7, linewidth=0)
        for rect, (r, n) in zip(bars, zip(d.r, d.n)):
            ax.text(rect.get_x() + rect.get_width() / 2, r + 0.015, f"{r:.2f}\nn={n}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(range(len(d)), d.index, fontsize=8)
        ax.set(xlabel=xl, ylim=(0, 1.05))
    axes[0].set_ylabel("recall (GT instances hit, IoU > 0.5)")
    fig.suptitle("Recall of the 3-model ensemble by filament property (validation)", fontsize=10, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_recall_bins.png", dpi=200)
    plt.close(fig)


def observatory(gt, preds):
    rows = []
    for s in "BCLMTU":
        stems = [x for x in gt if x[14] == s]
        pq, _ = pq_score(preds, gt, stems)
        rows.append((f"{SITES[s]} ({s})", pq, len(stems)))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    bars = ax.barh(range(len(rows)), [r[1] for r in rows], color=BLUE, height=0.6, linewidth=0)
    for rect, (_, v, n) in zip(bars, rows):
        ax.text(v + 0.005, rect.get_y() + rect.get_height() / 2, f"{v:.3f}  ({n} images)", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.set(xlim=(0, 0.62), xlabel="local PQ", title="PQ by observatory (3-model ensemble, validation)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_observatory_pq.png", dpi=200)
    plt.close(fig)


def panels(gt, preds):
    H = pd.read_csv(W / "analysis" / "c2_gt_hits.csv")
    col = "ensemble b+a+c (baseline)"
    found = H[H[col] & (H.contrast > 0.15) & (H.area > 1500)].iloc[0]

    def untouched(row):  # missed GT with no prediction overlapping it at all
        g = min((decode(c) for c in gt[row.stem][row.rec]), key=lambda m: abs(int(m.sum()) - row.area))
        return all(not (decode(c) & g).any() for c in preds.get(row.stem, []))
    cand = H[(~H[col]) & (H.contrast < 0.08) & (H.area > 800)]
    missed = next(r for _, r in cand.iterrows() if untouched(r))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9))
    for ax, row, title in ((axes[0], found, "Found: high contrast"), (axes[1], missed, "Missed: low contrast, no prediction near it")):
        gray = cv2.imread(str(image_path(row.stem)), cv2.IMREAD_GRAYSCALE)
        g = min((decode(c) for c in gt[row.stem][row.rec]), key=lambda m: abs(int(m.sum()) - row.area))
        ys, xs = np.nonzero(g)
        cy, cx, h = int(ys.mean()), int(xs.mean()), 180
        y0, x0 = np.clip(cy - h, 0, 2048 - 2 * h), np.clip(cx - h, 0, 2048 - 2 * h)
        ax.imshow(gray[y0:y0 + 2 * h, x0:x0 + 2 * h], cmap="gray")
        ax.contour(g[y0:y0 + 2 * h, x0:x0 + 2 * h], levels=[0.5], colors=[RED], linewidths=1)
        for c in preds.get(row.stem, []):
            m = decode(c)[y0:y0 + 2 * h, x0:x0 + 2 * h]
            if m.any():
                ax.contour(m, levels=[0.5], colors=[AQUA], linewidths=1)
        ax.set_title(f"{title}\n{row.stem}, contrast {row.contrast:.3f}", fontsize=8)
        ax.axis("off")
    s = [x for x in gt if len(gt[x]) == 3][3]
    gray = cv2.imread(str(image_path(s)), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)
    axes[2].imshow(small, cmap="gray")
    recs = sorted(gt[s])
    for r, colr in zip(recs, (BLUE, ORANGE, AQUA)):
        m = np.zeros((2048, 2048), np.uint8)
        for c in gt[s][r]:
            m |= decode(c).astype(np.uint8)
        m = cv2.resize(cv2.dilate(m, np.ones((5, 5), np.uint8)), (1024, 1024), interpolation=cv2.INTER_NEAREST)
        axes[2].contour(m, levels=[0.5], colors=[colr], linewidths=0.8)
    axes[2].set_title(f"Annotator disagreement (full disk, 3 annotators)\n{s}: "
                      f"{' / '.join(str(len(gt[s][r])) for r in recs)} filaments drawn", fontsize=8)
    axes[2].axis("off")
    fig.text(0.01, 0.01, "Panels 1-2: red = annotator outline, aqua = our prediction. "
             "Panel 3: one colour per annotator (blue / orange / aqua).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "fig_panels.png", dpi=200)
    plt.close(fig)


def main():
    gt = load_val_gt()
    final, fused = baseline()
    preds = {k: [c for _, c in v] for k, v in final.items()}
    pq, st = pq_score(preds, gt)
    ious, gt_deg, pred_deg = pairs_and_m2n(gt, preds)
    hist(ious, f"IoU distribution (ensemble vs annotators; PQ {pq:.4f})", "IoU", "fig_iou_dist.png")
    hist(2 * ious / (1 + ious), "Dice distribution (ensemble vs annotators)", "Dice", "fig_dice_dist.png")
    m2n_plot(gt_deg, pred_deg)
    T = pq_vs_threshold(fused, gt)
    recall_bins()
    observatory(gt, preds)
    panels(gt, preds)
    T.to_csv(OUT / "pq_vs_threshold.csv", index=False)
    print("figures in", OUT, "| PQ", round(pq, 4), "| pairs", len(ious))


if __name__ == "__main__":
    main()
