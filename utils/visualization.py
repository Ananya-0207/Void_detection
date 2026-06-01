"""
utils/visualization.py
=======================
Visualisation helpers: 4-panel comparisons and training curves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List


VOID_COLOUR = (255, 80, 80)   # red-ish tint for void overlay (R, G, B)


def overlay_mask(image: np.ndarray,
                 mask:  np.ndarray,
                 alpha: float = 0.45,
                 colour: tuple = VOID_COLOUR) -> np.ndarray:
    """
    Blend a coloured tint onto the image wherever mask == 1.

    image : [H, W, 3] uint8 RGB
    mask  : [H, W]    uint8 0/1
    """
    out = image.copy().astype(np.float32)
    where = mask == 1
    for c, v in enumerate(colour):
        out[:, :, c][where] = alpha * v + (1 - alpha) * out[:, :, c][where]
    return out.astype(np.uint8)


def compare_predictions(image:     np.ndarray,
                         gt_mask:   np.ndarray,
                         pred_mask: np.ndarray,
                         title:     str = "",
                         save_path: Optional[str] = None,
                         show:      bool = False) -> plt.Figure:
    """
    4-panel figure:
      [Original X-ray] | [Ground Truth mask] | [Prediction mask] | [Prediction overlay]

    Parameters
    ----------
    image, gt_mask, pred_mask  : numpy arrays (raw, un-normalised)
    title                      : figure title
    save_path                  : if given, save PNG there
    show                       : call plt.show() if True
    """
    gt_vis   = (gt_mask   * 255).astype(np.uint8)
    pred_vis = (pred_mask * 255).astype(np.uint8)
    ov       = overlay_mask(image, pred_mask)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, (img, lbl, cmap) in zip(axes, [
        (image,    "Original X-ray",    "gray"),
        (gt_vis,   "Ground Truth",      "gray"),
        (pred_vis, "Prediction",        "gray"),
        (ov,       "Prediction Overlay", None ),
    ]):
        ax.imshow(img, cmap=cmap)
        ax.set_title(lbl, color="white", fontsize=11, pad=5)
        ax.axis("off")

    if title:
        fig.suptitle(title, color="white", fontsize=12, y=1.01)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight",
                    dpi=150, facecolor=fig.get_facecolor())

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_training_curves(train_losses: List[float],
                          val_losses:   List[float],
                          val_mious:    List[float],
                          val_cpas:     List[float],
                          val_recalls:  List[float],
                          save_path:    Optional[str] = None,
                          show:         bool = False) -> plt.Figure:
    """
    Two-panel figure: Loss curves (left) + Validation metrics (right).
    Metrics shown: mIOU, CPA (Precision), Recall — same as paper Table 2.
    """
    epochs = range(1, len(train_losses) + 1)

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#16213e")
    ax1.plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, val_losses,   "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Loss",  color="white")
    ax1.set_title("Loss Curves", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#16213e", labelcolor="white")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#444")

    # ── Metrics ───────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#16213e")
    ax2.plot(epochs, [v*100 for v in val_mious],
             "g-o", markersize=3, label="mIOU (%)")
    ax2.plot(epochs, [v*100 for v in val_cpas],
             "m-o", markersize=3, label="CPA / Precision (%)")
    ax2.plot(epochs, [v*100 for v in val_recalls],
             "y-o", markersize=3, label="Recall (%)")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Score (%)", color="white")
    ax2.set_title("Validation Metrics", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#16213e", labelcolor="white")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight",
                    dpi=150, facecolor=fig.get_facecolor())
        print(f"Training curves saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig
