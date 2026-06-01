"""
utils/metrics.py
================
Evaluation metrics matching the PCB-DeepLabV3 paper (Table 2 & 3):

  mIOU   — mean Intersection over Union
  MPA    — Mean Pixel Accuracy
  CPA    — Class Pixel Accuracy  (= Precision for void class)
  Recall — Void-class recall

Extra metrics also computed:
  Dice / F1  — 2·TP / (2·TP + FP + FN)
  IoU_void   — IoU for the void class only
  IoU_bg     — IoU for the background class only
"""

import numpy as np
from typing import Dict


def _binary_stats(pred: np.ndarray, gt: np.ndarray):
    """
    Return TP, FP, FN, TN for the positive (void = 1) class.
    Both inputs are flat boolean or 0/1 integer arrays.
    """
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()
    tp = int(( p &  g).sum())
    fp = int(( p & ~g).sum())
    fn = int((~p &  g).sum())
    tn = int((~p & ~g).sum())
    return tp, fp, fn, tn


def compute_metrics(pred: np.ndarray,
                    gt:   np.ndarray,
                    eps:  float = 1e-7) -> Dict[str, float]:
    """
    Compute all metrics for one image.

    Parameters
    ----------
    pred : [H, W] predicted mask — integer class indices (0 or 1)
    gt   : [H, W] ground-truth mask — integer class indices (0 or 1)

    Returns
    -------
    dict with float values (in range 0–1, multiply by 100 for %)
    """
    tp, fp, fn, tn = _binary_stats(pred == 1, gt == 1)

    # ── Per-class IoU ─────────────────────────────────────────────────────────
    # Void class
    iou_void_num = tp
    iou_void_den = tp + fp + fn
    iou_void = iou_void_num / (iou_void_den + eps) if iou_void_den > 0 else float("nan")

    # Background class
    iou_bg_num = tn
    iou_bg_den = tn + fn + fp
    iou_bg = iou_bg_num / (iou_bg_den + eps) if iou_bg_den > 0 else float("nan")

    valid = [v for v in [iou_void, iou_bg] if not np.isnan(v)]
    miou  = float(np.mean(valid)) if valid else 0.0

    # ── CPA = Precision for void ──────────────────────────────────────────────
    cpa = tp / (tp + fp + eps)

    # ── Recall for void ───────────────────────────────────────────────────────
    recall = tp / (tp + fn + eps)

    # ── MPA = mean per-class accuracy ─────────────────────────────────────────
    acc_void = tp / (tp + fn + eps)   # sensitivity
    acc_bg   = tn / (tn + fp + eps)   # specificity
    mpa      = (acc_void + acc_bg) / 2.0

    # ── Dice = F1 ─────────────────────────────────────────────────────────────
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    return {
        "mIOU"    : miou,
        "MPA"     : mpa,
        "CPA"     : cpa,
        "Recall"  : recall,
        "Dice"    : dice,
        "F1"      : dice,
        "IoU_void": 0.0 if np.isnan(iou_void) else iou_void,
        "IoU_bg"  : 0.0 if np.isnan(iou_bg)   else iou_bg,
        # raw counts for aggregation
        "_tp": tp, "_fp": fp, "_fn": fn, "_tn": tn,
        "_iou_void": iou_void, "_iou_bg": iou_bg,
    }


class MetricAccumulator:
    """
    Accumulate metrics over a full dataset split.

    Usage
    -----
    acc = MetricAccumulator()
    for pred_np, gt_np in ...:
        acc.update(pred_np, gt_np)
    results = acc.compute()
    acc.print_summary("Test")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._tp, self._fp, self._fn, self._tn = 0, 0, 0, 0
        self._iou_void_list, self._iou_bg_list = [], []

    def update(self, pred: np.ndarray, gt: np.ndarray):
        m = compute_metrics(pred, gt)
        self._tp += m["_tp"]
        self._fp += m["_fp"]
        self._fn += m["_fn"]
        self._tn += m["_tn"]
        if not np.isnan(m["_iou_void"]):
            self._iou_void_list.append(m["_iou_void"])
        if not np.isnan(m["_iou_bg"]):
            self._iou_bg_list.append(m["_iou_bg"])

    def compute(self, eps: float = 1e-7) -> Dict[str, float]:
        tp, fp, fn, tn = self._tp, self._fp, self._fn, self._tn
        cpa    = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        mpa    = (tp / (tp + fn + eps) + tn / (tn + fp + eps)) / 2.0
        dice   = (2 * tp) / (2 * tp + fp + fn + eps)

        iou_void = float(np.mean(self._iou_void_list)) if self._iou_void_list else 0.0
        iou_bg   = float(np.mean(self._iou_bg_list))   if self._iou_bg_list   else 0.0
        miou     = (iou_void + iou_bg) / 2.0

        return {
            "mIOU"    : miou,
            "MPA"     : mpa,
            "CPA"     : cpa,
            "Recall"  : recall,
            "Dice"    : dice,
            "F1"      : dice,
            "IoU_void": iou_void,
            "IoU_bg"  : iou_bg,
        }

    def print_summary(self, split_name: str = "") -> Dict[str, float]:
        m = self.compute()
        tag = f" ({split_name})" if split_name else ""
        print(f"\n{'─'*48}")
        print(f"  Metrics{tag}")
        print(f"{'─'*48}")
        print(f"  {'mIOU':<12}: {m['mIOU']  *100:6.2f} %")
        print(f"  {'MPA':<12}: {m['MPA']   *100:6.2f} %")
        print(f"  {'CPA/Prec':<12}: {m['CPA']   *100:6.2f} %")
        print(f"  {'Recall':<12}: {m['Recall']*100:6.2f} %")
        print(f"  {'Dice/F1':<12}: {m['Dice']  *100:6.2f} %")
        print(f"  {'IoU_void':<12}: {m['IoU_void']*100:6.2f} %")
        print(f"  {'IoU_bg':<12}: {m['IoU_bg']  *100:6.2f} %")
        print(f"{'─'*48}\n")
        return m
