"""
utils/mask_utils.py
====================
Convert a Labelme JSON annotation into a binary numpy mask — entirely
in memory, no intermediate files needed.

Supported shape types
---------------------
  polygon    — filled polygon (most common; used in your dataset)
  rectangle  — two corner points → filled rectangle
  point      — small filled circle (radius = 5 px)

Output mask values
------------------
  0  = background
  1  = void  (or whatever label is passed as target_label)
"""

import json
import numpy as np
import cv2
from pathlib import Path


def _rect_to_poly(pts):
    """Two-point Labelme rectangle → 4-point polygon array."""
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    xmin, xmax = int(min(x1, x2)), int(max(x1, x2))
    ymin, ymax = int(min(y1, y2)), int(max(y1, y2))
    return np.array([[xmin, ymin], [xmax, ymin],
                     [xmax, ymax], [xmin, ymax]], dtype=np.int32)


def _poly_to_arr(pts):
    """List of [x, y] → int32 numpy array."""
    return np.array([[int(p[0]), int(p[1])] for p in pts], dtype=np.int32)


def json_to_mask(json_path: str,
                 target_label: str = "void",
                 point_radius: int = 5) -> np.ndarray:
    """
    Parse one Labelme JSON file and return a binary uint8 mask.

    Parameters
    ----------
    json_path     : full path to the .json annotation file
    target_label  : only shapes with this label are drawn (default 'void')
    point_radius  : radius in pixels for 'point' shape type

    Returns
    -------
    mask : np.ndarray of shape (H, W), dtype=uint8, values 0 or 1
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    H = data["imageHeight"]
    W = data["imageWidth"]
    mask = np.zeros((H, W), dtype=np.uint8)

    for shape in data.get("shapes", []):
        if shape["label"] != target_label:
            continue

        stype = shape["shape_type"]
        pts   = shape["points"]

        if stype == "polygon":
            poly = _poly_to_arr(pts)
            if len(poly) >= 3:                          # need ≥ 3 points
                cv2.fillPoly(mask, [poly], color=1)

        elif stype == "rectangle":
            poly = _rect_to_poly(pts)
            cv2.fillPoly(mask, [poly], color=1)

        elif stype == "point":
            cx, cy = int(pts[0][0]), int(pts[0][1])
            cv2.circle(mask, (cx, cy), point_radius, color=1, thickness=-1)

        # line / linestrip are ignored (not relevant for void detection)

    return mask   # values: 0 or 1


def mask_to_visual(mask: np.ndarray) -> np.ndarray:
    """Convert 0/1 mask to 0/255 uint8 image for display or saving."""
    return (mask * 255).astype(np.uint8)
