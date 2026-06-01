<<<<<<< HEAD
# X-ray Void Detection — PCB-DeepLabV3

Implementation aligned with:
**"Segmentation of void defects in X-ray images of chip solder joints
based on PCB-DeepLabV3 algorithm"**
Kong et al., Scientific Reports (2024)

---

## What data do you need?

A **single flat folder** (e.g. `void_dataset/`) containing:
```
void_dataset/
    img_000.png
    img_000.json       ← Labelme annotation for img_000.png
    img_001.png
    img_001.json
    ...
```

**That's it.** No preprocessing, no extra directories.

The JSONs must be Labelme format (version ≥ 5) with shapes labelled **"void"**.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open config.py and set the ONE line you must change:
#    DATASET_DIR = "/path/to/your/void_dataset"
```

---

## Step 1 — Adjust config.py

Open `config.py` and edit these values:

```python
DATASET_DIR = "/path/to/void_dataset"   # ← YOUR folder path

# Split sizes — must sum to ≤ number of images in your folder
TRAIN_SIZE = 7    # with 9 images
VAL_SIZE   = 1
TEST_SIZE  = 1

# With 70 images (full dataset) use:
# TRAIN_SIZE = 50
# VAL_SIZE   = 10
# TEST_SIZE  = 10
```

---

## Step 2 — Verify data loads correctly

```bash
python dataset.py
```

Expected output:
```
Found 9 matched image/JSON pairs in /path/to/void_dataset
Split → train=7, val=1, test=1
  train  images=torch.Size([4, 3, 512, 512])  masks=torch.Size([4, 512, 512])
  val    images=torch.Size([1, 3, 512, 512])  masks=torch.Size([1, 512, 512])
  test   images=torch.Size([1, 3, 512, 512])  masks=torch.Size([1, 512, 512])
```

---

## Step 3 — Verify model builds

```bash
python models/pcb_deeplabv3.py
```

Expected output:
```
Input : torch.Size([1, 3, 512, 512])
Output: torch.Size([1, 2, 512, 512])
Params: ~3,500,000
```

---

## Step 4 — Train

```bash
python train.py
```

What happens:
- Auto-discovers and splits your images
- Trains for up to 100 epochs (early stops if val loss stagnates for 3 epochs)
- Prints per-epoch: loss, mIOU, CPA, Recall
- Saves `checkpoints/best_model.pth` when validation loss improves
- Saves `results/training_curves.png`

---

## Step 5 — Evaluate

```bash
python evaluate.py
```

What happens:
- Loads `checkpoints/best_model.pth`
- Runs inference on the test split
- Prints all metrics (mIOU, MPA, CPA, Recall, Dice, F1)
- Saves `results/test_metrics.csv`
- Saves `results/test_visuals/img_XXX.png` (4-panel: original/GT/prediction/overlay)

To evaluate a specific checkpoint:
```bash
python evaluate.py --ckpt checkpoints/last_model.pth
```

---


## File structure

```
xray_void_detection/
├── config.py              ← ★ Only file you need to edit
├── dataset.py             ← Loads images+JSON, splits, augments
├── train.py               ← Training loop
├── evaluate.py            ← Test evaluation + visuals + CSV
├── requirements.txt
├── models/
│   ├── channel_attention.py   ← SENet channel attention (paper §CAM)
│   ├── amtpnet.py             ← AMTPNet (paper's core contribution)
│   ├── aspp.py                ← Atrous Spatial Pyramid Pooling
│   └── pcb_deeplabv3.py       ← Full PCB-DeepLabV3 model
└── utils/
    ├── mask_utils.py          ← Labelme JSON → binary mask (in memory)
    ├── metrics.py             ← mIOU, MPA, CPA, Recall, Dice, F1
    └── visualization.py       ← 4-panel figures, training curves
```

---

## Metrics explained (from paper)

| Metric | Formula | What it means |
|--------|---------|---------------|
| **mIOU** | mean(IoU_void, IoU_bg) | Primary metric — higher is better |
| **MPA**  | mean(sensitivity, specificity) | Mean per-class accuracy |
| **CPA**  | TP/(TP+FP) | Precision — how accurate detections are |
| **Recall** | TP/(TP+FN) | How many actual voids are found |
| **Dice/F1** | 2·TP/(2·TP+FP+FN) | Balanced precision-recall score |

Paper results on their dataset (Table 3):
- mIOU = 81.69%  |  FPS = 78.81

---

