import os
import sys
import csv
import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from dataset import VoidDataset, make_splits, get_val_transforms
from models.pcb_deeplabv3 import PCBDeepLabV3
from utils.metrics import MetricAccumulator, compute_metrics
from utils.visualization import compare_predictions

import matplotlib.pyplot as plt

'''
# Save prediction overlay

def save_prediction_overlay(image, gt_mask, pred_mask, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = np.array(image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("X-ray")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

'''

# Load checkpoint


def load_model(ckpt_path:str=None,
                device: torch.device=None)->PCBDeepLabV3:

    if device is None:
        device = torch.device(config.DEVICE)

    if ckpt_path is None:
        ckpt_path = os.path.join(config.CKPT_DIR, "best_model.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run train.py first to create it."
        )

    model = PCBDeepLabV3(
        num_classes=config.NUM_CLASSES,
        pretrained=False
    )

    ckpt = torch.load(
        ckpt_path,
        map_location=device
    )
    print(ckpt.keys())

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    #print(f"Saved at epoch {epoch} | val mIOU = {miou*100:.2f}%\n")
    return model



# Main evaluation


@torch.no_grad()
def evaluate(
    ckpt_path=None,
    save_visuals=True,
    vis_dir=None,
    csv_path=None,
):

    device = torch.device(config.DEVICE)

    if vis_dir is None:
        vis_dir = os.path.join(config.RESULTS_DIR,"test_visuals")

    if csv_path is None:
        csv_path = os.path.join(config.RESULTS_DIR,"test_metrics.csv")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs( vis_dir,exist_ok=True)

    model = load_model(ckpt_path,device)
# Get test dataset
    _, _, test_pairs = make_splits(
        dataset_dir=config.DATASET_DIR,
        train_size=config.TRAIN_SIZE,
        val_size=config.VAL_SIZE,
        test_size=config.TEST_SIZE,
    )

    test_ds_transformed = VoidDataset(test_pairs,transform=get_val_transforms())
    test_ds_raw = VoidDataset(test_pairs, transform=None)

    print( f"Evaluating {len(test_pairs)} test image(s)...\n")

    accumulator = MetricAccumulator()
    per_img_rows = []


    # Loop


    for idx in tqdm(range(len(test_ds_transformed))):
        # Model inference 
        img_t, mask_t = test_ds_transformed[idx]
        logits  = model(img_t.unsqueeze(0).to(device))     # [1, 2, H, W]
        pred    = logits.argmax(dim=1)[0].cpu().numpy()    # [H, W]
        gt      = mask_t.numpy()                           # [H, W]

        # Compute per-image metrics 
        m    = compute_metrics(pred, gt)
        accumulator.update(pred, gt)
        name = test_ds_raw.filename(idx)

        per_img_rows.append({
            "filename": name,
            "mIOU"    : round(m["mIOU"]   * 100, 2),
            "MPA"     : round(m["MPA"]    * 100, 2),
            "CPA"     : round(m["CPA"]    * 100, 2),
            "Recall"  : round(m["Recall"] * 100, 2),
            "Dice"    : round(m["Dice"]   * 100, 2),
            "F1"      : round(m["F1"]     * 100, 2),
        })


        # visualfigure
        if save_visuals:
            raw_img, raw_mask = test_ds_raw.get_raw_pair(idx)
            H, W = raw_img.shape[:2]

            # Resize predicted mask back to original image resolution for display
            pred_resized = cv2.resize(
                pred.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST
            )

            compare_predictions(
                image     = raw_img,
                gt_mask   = raw_mask,
                pred_mask = pred_resized,
                title     = (f"{name}  |  "
                             f"mIOU={m['mIOU']*100:.1f}%  "
                             f"CPA={m['CPA']*100:.1f}%  "
                             f"Recall={m['Recall']*100:.1f}%"),
                save_path = os.path.join(vis_dir, f"{name}.png"),
            )



    # Final metrics

    final = accumulator.print_summary("Test set")
    fields = ["filename","mIOU","MPA","CPA","Recall","Dice","F1","status","note"]

    with open(csv_path,"w",newline="",encoding="utf-8") as f:

         writer = csv.DictWriter(f, fieldnames=fields)
         writer.writeheader()
         writer.writerows(per_img_rows)
         writer.writerow({
            "filename": "─── MEAN ───",
            "mIOU"    : round(final["mIOU"]   * 100, 2),
            "MPA"     : round(final["MPA"]    * 100, 2),
            "CPA"     : round(final["CPA"]    * 100, 2),
            "Recall"  : round(final["Recall"] * 100, 2),
            "Dice"    : round(final["Dice"]   * 100, 2),
            "F1"      : round(final["F1"]     * 100, 2),
            "status"  : "OK",
            "note"    : "Warning: low confidence"
        })

    print(f"Per-image metrics → {csv_path}")
    if save_visuals:
        print(f"Visual results    → {vis_dir}/")

    return final

# CLI usage instructions
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate PCB-DeepLabV3 on the test split"
    )
    parser.add_argument(
        "--ckpt",       default=None,
        help="Path to .pth checkpoint (default: checkpoints/best_model.pth)"
    )
    parser.add_argument(
        "--no_visuals",  action="store_true",
        help="Skip saving visualisation figures"
    )
    parser.add_argument("--vis_dir",  default=None)
    parser.add_argument("--csv",      default=None)
    args = parser.parse_args()

    evaluate(
        ckpt_path    = args.ckpt,
        save_visuals = not args.no_visuals,
        vis_dir      = args.vis_dir,
        csv_path     = args.csv,
    )
#in previous version, the prediction overlay was above visual figure
'''
        # Save prediction image
        save_name = (
            os.path.splitext(name)[0]
            + "_pred.png"
        )
        raw_img, raw_mask = test_ds_raw.get_raw_pair(idx)

        save_prediction_overlay(
            image=raw_img,
            gt_mask=gt,
            pred_mask=pred,
            save_path=os.path.join(
                config.RESULTS_DIR,
                save_name
            ),
        )'
'''