
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from dataset              import get_dataloaders
from models.pcb_deeplabv3 import PCBDeepLabV3
from utils.metrics        import MetricAccumulator
from utils.visualization  import plot_training_curves



#  Loss functions


class DiceLoss(nn.Module):
  
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # void class probability
        probs  = F.softmax(logits, dim=1)[:, 1, ...]   # [B, H, W]
        tgt    = targets.float()
        inter  = (probs * tgt).sum(dim=(1, 2))
        denom  = probs.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        dice   = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)



#  One epoch helpers


def run_epoch(model, loader, optimizer, criterion, device, epoch, mode="train"):
    training = (mode == "train")
    model.train() if training else model.eval()

    total_loss = 0.0
    acc = MetricAccumulator()

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"[{mode.upper():<5}] Epoch {epoch}", leave=False)
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss   = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy()
            gts   = masks.cpu().numpy()
            for p, g in zip(preds, gts):
                acc.update(p, g)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader), acc.compute()



#  Main training routine


def train():
    #Device configuration
    device = torch.device(config.DEVICE)
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    #  Reproducibility 
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Data 
    print(f"\nLoading data from: {config.DATASET_DIR}")
    loaders = get_dataloaders(
        dataset_dir = config.DATASET_DIR,
        batch_size  = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
    )

    #  Model 
    print("\nBuilding PCB-DeepLabV3 …")
    model = PCBDeepLabV3(num_classes=config.NUM_CLASSES,
                          pretrained=config.PRETRAINED).to(device)
    print(f"  Trainable parameters: {model.count_params():,}")

    # For very small datasets (< 30 images), freeze backbone initially
    # so the new heads learn first; unfreeze after epoch 10.
    freeze_backbone_initially = (
        config.TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE < 30
    )
    if freeze_backbone_initially:
        model.freeze_backbone()
        print("  (Backbone frozen for first 10 epochs — small dataset mode)")

    #  Loss, Optimiser, Scheduler 
    criterion = CombinedLoss()

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config.INITIAL_LR,
        momentum     = config.MOMENTUM,
        weight_decay = config.WEIGHT_DECAY,
        nesterov     = True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR
    )

    #  Checkpoints 
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    best_ckpt = os.path.join(config.CKPT_DIR, "best_model.pth")
    last_ckpt = os.path.join(config.CKPT_DIR, "last_model.pth")

    #  History 
    history = {"train_loss": [], "val_loss": [],
               "val_miou": [], "val_cpa": [], "val_recall": []}

    best_val_loss = float("inf")
    no_improve    = 0

    print(f"\nTraining for up to {config.EPOCHS} epochs "
          f"(early stop after {config.EARLY_STOP_PAT} epochs without improvement)\n")

    for epoch in range(1, config.EPOCHS + 1):

        # Unfreeze backbone after epoch 10 if it was frozen
        if freeze_backbone_initially and epoch == 11:
            model.unfreeze_backbone()
            # rebuild optimiser to include backbone params
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.INITIAL_LR * 0.1,   # lower LR for fine-tuning
                momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY,
                nesterov=True,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.EPOCHS - epoch, eta_min=config.MIN_LR
            )

        t0 = time.time()

        tr_loss, tr_m   = run_epoch(model, loaders["train"],
                                     optimizer, criterion, device, epoch, "train")
        val_loss, val_m = run_epoch(model, loaders["val"],
                                     None, criterion, device, epoch, "val")

        scheduler.step()
        elapsed = time.time() - t0

        #  Print epoch summary 
        print(
            f"Ep [{epoch:3d}/{config.EPOCHS}]  "
            f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
            f"mIOU={val_m['mIOU']*100:5.1f}%  "
            f"CPA={val_m['CPA']*100:5.1f}%  "
            f"Recall={val_m['Recall']*100:5.1f}%  "
            f"LR={optimizer.param_groups[0]['lr']:.5f}  "
            f"({elapsed:.0f}s)"
        )

        #  Record history 
        history["train_loss"].append(tr_loss)
        history["val_loss"  ].append(val_loss)
        history["val_miou"  ].append(val_m["mIOU"])
        history["val_cpa"   ].append(val_m["CPA"])
        history["val_recall"].append(val_m["Recall"])

        #  Save last checkpoint 
        torch.save({
            "epoch"       : epoch,
            "model_state" : model.state_dict(),
            "optim_state" : optimizer.state_dict(),
            "val_loss"    : val_loss,
            "val_miou"    : val_m["mIOU"],
        }, last_ckpt)

        #  Save best checkpoint 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optim_state" : optimizer.state_dict(),
                "val_loss"    : val_loss,
                "val_miou"    : val_m["mIOU"],
            }, best_ckpt)
            print(f"  ✓ Best model saved  (val_loss={val_loss:.4f})")
        else:
            no_improve += 1

        #  Early stopping 
        if no_improve >= config.EARLY_STOP_PAT:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    #  Save training curves 
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plot_training_curves(
        train_losses = history["train_loss"],
        val_losses   = history["val_loss"],
        val_mious    = history["val_miou"],
        val_cpas     = history["val_cpa"],
        val_recalls  = history["val_recall"],
        save_path    = os.path.join(config.RESULTS_DIR, "training_curves.png"),
    )

    print(f"\nDone.  Best checkpoint → {best_ckpt}")
    return model, history


if __name__ == "__main__":
    train()
