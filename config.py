import os
import torch

DATASET_DIR = r"C:\Users\anany\OneDrive\Documents\dataset"  


#  Directories for saving checkpoints and results 

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR     = os.path.join(BASE_DIR, "checkpoints")   # model weights saved here
RESULTS_DIR  = os.path.join(BASE_DIR, "results")       # visuals + metrics saved here


#  Dataset split

TRAIN_SIZE = 0.8
VAL_SIZE   = 0.1
TEST_SIZE  = 0.1

#  Model settings

NUM_CLASSES = 2          # 0 = background, 1 = void
INPUT_H     = 512        # height fed into the network
INPUT_W     = 512        # width  fed into the network

# MobileNetV2 backbone channel counts

LOW_LEVEL_CHANNELS   = 32    # channels of each bottleneck-group-3 layer
AMTPNET_OUT_CHANNELS = 288   # 9 × 32
HIGH_LEVEL_CHANNELS  = 160   # channels at end of bottleneck-group-6
ASPP_OUT_CHANNELS    = 256
ASPP_ATROUS_RATES    = [6, 12, 18]
LOW_LEVEL_REDUCED    = 48    # paper: reduce AMTPNet output to 48 in decoder


#  Training hyper-parameters  
EPOCHS         = 100
BATCH_SIZE     = 4
INITIAL_LR     = 0.007
MIN_LR         = 0.0001
MOMENTUM       = 0.9
WEIGHT_DECAY   = 0.0001
LR_DECAY_TYPE  = "cosine"
EARLY_STOP_PAT = 3        

# For small datasets / Colab:

NUM_WORKERS    = 0

#  Misc settings

RANDOM_SEED = 42
PRETRAINED  = True   # use ImageNet pretrained MobileNetV2 backbone
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Labelme label name that marks void regions
VOID_LABEL  = "void"
