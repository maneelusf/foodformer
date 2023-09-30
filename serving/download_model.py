import os
from pathlib import Path

import wandb

wandb_team = "maneel"
wandb_project = "Foodformer"
wandb_model = "vit:v0"
wandb_model_path = f"{wandb_team}/{wandb_project}/{wandb_model}"

wandb.init()

current_folder = Path(__file__).parent
print(f"Folder: {current_folder}")
path = wandb.use_artifact("maneel/Foodformer/vit:v0").get_path("model.ckpt").download()
print(f"Model downloaded to: {path}")