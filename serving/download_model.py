import wandb
import os
wandb.init()
path = wandb.use_artifact("maneel/Foodformer/vit:v0").download()
### Moving to serving directory
# os.system("mv artifacts serving")