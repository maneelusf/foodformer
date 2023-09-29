import wandb
wandb.init()
path = wandb.use_artifact("maneel/Foodformer/vit:v0").download()
print(path)