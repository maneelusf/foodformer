import wandb
from functools import partial
from io import BytesIO
from pathlib import Path
import wandb
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch.nn.functional import softmax
from transformers import ViTImageProcessor
wandb.init()
path = wandb.use_artifact("maneel/Foodformer/vit:v0").download()
print(path)