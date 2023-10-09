import random
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn.functional import cross_entropy, softmax
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from torchvision.datasets import Food101
from transformers import ViTForImageClassification, ViTImageProcessor

import wandb

model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)

preprocessor = partial(feature_extractor, return_tensors="pt")

train_ds = Food101(
    root="food101_dataset", split="train", transform=preprocessor, download=True
)
test_ds = Food101(
    root="food101_dataset",
    split="test",
    transform=preprocessor,
)

labels = train_ds.classes
print(f"Classes examples: {labels[:10]}")


model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)

preprocessor = partial(feature_extractor, return_tensors="pt")

train_ds = Food101(
    root="food101_dataset", split="train", transform=preprocessor, download=True
)
test_ds = Food101(
    root="food101_dataset",
    split="test",
    transform=preprocessor,
)

labels = train_ds.classes
print(f"Classes examples: {labels[:10]}")


class ViTDataset(Dataset):
    """Package images pixel values and labels into a dictionary."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        """In PyTorch datasets have to override the length method."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        """This method defines how to feed the data during model training."""
        inputs_and_labels = dict()
        inputs_and_labels["pixel_values"] = self.dataset[index][0]["pixel_values"][0]
        inputs_and_labels["labels"] = self.dataset[index][1]
        return inputs_and_labels


train_dataset = ViTDataset(train_ds)
test_dataset = ViTDataset(test_ds)

# Optional: work with a subset of the data for development
train_dataset = torch.utils.data.Subset(
    train_dataset, random.sample(range(len(train_dataset)), 10000)
)
test_dataset = torch.utils.data.Subset(
    test_dataset, random.sample(range(len(train_dataset)), 1000)
)

# Create a dataloader object to chunk the datasets into batches
training_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
testing_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
logger = WandbLogger(
    project="Foodformer",
    name="VisionTransformer-base",
    checkpoint_name="vit",
    save_dir=".",
    log_model=True,
)

names = ["train", "test"]
datasets = [train_dataset, test_dataset]

# 🏺 create our Artifact
raw_data = wandb.Artifact(
    "food101-custom",
    type="dataset",
    description="Custom Food101 dataset, split into train/test",
    metadata={"sizes": [len(dataset) for dataset in datasets]},
)

for name, data in zip(names, datasets):
    with raw_data.new_file(name + ".pt", mode="wb") as fs:
        torch.save(data, fs)


class LightningVisionTransformer(pl.LightningModule):
    """Main model object: contains the model, defines how
    to run a forward pass, what the loss is, and the optimizer."""

    def __init__(self, model, learning_rate=2e-4, label_names=labels):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.label_names = label_names
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        pixel_values = batch["pixel_values"]
        logits = self.model(pixel_values).logits
        return logits

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.forward(batch)
        loss = cross_entropy(outputs, labels)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.forward(batch)
        loss = cross_entropy(outputs, labels)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        preds = outputs.argmax(dim=1)
        acc = accuracy(
            preds, labels, task="multiclass", num_classes=len(self.label_names)
        )
        self.log(
            "accuracy",
            acc,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, pixel_values):
        logits = self.model(pixel_values).logits
        probas = softmax(logits, dim=1)
        values, indices = torch.topk(probas[0], 5)
        return_dict = {
            self.label_names[int(i)]: float(v) for i, v in zip(indices, values)
        }
        return return_dict


# Download pre-trained model from the HuggingFace model hub
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)

# Create the Pytorch Lightning model
lightning_vit = LightningVisionTransformer(model)

# define pytorch lightning checkpoint callback
checkpoint_callback = ModelCheckpoint(every_n_epochs=1)
trainer = pl.Trainer(
    # max_steps=100, # For debug, comment for real training
    default_root_dir="content",
    max_epochs=10,
    logger=logger,
    log_every_n_steps=23,
)

# # Run evaluation before training
# trainer.validate(lightning_vit)

# Train the model
trainer.fit(lightning_vit, training_loader, testing_loader)

trainer.save_checkpoint("model.ckpt")
