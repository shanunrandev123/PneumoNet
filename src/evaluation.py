from model import main, PneumoniaModel
import preprocessing
from Train import train_loader, train_dataset, val_dataset, val_loader

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchmetrics
import pytorch_lightning as p1
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.multiprocessing import freeze_support


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PneumoniaModel()

model.eval()

model.to(device)

preds = []

labels = []

with torch.no_grad():
    for data, label in tqdm(val_dataset):
        data = data.to(device).float().unsqueeze(0)
        pred = torch.sigmoid(model(data)[0].cpu())
        preds.append(pred)
        labels.append(label)
        
preds = torch.tensor(preds)
labels = torch.tensor(labels).int()

acc = torchmetrics.Accuracy(task='binary')(preds, labels)
precision = torchmetrics.Precision(task='binaryclass')(preds, labels)
recall = torchmetrics.Recall(task='binary')(preds, labels)
# cm = torchmetrics.ConfusionMatrix(num_classes=2, )(preds, labels)

print(f"val accuracy {acc}")
# print(f"precision score {precision}")
# print(f"recall score {recall}")
# print(f"confusion matrix {cm}")

