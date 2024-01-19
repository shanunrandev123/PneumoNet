
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

def load_file(path):
    return np.load(path).astype(np.float32)


train_transforms = transforms.Compose([
    
    transforms.ToTensor(),
    transforms.Normalize(0.49, 0.248),
    transforms.RandomAffine(degrees=(-5,5), translate=(0,0.05), scale=(0.9,1.1)),
    transforms.RandomResizedCrop((224,224), scale = (0.35,1))
    
    
])

val_transforms = transforms.Compose([
    
    transforms.ToTensor(),
    transforms.Normalize(0.49, 0.248)
])




train_dataset = torchvision.datasets.DatasetFolder("Processed/train/", loader=load_file, extensions="npy", transform=train_transforms)

val_dataset = torchvision.datasets.DatasetFolder("Processed/val/", loader=load_file, extensions="npy", transform=val_transforms)


batch_size = 32

num_workers = 2

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle = True)

val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle = False)


#class imbalance problem

print(np.unique(train_dataset.targets, return_counts=True))
