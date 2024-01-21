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




class PneumoniaModel(p1.LightningModule):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        
        
    def forward(self, data):
        pred = self.model(data)
        return pred
    
    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        preds = self(x_ray)[:, 0]
        loss = self.loss_fn(preds, label)
        
        self.log("TRAIN LOSS", loss)
        self.log("STEP TRAIN ACC", self.train_acc(torch.sigmoid(preds), label.int()))
    
    
    
    
    def on_training_epoch_end(self):
        self.log("TRAIN ACC", self.train_acc.compute())
        
        
        
    
    
    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        preds = self(x_ray)[:, 0]
        loss = self.loss_fn(preds, label)
        self.log("VAL LOSS", loss)
        
        self.log("STEP VAL ACC", self.val_acc(torch.sigmoid(preds), label.int()))
        

    
    
    
    def on_validation_epoch_end(self):
        self.log('VAL ACC', self.val_acc.compute())
        
        
    def configure_optimizers(self):
        return [self.optimizer]
    
    
def main():
    model = PneumoniaModel()
    checkpoint_callback = ModelCheckpoint(monitor='VAL ACC', save_top_k=10, mode='max')
    trainer = p1.Trainer(logger=TensorBoardLogger(save_dir="./logs"), log_every_n_steps=1, callbacks=checkpoint_callback, max_epochs=10)
    trainer.fit(model, train_loader, val_loader)


    
if __name__ == "__main__":
    freeze_support()
    main()



#%%




    
    
    
        
        
            
# %%
