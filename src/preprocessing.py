

#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pydicom
import cv2


#%%
labels = pd.read_csv(r'C:/Users/Asus/OneDrive/Desktop/Pneumonia classification/src/input/stage_2_train_labels.csv')



#%%
labels = labels.drop_duplicates('patientId')
#%%
ROOT_PATH = Path("C:/Users/Asus/Downloads/stage_2_train_images/")
SAVE_PATH = Path("C:/Users/Asus/OneDrive/Desktop/Pneumonia classification/src/input/Processed")


#%%
fig, axis = plt.subplots(3,3,figsize = (9,9))

c = 0

for i in range(3):
    for j in range(3):
        patient_id = labels.patientId.iloc[c]
        dcm_path = ROOT_PATH/patient_id
        dcm_path = dcm_path.with_suffix(".dcm")
        dcm = pydicom.read_file(dcm_path).pixel_array
        
        label = labels["Target"].iloc[c]
        
        axis[i][j].imshow(dcm, cmap = 'bone')
        axis[i][j].set_title(label)
        c += 1
                
        
        
#%%

sums, sums_squared = 0, 0

for c, patient_id in enumerate(tqdm(labels.patientId)):
    patient_id = labels.patientId.iloc[c]
    dcm_path = ROOT_PATH/patient_id
    dcm_path = dcm_path.with_suffix(".dcm")
    dcm = pydicom.read_file(dcm_path).pixel_array / 255
    dcm_array = cv2.resize(dcm, (224,224)).astype(np.float16)
    
    label = labels.Target.iloc[c]
    
    train_or_val = "train" if c < 24000 else "val"
    
    current_save_path = SAVE_PATH/train_or_val/str(label)
    
    current_save_path.mkdir(parents=True, exist_ok=True)
    
    np.save(current_save_path/patient_id, dcm_array)
    
    normalizer = 224 * 224
    
    if train_or_val == "train":
        sums += np.sum(dcm_array) / normalizer
        sums_squared += (dcm_array ** 2).sum() / normalizer
        
        
    
    
    


# %%
