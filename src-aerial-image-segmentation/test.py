import torch
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import  DataLoader
import cv2
import numpy as np
import torch
from utils import *
from model import SegmentationModel
import matplotlib.pyplot as plt

df = pd.read_csv(CSV_FILE)

train_df, test_df = train_test_split(df, test_size = 0.2)

train_set = SegmentationDataset(train_df, get_train_augs())
test_set = SegmentationDataset(test_df, get_test_augs())

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = SegmentationModel().to(DEVICE)
model.load_state_dict(torch.load('./model/aerial_segmentation_best_model.pth'))


images, masks = next(iter(test_loader))
images, masks = images.to(DEVICE), masks.to(DEVICE)
for i in range(len(images)):

    logit_mask = model(images[i].unsqueeze(0))
    pred_mask = torch.sigmoid(logit_mask)
    pred_mask = (pred_mask > 0.5) * 1.0
    pred_mask = pred_mask.cpu()
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5))
        
    ax1.set_title('ORIGINAL IMAGE')
    ax1.imshow(images[i].cpu().permute(1,2,0).squeeze(),cmap = 'gray')

    ax2.set_title('GROUND TRUTH')
    ax2.imshow(masks[i].cpu().permute(1,2,0).squeeze(),cmap = 'gray')

    ax3.set_title('MODEL OUTPUT')
    ax3.imshow(pred_mask.permute(0,2,3,1).squeeze(0),cmap = 'gray')

    plt.savefig(f'result_{i+1}.png')
    # Close the plot to release resources
    plt.close()