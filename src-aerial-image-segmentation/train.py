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

optimizer = torch.optim.Adam(model.parameters(), lr = LR)

best_test_loss = np.Inf

for i in range(EPOCHS):
    train_loss = train_fn(model=model,
                          data_loader = train_loader,
                          optimizer=optimizer)
    test_loss = eval_fn(data_loader=test_loader,
                        model = model)
    
    if test_loss < best_test_loss:
        torch.save(model.state_dict(), './model/aerial_segmentation_best_model.pth')
        print("SAVED MODEL")

        best_test_loss = test_loss

    print(f"Epoch :{i + 1} | train_loss: {train_loss} | test_loss: {test_loss} ")



