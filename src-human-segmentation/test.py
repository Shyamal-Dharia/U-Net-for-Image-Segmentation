from model import SegmentationModel
from utils import *
import pandas as  pd 
import matplotlib.pyplot as plt

model = SegmentationModel().to(DEVICE)
model.load_state_dict(torch.load('./model/best_model.pth'))

for i in range(len(TEST_PATH)):
    image = get_test_img(image_path=TEST_PATH[i])
    logit_mask = model(image.to(DEVICE).unsqueeze(0))
    pred_mask = torch.sigmoid(logit_mask)
    pred_mask = (pred_mask > 0.9) * 1.0
    pred_mask = pred_mask.cpu()
    f , (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(image.permute(1,2,0))
    ax1.set_title("ORIGINAL Image")
    ax2.imshow(pred_mask[0].permute(1,2,0),cmap="gray")
    ax2.set_title("MODEL MASK")
    
    # Save the plot as an image file
    plt.savefig(f'result_{i+1}.png')
    # Close the plot to release resources
    plt.close()

# Optionally, you can print a message indicating that the process is complete
print("Plots saved as image files.")
