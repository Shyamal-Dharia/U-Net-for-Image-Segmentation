import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import albumentations as A
from tqdm.auto import tqdm
import glob

CSV_FILE = './Human-Segmentation-Dataset-master/train.csv'
DATA_DIR = './Human-Segmentation-Dataset-master/'

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
EPOCHS = 35
LR = 0.003
IMAGE_SIZE = (448,448)
BATCH_SIZE = 16

ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

TEST_PATH = glob.glob('./Human-Segmentation-Dataset-master/test_images/*.jpg')


def get_train_augs():
    """
    Defines the augmentation pipeline for training images.

    Returns:
    A.Compose: Augmentation pipeline object configured for training data.
                - Resizes images to the specified dimensions.
                - Applies horizontal flipping with a probability of 0.5.
                - Applies vertical flipping with a probability of 0.5.
                - Ignores shape checking for additional targets ('mask').

    """
    return A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], additional_targets={'mask': 'image'}, is_check_shapes=False)

def get_test_augs():
    """
    Defines the augmentation pipeline for test/validation images.

    Returns:
    A.Compose: Augmentation pipeline object configured for test/validation data.
                - Resizes images to the specified dimensions.

    """
    return A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    ], additional_targets={'mask': 'image'}, is_check_shapes=False)


def get_test_img_augs():
    """
    Defines the augmentation pipeline for test/validation images.

    Returns:
    A.Compose: Augmentation pipeline object configured for test/validation images.
                - Resizes images to the specified dimensions.
                - Ignores shape checking for additional targets.

    """
    return A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    ], is_check_shapes=False)



class SegmentationDataset(Dataset):
    """
    Dataset class for semantic segmentation tasks.

    Args:
    df (DataFrame): DataFrame containing image and mask file paths.
    augmentations (callable): Augmentation pipeline function.

    """

    def __init__(self, df, augmentations=None):
        self.df = df
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        """
        Fetches and preprocesses the image and mask for a given index.

        Args:
        index (int): Index of the sample to fetch.

        Returns:
        tuple: A tuple containing the preprocessed image and mask.

        """
        row = self.df.iloc[index]
        image_path = row.images
        mask_path = row.masks

        # Load image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        # Apply augmentations
        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        # Transpose and normalize
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image) / 255.0
        mask = torch.round(torch.tensor(mask) / 255.0)

        return image, mask
    

def train_fn(data_loader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             optimizer:torch.optim):
    """
    Function to train the segmentation model.

    Args:
    data_loader (DataLoader): DataLoader object for training data.
    model (nn.Module): Segmentation model.
    optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.

    Returns:
    float: Average loss over the training dataset.

    """
    model.train()

    total_loss = 0

    for images, masks in tqdm(data_loader):
        
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(data_loader,
            model):
    """
    Function to evaluate the segmentation model.

    Args:
    data_loader (DataLoader): DataLoader object for evaluation data.
    model (nn.Module): Segmentation model.

    Returns:
    float: Average loss over the evaluation dataset.

    """
    model.eval()
    total_loss = 0
    with torch.inference_mode():
        for images, masks in tqdm(data_loader):
            
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            logits, loss = model(images, masks)

            total_loss += loss.item()

    return total_loss / len(data_loader)

def get_test_img(image_path: str = "./Human-Segmentation-Dataset-master/test_images/",
                 augmentations=get_test_img_augs()):
    """
    Loads and preprocesses a test/validation image.

    Args:
    image_path (str, optional): Path to the test/validation image.
    augmentations (callable, optional): Augmentation pipeline function.

    Returns:
    torch.Tensor: Preprocessed image tensor.

    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply augmentations
    if augmentations:
        data = augmentations(image=image)
        image = data['image']

    # Transpose and normalize
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image) / 255.0

    return image
