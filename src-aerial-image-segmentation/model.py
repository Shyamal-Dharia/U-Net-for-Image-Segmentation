from utils import *
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class SegmentationModel(nn.Module):
    """
    Semantic Segmentation Model based on the U-Net architecture.

    Attributes:
    arc (nn.Module): U-Net model.
    
    """

    def __init__(self):
        """
        Initializes the SegmentationModel class.

        """
        super(SegmentationModel, self).__init__()

        self.arc = smp.Unet(encoder_name=ENCODER,
                            encoder_weights=WEIGHTS,
                            in_channels=3,
                            classes=1,
                            activation=None)
        
    def forward(self, images, masks=None):
        """
        Forward pass of the SegmentationModel.

        Args:
        images (torch.Tensor): Input images.
        masks (torch.Tensor, optional): Ground truth masks.

        Returns:
        torch.Tensor or tuple: If masks are provided, returns logits and combined loss;
                               otherwise, returns logits only.

        """
        logits = self.arc(images)

        if masks is not None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
        
            return logits, loss1 + loss2
    
        return logits
    

