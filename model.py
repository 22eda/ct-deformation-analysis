import torch
import torch.nn as nn
from monai.networks.nets import UNet


class Probabilistic3DUNet(nn.Module):

  def __init__(
      self,
      in_channels=1,
      out_channels=3,
      dropout_prob=0.2,
  ):
    super(Probabilistic3DUNet, self).__init__()

    # MONAI-based 3D U-Net 
    # Input: nth fraction CT [B, 1, D, H, W] | Output: Deformation Field [B, 3, D, H, W]
    self.unet = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=dropout_prob,
    )

  def forward(self, x):
    # Outputs the 3-channel displacement field (X, Y, Z vector components)
    return self.unet(x)

  def enable_inference_dropout(self):
    """Crucial requirement by Prof. Tabor: 

    Forces dropout layers to remain active during the inference (evaluation) phase 
    to generate anatomical variation distributions.
    """
    for m in self.modules():
      if isinstance(m, nn.Dropout3d) or isinstance(m, nn.Dropout):
        m.train()
