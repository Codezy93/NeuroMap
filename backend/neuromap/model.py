from monai.networks.layers import Norm
from monai.networks.nets import UNet


def build_unet(
    in_channels: int,
    out_channels: int,
    dropout: float = 0.2,
) -> UNet:
    """Build a 3D U-Net with dropout enabled for MC-dropout inference."""
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=dropout,
    )
