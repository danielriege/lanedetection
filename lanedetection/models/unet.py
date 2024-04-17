import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_conv: int) -> None:
        super(ConvBlock, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1) for i in range(n_conv)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(n_conv)])
        self.droputs = nn.ModuleList([nn.Dropout(0.3) for _ in range(n_conv)])
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for bn, conv, dropout in zip(self.bns, self.convs, self.droputs):
            x = conv(x)
            x = F.relu(bn(x))
            x = dropout(x)
        return self.maxpool(x), x # skip conn
    
class ConvBlockTranspose(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, fct: int = 2) -> None:
        super(ConvBlockTranspose, self).__init__()
        self.conv1 = nn.Conv2d(in_channels*fct, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor, skip_conn: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout2(x)
        if skip_conn is not None:
            x = torch.cat((x, skip_conn), dim=1)
        return x

class VGGU(nn.Module):
    def __init__(self, n_classes=7, n: int=16) -> None:
        super(VGGU, self).__init__()
        self.n_classes = n_classes
        self.n = n
        self.blocks = {
            16: [(64, 2), (128, 2), (256, 3), (512, 3), (512,3)],
            8: [(32, 2), (48, 2), (64, 3), (128, 3), (256,3)]
            }
        self.upblocks = list(reversed(self.blocks[n]))
        self.conv_blocks = nn.ModuleList([ConvBlock(3 if i == 0 else self.blocks[n][i-1][0], out_channels, n_conv) for i, (out_channels, n_conv) in enumerate(self.blocks[n])])
        self.upconv_blocks = nn.ModuleList([ConvBlockTranspose(self.blocks[n][-1][0] if i == 0 else self.upblocks[i-1][0], out_channels, 1 if i == 0 else 2) for i, (out_channels, _) in enumerate(self.upblocks)])
        last_n_channels = self.blocks[n][0][0]
        self.last_conv1 = nn.Conv2d(last_n_channels*2, last_n_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(last_n_channels)
        self.dropout1 = nn.Dropout(0.3)
        self.last_conv2 = nn.Conv2d(last_n_channels, n_classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        skips = [None] # for last layer
        for block in self.conv_blocks:
            x, skip_conn = block(x)
            skips.append(skip_conn)
        # decoder
        for block in self.upconv_blocks:
            x = block(x, skips.pop())
        x = self.last_conv1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout1(x)
        return F.softmax(self.last_conv2(x), dim=1)
    
    def load_pretrained(self, device) -> bool:
        """
        Loads the pretrained weights for the model.
        Provide torch device for right mapping location.
        Returns True if pretrained weights are found, False otherwise.
        """
        if (self.n, self.n_classes) in model_urls:
            model_url = model_urls[(self.n, self.n_classes)]
            cached_file = os.path.join("/tmp", os.path.basename(model_url))
            if not os.path.exists(cached_file):
                torch.hub.download_url_to_file(model_url, cached_file)
            self.load_state_dict(torch.load(cached_file, map_location=device))
            return True
        print(f"No pretrained weights found for model with n={self.n} and n_classes={self.n_classes}.")
        return False
    
model_urls: Dict[Tuple[int,int], str] = {   
    (8,7): "http://riege.com.de/lanedetection/vgg8u_7c.pt",
}
    
VGG16U = lambda n_classes=7: VGGU(n_classes=n_classes, n=16)
VGG8U = lambda n_classes=7: VGGU(n_classes=n_classes, n=8)

if __name__ == "__main__":
    m = VGG16U()
    x = torch.randn(1, 3, 64, 160)
    y = m(x)
    assert y.shape == (1, 7, 64, 160)
