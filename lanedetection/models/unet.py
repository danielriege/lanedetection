from tinygrad import Tensor, nn
from typing import Any, List, Tuple, Optional

class ConvBlock:
    def __init__(self, in_channels: int, out_channels: int, n_conv: int) -> None:
        self.convs = [nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1) for i in range(n_conv)]
        self.bns = [nn.BatchNorm2d(out_channels) for _ in range(n_conv)]
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        for bn, conv in zip(self.bns, self.convs):
            x = conv(x)
            x = bn(x).relu().dropout(0.3)
        return x.max_pool2d(), x # skip conn
    
    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward(x)
    
class ConvBlockTranspose:
    def __init__(self, in_channels: int, out_channels: int, fct: int = 2) -> None:
        self.conv1 = nn.Conv2d(in_channels*fct, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, output_padding=-1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: Tensor, skip_conn: Optional[Tensor]) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x).relu().dropout(0.3)
        x = self.conv2(x)
        x = self.bn2(x).relu().dropout(0.3)
        if skip_conn is not None:
            x = x.cat(skip_conn, dim=1)
        return x
    
    def __call__(self, x: Tensor, skip_conn: Optional[Tensor]) -> Tensor:
        return self.forward(x, skip_conn)

class VGGU:
    def __init__(self, n_classes=7, n: int=16) -> None:
        self.blocks = {
            16: [(64, 2), (128, 2), (256, 3), (512, 3), (512,3)],
            8: [(24, 2), (32, 2), (48, 2), (128, 2), (256,2)]
            }
        self.upblocks = list(reversed(self.blocks[n]))
        self.conv_blocks = [ConvBlock(3 if i == 0 else self.blocks[n][i-1][0], out_channels, n_conv) for i, (out_channels, n_conv) in enumerate(self.blocks[n])]
        self.upconv_blocks = [ConvBlockTranspose(self.blocks[n][-1][0] if i == 0 else self.upblocks[i-1][0], out_channels, 1 if i == 0 else 2) for i, (out_channels, _) in enumerate(self.upblocks)]
        last_n_channels = self.blocks[n][0][0]
        self.last_conv1 = nn.Conv2d(last_n_channels*2, last_n_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(last_n_channels)
        self.last_conv2 = nn.Conv2d(last_n_channels, n_classes, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # encoder
        skips = [None] # for last layer
        for block in self.conv_blocks:
            x, skip_conn = block(x)
            skips.append(skip_conn)
        # decoder
        for block in self.upconv_blocks:
            x = block(x, skips.pop())
        x = self.last_conv1(x)
        x = self.bn1(x).relu().dropout(0.3)
        return self.last_conv2(x).sigmoid()
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
VGG16U = lambda n_classes=7: VGGU(n_classes=n_classes, n=16)
VGG8U = lambda n_classes=7: VGGU(n_classes=n_classes, n=8)

if __name__ == "__main__":
    m = VGG16U()
    x = Tensor.randn(1, 3, 64, 160)
    y = m(x)
    assert y.shape == (1, 7, 64, 160)
        
