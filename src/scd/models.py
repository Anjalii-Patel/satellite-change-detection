import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class EarlyFusionUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(UNetBlock(in_channels, feature))
            in_channels = feature
            
        # Bottleneck
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(UNetBlock(feature * 2, feature))
            
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, t1, t2):
        # Early fusion: concatenate along the channel dimension
        x = torch.cat([t1, t2], dim=1)
        
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            
            # Handle potential padding issues
            if x.shape != skip_connection.shape:
                import torch.nn.functional as F
                x = F.interpolate(x, size=skip_connection.shape[2:])
                
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](x)
            
        return self.final_conv(x)
