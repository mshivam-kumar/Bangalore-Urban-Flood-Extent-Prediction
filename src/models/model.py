# class UNetWithRainfallFusion(nn.Module):
#     def __init__(self, img_channels=10, rainfall_dim=1, rainfall_feats=4):
#         """
#         img_channels: number of image input channels (NOT counting separate rainfall tensor).
#                       If you pass rainfall as part of features, adjust accordingly.
#         rainfall_dim: expected flattened rainfall vector size after pooling (1 if single-channel)
#         rainfall_feats: number of channels produced from rainfall MLP to concat at bottleneck
#         """
#         super().__init__()
#         # Encoder
#         self.enc1 = nn.Sequential(nn.Conv2d(img_channels, 32, 3, padding=1), nn.ReLU(),
#                                   nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
#         self.pool1 = nn.MaxPool2d(2)  # 256 -> 128

#         self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
#                                   nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
#         self.pool2 = nn.MaxPool2d(2)  # 128 -> 64

#         self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
#                                   nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
#         self.pool3 = nn.MaxPool2d(2)  # 64 -> 32

#         # Rainfall MLP (works with rainfall flattened size = rainfall_dim)
#         # We will adaptive-pool rainfall to (1,1) if it's a map, producing rainfall_dim=1
#         self.r_fc1 = nn.Linear(rainfall_dim, 256)
#         self.r_fc2 = nn.Linear(256, rainfall_feats * 32 * 32)  # reshape later to (B, rainfall_feats, 32,32)
#         self.rainfall_feats = rainfall_feats

#         # Decoder (upsample back to 256)
#         # After concatenation channels = 128 + rainfall_feats
#         self.up1 = nn.ConvTranspose2d(128 + rainfall_feats, 128, kernel_size=2, stride=2)  # 32 -> 64
#         self.dec1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())

#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 64 -> 128
#         self.dec2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

#         self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   # 128 -> 256
#         self.dec3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())

#         self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

#     def forward(self, x, rainfall):
#         """
#         x: (B, img_channels, 256, 256)
#         rainfall: either (B,1,256,256) or (B,1) or (B,) or (B, k) depending on your design.
#         """
#         B = x.shape[0]

#         # Encoder
#         x1 = self.enc1(x)   # (B,32,256,256)
#         p1 = self.pool1(x1) # (B,32,128,128)

#         x2 = self.enc2(p1)  # (B,64,128,128)
#         p2 = self.pool2(x2) # (B,64,64,64)

#         x3 = self.enc3(p2)  # (B,128,64,64)
#         p3 = self.pool3(x3) # (B,128,32,32)  <-- bottleneck

#         # Process rainfall:
#         # If rainfall is a map -> adaptively average pool to (B, channels, 1, 1)
#         if rainfall.dim() == 4:
#             # typical shape (B,1,256,256) -> pool to (B,1,1,1) then flatten
#             r = F.adaptive_avg_pool2d(rainfall, (1,1)).view(B, -1)  # (B, channels) -> usually (B,1)
#         elif rainfall.dim() == 2 or rainfall.dim() == 1:
#             r = rainfall.view(B, -1)  # already flattened vector
#         else:
#             raise ValueError("Unsupported rainfall tensor shape: " + str(rainfall.shape))

#         # MLP
#         r = F.relu(self.r_fc1(r))  # (B, 256)
#         r = F.relu(self.r_fc2(r))  # (B, rainfall_feats * 32 * 32)
#         r = r.view(B, self.rainfall_feats, 32, 32)  # (B, rainfall_feats, 32,32)

#         # Concatenate at bottleneck
#         bottleneck = torch.cat([p3, r], dim=1)  # (B, 128 + rainfall_feats, 32,32)

#         # Decoder: upsample back to original resolution
#         d1 = self.up1(bottleneck)  # (B,128,64,64)
#         d1 = self.dec1(d1)

#         d2 = self.up2(d1)  # (B,64,128,128)
#         d2 = self.dec2(d2)

#         d3 = self.up3(d2)  # (B,32,256,256)
#         d3 = self.dec3(d3)

#         out = self.out_conv(d3)  # (B,1,256,256)
#         return out


import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Helpers
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)     # conv features for skip
        x_pooled = self.maxpool(x)  # downsample for next stage
        return x, x_pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat with skip, we get out_channels + skip_channels
        # We need to know the skip channels to set this correctly
        self.double_conv = DoubleConv(out_channels * 2, out_channels)  # assuming skip has same channels

    def forward(self, x, skip):
        x = self.upconv(x)
        # Handle odd sizes (safety)
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class CustomDecoderBlock(nn.Module):
    """Custom decoder block that handles arbitrary input/skip channel combinations"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(CustomDecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat: out_channels (from upconv) + skip_channels
        self.double_conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        # Handle odd sizes (safety)
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


# -----------------------------
# Binary Model with Mid-Fusion
# -----------------------------
class BinaryModel(nn.Module):
    def __init__(self, in_channels=11, out_channels=1, rainfall_dim=256*256):
        super(BinaryModel, self).__init__()

        # --- Encoder ---
        self.enc1 = EncoderBlock(in_channels, 32)   # Output: (B, 32, 256, 256) -> pooled (B, 32, 128, 128)
        self.enc2 = EncoderBlock(32, 64)            # Output: (B, 64, 128, 128) -> pooled (B, 64, 64, 64)
        self.enc3 = EncoderBlock(64, 128)           # Output: (B, 128, 64, 64) -> pooled (B, 128, 32, 32)

        # --- Rainfall MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(rainfall_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 16*32*32),  # reshape into (16, 32, 32)
            nn.ReLU()
        )

        # --- Decoder with correct channel calculations ---
        # dec1: input (144, 32, 32), skip (128, 64, 64) -> output (64, 64, 64)
        self.dec1 = CustomDecoderBlock(144, 128, 64)  # 144->64 upconv, then 64+128=192 -> 64
        
        # dec2: input (64, 64, 64), skip (64, 128, 128) -> output (32, 128, 128)  
        self.dec2 = CustomDecoderBlock(64, 64, 32)    # 64->32 upconv, then 32+64=96 -> 32
        
        # dec3: input (32, 128, 128), skip (32, 256, 256) -> output (16, 256, 256)
        self.dec3 = CustomDecoderBlock(32, 32, 16)    # 32->16 upconv, then 16+32=48 -> 16

        # Final output
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x_static, x_rainfall):
        # x_static: (B, C, 256, 256)
        # x_rainfall: (B, 1, 256, 256)  <- full raster patch

        # --- Encoder ---
        x1, x1_pooled = self.enc1(x_static)   # x1: (B, 32, 256, 256), x1_pooled: (B, 32, 128, 128)
        x2, x2_pooled = self.enc2(x1_pooled)  # x2: (B, 64, 128, 128), x2_pooled: (B, 64, 64, 64)
        x3, x3_pooled = self.enc3(x2_pooled)  # x3: (B, 128, 64, 64), x3_pooled: (B, 128, 32, 32)

        # --- Rainfall processing ---
        B = x_rainfall.size(0)
        x_rainfall = x_rainfall.view(B, -1)     # flatten to (B, 256*256)
        x_rainfall = self.mlp(x_rainfall)       # (B, 16*32*32)
        x_rainfall = x_rainfall.view(B, 16, 32, 32)  # (B, 16, 32, 32)

        # --- Fusion at deepest bottleneck (32x32) ---
        x_fused = torch.cat([x3_pooled, x_rainfall], dim=1)  # (B, 128+16, 32, 32) = (B, 144, 32, 32)

        # --- Decoder with proper skip connections ---
        x5 = self.dec1(x_fused, x3)    # input: (B, 144, 32, 32), skip: (B, 128, 64, 64) -> (B, 64, 64, 64)
        x6 = self.dec2(x5, x2)         # input: (B, 64, 64, 64), skip: (B, 64, 128, 128) -> (B, 32, 128, 128)
        x7 = self.dec3(x6, x1)         # input: (B, 32, 128, 128), skip: (B, 32, 256, 256) -> (B, 16, 256, 256)

        return self.final_conv(x7)      # (B, 1, 256, 256)
    



class BinaryModelDeeper(nn.Module):
    def __init__(self, in_channels=11, out_channels=1, rainfall_dim=256*256):
        super(BinaryModelDeeper, self).__init__()

        # --- Encoder ---
        self.enc1 = EncoderBlock(in_channels, 32)   # Output: (B, 32, 256, 256) -> pooled (B, 32, 128, 128)
        self.enc2 = EncoderBlock(32, 64)            # Output: (B, 64, 128, 128) -> pooled (B, 64, 64, 64)
        self.enc3 = EncoderBlock(64, 128)           # Output: (B, 128, 64, 64) -> pooled (B, 128, 32, 32)

        # --- Rainfall MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(rainfall_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 16*32*32),  # reshape into (16, 32, 32)
            nn.ReLU()
        )

        # --- Decoder with correct channel calculations ---
        # dec1: input (144, 32, 32), skip (128, 64, 64) -> output (64, 64, 64)
        self.dec1 = CustomDecoderBlock(144, 128, 64)  # 144->64 upconv, then 64+128=192 -> 64
        
        # dec2: input (64, 64, 64), skip (64, 128, 128) -> output (32, 128, 128)  
        self.dec2 = CustomDecoderBlock(64, 64, 32)    # 64->32 upconv, then 32+64=96 -> 32
        
        # dec3: input (32, 128, 128), skip (32, 256, 256) -> output (16, 256, 256)
        self.dec3 = CustomDecoderBlock(32, 32, 16)    # 32->16 upconv, then 16+32=48 -> 16

        # Final output
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x_static, x_rainfall):
        # x_static: (B, C, 256, 256)
        # x_rainfall: (B, 1, 256, 256)  <- full raster patch

        # --- Encoder ---
        x1, x1_pooled = self.enc1(x_static)   # x1: (B, 32, 256, 256), x1_pooled: (B, 32, 128, 128)
        x2, x2_pooled = self.enc2(x1_pooled)  # x2: (B, 64, 128, 128), x2_pooled: (B, 64, 64, 64)
        x3, x3_pooled = self.enc3(x2_pooled)  # x3: (B, 128, 64, 64), x3_pooled: (B, 128, 32, 32)

        # --- Rainfall processing ---
        B = x_rainfall.size(0)
        x_rainfall = x_rainfall.view(B, -1)     # flatten to (B, 256*256)
        x_rainfall = self.mlp(x_rainfall)       # (B, 16*32*32)
        x_rainfall = x_rainfall.view(B, 16, 32, 32)  # (B, 16, 32, 32)

        # --- Fusion at deepest bottleneck (32x32) ---
        x_fused = torch.cat([x3_pooled, x_rainfall], dim=1)  # (B, 128+16, 32, 32) = (B, 144, 32, 32)

        # --- Decoder with proper skip connections ---
        x5 = self.dec1(x_fused, x3)    # input: (B, 144, 32, 32), skip: (B, 128, 64, 64) -> (B, 64, 64, 64)
        x6 = self.dec2(x5, x2)         # input: (B, 64, 64, 64), skip: (B, 64, 128, 128) -> (B, 32, 128, 128)
        x7 = self.dec3(x6, x1)         # input: (B, 32, 128, 128), skip: (B, 32, 256, 256) -> (B, 16, 256, 256)

        return self.final_conv(x7)      # (B, 1, 256, 256)