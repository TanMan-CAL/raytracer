import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Denoiser:
    def __init__(self, model_path=None, device=None):
        """
        Initialize the denoiser with a U-Net model.
        
        Args:
            model_path: Path to a pretrained model, if available
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = UNet(n_channels=3, n_classes=3, bilinear=True)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
    def denoise_image(self, noisy_image_path, output_path=None):
        """
        Denoise an image using the U-Net model.
        
        Args:
            noisy_image_path: Path to the noisy image
            output_path: Path to save the denoised image (optional)
            
        Returns:
            Denoised image as a numpy array
        """
        # Load and preprocess the image
        img = Image.open(noisy_image_path).convert('RGB')
        img_np = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Denoise the image
        with torch.no_grad():
            output = self.model(img_tensor)
            
        # Convert back to numpy array
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output, 0, 1)  # Ensure values are in [0, 1]
        
        # Save the denoised image if output_path is provided
        if output_path:
            denoised_img = Image.fromarray((output * 255).astype(np.uint8))
            denoised_img.save(output_path)
            
        return output
    
    def denoise_array(self, noisy_array):
        """
        Denoise a numpy array directly.
        
        Args:
            noisy_array: Numpy array of shape (H, W, 3) with values in [0, 1]
            
        Returns:
            Denoised array with the same shape
        """
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(noisy_array).permute(2, 0, 1).float().unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Denoise the image
        with torch.no_grad():
            output = self.model(img_tensor)
            
        # Convert back to numpy array
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output, 0, 1)  # Ensure values are in [0, 1]
            
        return output
    
    def train(self, dataloader, epochs=10, lr=0.001, save_path=None):
        """
        Train the denoiser model.
        
        Args:
            dataloader: DataLoader with pairs of noisy and clean images
            epochs: Number of training epochs
            lr: Learning rate
            save_path: Path to save the trained model
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (noisy, clean) in enumerate(dataloader):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(noisy)
                loss = criterion(output, clean)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
            
            print(f'Epoch: {epoch+1}/{epochs}, Average Loss: {epoch_loss/len(dataloader):.6f}')
        
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f'Model saved to {save_path}')
            
        self.model.eval()  # Set back to evaluation mode