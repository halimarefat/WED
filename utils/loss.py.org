import torch
import torch.nn.functional as F

def dwt(x, wavelet='haar'):
    """Perform discrete wavelet transform using Haar wavelet"""
    wavelet_filters = {
        'haar': torch.tensor([1, 1], dtype=torch.float64) / 2**0.5,
    }

    if wavelet not in wavelet_filters:
        raise ValueError(f"Unsupported wavelet: {wavelet}")

    h = wavelet_filters[wavelet].to(x.device)
    lpf = h.unsqueeze(0).unsqueeze(0)
    hpf = h.flip(0).unsqueeze(0).unsqueeze(0)

    # Ensure input is 3D (batch_size, channels, sequence_length)
    if x.dim() == 2:
        x = x.unsqueeze(1)  # Add a channel dimension
    elif x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Perform convolution and downsampling
    low = F.conv1d(x, lpf, stride=2)
    high = F.conv1d(x, hpf, stride=2)

    return low, high

def idwt(low, high, wavelet='haar'):
    """Perform inverse discrete wavelet transform using Haar wavelet"""
    wavelet_filters = {
        'haar': torch.tensor([1, 1], dtype=torch.float64) / 2**0.5,
    }

    if wavelet not in wavelet_filters:
        raise ValueError(f"Unsupported wavelet: {wavelet}")

    h = wavelet_filters[wavelet].to(low.device)
    lpf = h.unsqueeze(0).unsqueeze(0)
    hpf = h.flip(0).unsqueeze(0).unsqueeze(0)

    low = F.conv_transpose1d(low, lpf, stride=2)
    high = F.conv_transpose1d(high, hpf, stride=2)

    return low + high

class WaveletLoss(torch.nn.Module):
    def __init__(self, wavelet='haar'):
        super(WaveletLoss, self).__init__()
        self.wavelet = wavelet

    def forward(self, input, target):
        # Ensure input and target have the same shape
        assert input.shape == target.shape

        # Apply wavelet transform
        low_input, high_input = dwt(input, self.wavelet)
        low_target, high_target = dwt(target, self.wavelet)

        # Compute the loss on the wavelet coefficients
        loss = F.mse_loss(low_input, low_target) + F.mse_loss(high_input, high_target)

        return loss