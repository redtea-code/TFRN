import lightning as L
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

adaptive_filter = True


class complex_conv(torch.nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 1)

    def forward(self, real, imag):
        real_out = self.conv1(real)
        imag_out = self.conv2(imag)
        return real_out, imag_out


class TDC(L.LightningModule):
    """Time-Domain Convolution"""

    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class FDC(nn.Module):
    "Frequency Domain Convolution"

    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv1d(dim, dim, 1)
        dim = dim // 2 + 1
        self.cp_conv1 = complex_conv(dim, dim)
        self.cp_conv2 = complex_conv(dim, dim)
        self.threshold_param = nn.Parameter(torch.rand(1))  # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = ((
                                 normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        x_res = self.conv1(x)
        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        x_real, x_imag = x_fft.real, x_fft.imag
        x_real, x_imag = self.cp_conv1(x_real, x_imag)
        x_fft = torch.complex(x_real, x_imag)

        if adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            x_real, x_imag = x_masked.real, x_masked.imag
            x_real, x_imag = self.cp_conv2(x_real, x_imag)
            x_masked = torch.complex(x_real, x_imag)

            x_weighted = x_fft + x_masked

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x + x_res
        x = x.view(B, N, C)  # Reshape back to original shape
        return x


class FCB_block(nn.Module):
    """学习表示"""

    def __init__(self, in_features, hidden_features, drop, seq_len=22):
        super().__init__()
        self.conv_a = TDC(in_features, hidden_features, drop)
        self.LN = nn.LayerNorm(in_features)
        self.conv_b = FDC(seq_len)
        self.LN = nn.LayerNorm(in_features)

    def forward(self, x):
        """input  x:b,l,n   c:b,n  """
        x = self.conv_a(x)
        x = self.LN(x)
        x = self.conv_b(x)
        x = self.LN(x)
        return x


if __name__ == '__main__':
    in_features = 16
    hidden_features = 32
    drop = 0.1
    model = TDC(in_features, hidden_features, drop)
    input = torch.randn(4, 22, in_features)
    output = model(input)
    print("TDC输入张量形状:", input.shape)
    print("TDC输出张量形状:", output.shape)
    print("-------------------------")

    dim = in_features  # 设置输入维度
    model = FDC(22)
    input = torch.randn(8, 22, dim)  # (batch_size, sequence_length, input_dim)
    output = model(input)
    print("FDC输入张量形状:", input.shape)
    print("FDC输出张量形状:", output.shape)
