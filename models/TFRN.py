from torch import nn
from models.block.MIXer import *


class TFRN_base(nn.Module):
    """包含t-365 - t-365 + 22"""

    def __init__(self, month=9):
        super().__init__()
        # self.backbone = DIT2()
        self.month = month
        self.backbone = TFRN_model(static_size=52)
        # self.backbone = TFT(static_size=108)

    def forward(self, seq_x, seq_y_past, attr, device="cuda"):
        attr = torch.where(torch.isnan(attr), torch.full_like(attr, 0), attr).float()

        seq_y_past, seq_y_ppast = seq_y_past[:, -15:, :], seq_y_past[:, self.month * 30:self.month * 30 + 22, :]
        seq_x, seq_past = seq_x[:, -22:, :], seq_x[:, self.month * 30:self.month * 30 + 22, :]
        batch, seq_len = seq_x.shape[0], 22

        seq = torch.zeros(batch, seq_len, 1).to(device)
        seq[:, :-7], seq[:, -7:] = seq_y_past, 0

        seq = torch.cat([seq_x, seq], dim=2)
        seq_p = torch.cat([seq_past, seq_y_ppast], dim=2)

        output = self.backbone(seq, attr[:, 0, :], seq_p)
        return output


class TFRN_lenchose(nn.Module):
    """包含t-365 - t-365 + 22"""

    def __init__(self, month=0, seq_len=180, seq_past_len=187):
        super().__init__()
        # self.backbone = DIT2()
        self.month = month
        self.seq_len = seq_len
        self.seq_past_len = seq_past_len
        # self.backbone = TFRN_wo(static_size=108)
        self.backbone = TFRN_len(static_size=52, seq_len=seq_len + 7, seq_past_len=seq_past_len)
        # self.backbone = TFT(static_size=108)

    def forward(self, seq_x, seq_y_past, attr, device="cuda"):
        attr = torch.where(torch.isnan(attr), torch.full_like(attr, 0), attr).float()

        seq_y_past, seq_y_ppast = seq_y_past[:, -self.seq_len:, :], seq_y_past[:,
                                                                    self.month * 30:self.month * 30 + self.seq_past_len,
                                                                    :]
        seq_x, seq_past = seq_x[:, -self.seq_len - 7:, :], seq_x[:, self.month * 30:self.month * 30 + self.seq_past_len,
                                                           :]
        # 处理 input
        batch, seq_len = seq_x.shape[0], self.seq_len + 7
        seq = torch.zeros(batch, seq_len, 1).to(device)
        seq[:, :-7], seq[:, -7:] = seq_y_past, 0
        seq = torch.cat([seq_x, seq], dim=2)

        seq_p = torch.cat([seq_past, seq_y_ppast], dim=2)

        output = self.backbone(seq, attr[:, 0, :], seq_p)
        return output

