import torch
import torch.nn as nn
import torch.nn.functional as F



class sAT(nn.Module):
    def __init__(self, input_size, cond_size, hidden_size):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(input_size + cond_size, hidden_size),
                                    nn.Tanh()
                                    )
        self.dense4 = Gate(hidden_size, input_size)

    def forward(self, a, c):
        if len(a.shape) == 2:
            merge_dim = 1
        else:
            merge_dim = 2
        s_u = self.dense1(torch.cat([a, c], dim=merge_dim))
        a_c = F.softmax(self.dense4(s_u), dim=merge_dim)
        out = a_c * a
        return out


class Gate(nn.Module):
    """门控机制"""

    def __init__(self, input_size, hidden_size, ):
        super().__init__()
        self.dense3 = nn.Linear(input_size, hidden_size)
        self.dense4 = nn.Linear(input_size, hidden_size)

    def forward(self, a):
        gate = F.sigmoid(self.dense3(a)) * self.dense4(a)
        return gate
