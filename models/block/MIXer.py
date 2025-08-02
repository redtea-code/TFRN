from models.block.DIT import *
from models.block.FCB import FCB_block

from torch.nn import functional as F

from models.block.SAT import sAT


class MultiScaleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[16, 32, 64]):
        """
        :param input_size: 输入特征的维度大小
        :param hidden_sizes: 包含不同隐藏层大小的列表，例如 [64, 128, 256]
        """
        super(MultiScaleMLP, self).__init__()

        # 根据 hidden_sizes 构建多个 MLP 分支，每个分支有一个指定的隐藏层大小
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2)  # 输出特征减半
            )
            for hidden_size in hidden_sizes
        ])
        self.Linear = nn.Linear(sum(hidden_sizes) // 2, input_size)

    def forward(self, x):
        if len(x.shape) == 3:
            merge_dim = 2
        else:
            merge_dim = 1
        outputs = [mlp(x) for mlp in self.mlps]

        # 将每个分支的输出拼接在一起
        out = torch.cat(outputs, dim=merge_dim)
        out = F.softmax(self.Linear(out), dim=-1) * x
        return out


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads=4, mlp_ratio=4, drop_rate=0.2):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = Attention(hidden_size, num_heads=n_heads)
        self.ffn = Mlp(in_features=hidden_size, hidden_features=hidden_size * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, enc_layer_inputs):
        residual1 = enc_layer_inputs.clone()
        enc_self_attn_outputs = self.enc_self_attn(enc_layer_inputs)
        outputs1 = self.norm1(enc_self_attn_outputs + residual1)

        residual2 = outputs1.clone()
        ffn_outputs = self.ffn(outputs1)
        # ffn_outputs: [batch_size, src_len, d_model]
        ffn_outputs = self.dropout(ffn_outputs)
        outputs2 = self.norm2(ffn_outputs + residual2)

        return outputs2


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super(ScaledDotProductAttention, self).__init__()
        self.d_head = d_head

    def forward(self, Q, K, V, mask):
        """
        Q: [batch_size, n_heads, len_q, d_head)]
        K: [batch_size, n_heads, len_k(=len_v), d_head]
        V: [batch_size, n_heads, len_v(=len_k), d_head]
        mask: [batch_size, n_heads, seq_len, seq_len]
        """
        qk = torch.matmul(Q, K.transpose(-1, -2))
        scores = qk / torch.sqrt(
            torch.tensor(self.d_head, dtype=torch.float32))  # scores : [batch_size, n_heads, len_q, len_k]
        assert torch.isnan(qk).sum() == 0
        assert torch.isnan(scores).sum() == 0

        if mask is not None:
            scores += mask

        attn = torch.softmax(scores, dim=-1)
        # attn: [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V)
        # context: [batch_size, n_heads, len_q, d_head]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, n_head=4):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k(=len_v), d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        context = ScaledDotProductAttention(d_head=self.d_head)(Q, K, V, mask)
        assert torch.isnan(context).sum() == 0

        # context: [batch_size, n_heads, len_q, d_head]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_head)
        output = self.fc(context)
        # output: [batch_size, len_q, d_model]
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model=64, dropout_rate=0.4, d_ff=256, n_head=4):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dec_enc_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = Mlp(in_features=d_model, hidden_features=d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, dec_layer_inputs, enc_outputs, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        """
        dec_layer_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        residual1 = dec_layer_inputs.clone()
        dec_self_attn_outputs = self.dec_self_attn(dec_layer_inputs, dec_layer_inputs, dec_layer_inputs,
                                                   dec_self_attn_mask)
        outputs1 = self.norm1(dec_self_attn_outputs + residual1)

        residual2 = outputs1.clone()
        dec_enc_attn_outputs = self.dec_enc_attn(outputs1, enc_outputs, enc_outputs, dec_enc_attn_mask)
        outputs2 = self.norm2(dec_enc_attn_outputs + residual2)

        residual3 = outputs2.clone()
        ffn_outputs = self.ffn(outputs2)
        ffn_outputs = self.dropout(ffn_outputs)
        outputs3 = self.norm3(ffn_outputs + residual3)

        return outputs3


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, ):
        """x: (B,N*2) c: (B,N)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        attn_weight, attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_msa * attn_weight
        mlp_weight = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * mlp_weight
        return x, attn


class TFRN_model(nn.Module):
    """sota"""

    def __init__(self, input_size=6, hidden_size=16, static_size=52, pre_len=7):
        super().__init__()
        self.pre_len = pre_len
        self.proj = nn.Linear(static_size, hidden_size)
        self.proj2 = nn.Linear(static_size, hidden_size)

        self.represent = FCB_block(input_size + hidden_size, 32, drop=0.2, seq_len=22)
        self.represent2 = FCB_block(input_size + hidden_size, 32, drop=0.2, seq_len=22)

        self.SAT = sAT(input_size + hidden_size, input_size + hidden_size, hidden_size)

        self.DIT = nn.ModuleList([
            DiTBlock(hidden_size=hidden_size + input_size, num_heads=2, mlp_ratio=4.0, ) for _ in range(3)
        ])
        self.LSTM = nn.LSTM(input_size + hidden_size, hidden_size, batch_first=True)
        self.proj3 = nn.Linear(hidden_size, hidden_size)
        self.LSTM2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.head = nn.Linear(hidden_size, 1)

    def forward(self, seq, attr, seq_p):
        """input  x:b,l,n   c:b,n  """
        seq_len = seq.shape[1]

        attr = attr[:, None, :].expand(-1, seq_len, -1)
        # attr_ = torch.fft.rfft(attr, dim=1, norm='ortho')
        C1 = self.proj(attr)

        C2 = self.proj2(attr)

        seq = torch.cat([seq, C1], dim=2)
        seq_p = torch.cat([seq_p, C2], dim=2)

        seq = self.SAT(seq, seq)
        seq = self.represent(seq)
        seq_p = self.represent2(seq_p)

        for block in self.DIT:
            out, attn = block(seq, seq_p)

        out, _ = self.LSTM(out)
        out = self.proj3(out)
        out, _ = self.LSTM2(out)

        out = out[:, -self.pre_len:, ]
        out = self.head(out)
        return out


class TFRN_len(nn.Module):
    """长度选择"""

    def __init__(self, input_size=6, hidden_size=16, static_size=52, pre_len=7, seq_len=30, seq_past_len=30):
        super().__init__()
        self.pre_len = pre_len
        self.proj = nn.Linear(static_size, hidden_size)
        self.proj2 = nn.Linear(static_size, hidden_size)

        self.represent = nn.ModuleList([
            FCB_block(input_size + hidden_size, 32, drop=0.2, seq_len=seq_len) for _ in range(1)
        ])
        self.represent2 = nn.ModuleList([
            FCB_block(input_size + hidden_size, 32, drop=0.2, seq_len=seq_past_len) for _ in range(1)
        ])

        self.SAT = sAT(input_size + hidden_size, input_size + hidden_size, hidden_size)

        self.DIT = nn.ModuleList([
            DiTBlock(hidden_size=input_size + hidden_size, num_heads=2, mlp_ratio=4.0, ) for _ in range(1)
        ])
        self.LSTM = nn.LSTM((input_size + hidden_size), hidden_size, batch_first=True)  #
        self.proj3 = nn.Linear(hidden_size, hidden_size)
        self.LSTM2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, seq, attr, seq_p):
        """input  x:b,l,n   c:b,n  """
        seq_len = seq.shape[1]
        seq_len2 = seq_p.shape[1]
        attr1 = attr[:, None, :].expand(-1, seq_len, -1)
        attr2 = attr[:, None, :].expand(-1, seq_len2, -1)
        C1 = self.proj(attr1)
        C2 = self.proj2(attr2)
        seq = torch.cat([seq, C1], dim=2)
        seq_p = torch.cat([seq_p, C2], dim=2)

        seq = self.SAT(seq, seq)
        for block in self.represent:
            seq = block(seq)
        for block in self.represent2:
            seq_p = block(seq_p)

        seq_p = seq_p[:, -seq_len:, :]
        for block in self.DIT:
            out, attn = block(seq, seq_p)

        out, _ = self.LSTM(out)
        out = self.proj3(out)
        out, _ = self.LSTM2(out)

        out = out[:, -self.pre_len:, ]
        out = self.head(out)
        return out


if __name__ == '__main__':
    from thop import profile

    # x = torch.rand((4, 187, 6))
    # x_p = torch.rand((4, 187, 6))
    x = torch.rand((4, 22, 6))
    x_p = torch.rand((4, 22, 6))
    c = torch.rand(4, 52)
    # model = Backbone()
    model = TFRN_model(input_size=6, hidden_size=16, static_size=52, pre_len=7)

    flops, params = profile(model, inputs=(x, c, x_p))
    print(f"Params: {params / 1e6:.4f}M")
    print(f"FLOPs: {(flops + 980.98) / 1e9:.4f}G")
