from JimmyTorch.Models import *


class CGRU(nn.Module):
    """
    Convolutional GRU
    """
    def __init__(self, d_in: int, d_state: int):
        super().__init__()
        self.input_proj = GnSiLUConv1D(d_in, d_state * 3, k=3, s=1, p=1, gn_groups=8)
        self.state_proj = nn.Conv1d(d_state, d_state * 2, 3, 1, 1)
        self.state_r_proj = nn.Conv1d(d_state, d_state, 3, 1, 1)

    def forward(self, x, h):
        Wz_xt, Wr_xt, Wh_xt = torch.chunk(self.input_proj(x), 3, dim=1)
        Uz_ht, Ur_ht = torch.chunk(self.state_proj(h), 2, dim=1)

        z = torch.sigmoid(Wz_xt + Uz_ht)
        r = torch.sigmoid(Wr_xt + Ur_ht)
        h_tilde = torch.tanh(Wh_xt + self.state_r_proj(r * h))

        return (1 - z) * h + z * h_tilde

    
class TrajWeaverBlock(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_state: int, 
                 d_cond: int, 
                 d_hidden: int, 
                 d_out: int, 
                 n_heads: int, 
                 dropout: float, 
                 scaling: Literal["down", "same", "up"]):
        """
        :param d_in: input dim
        :param d_state: the dim of state feature
        :param d_cond: the dim of coditional feature
        :param d_hidden: the hidden dimension
        :param d_out: the output dimension
        :param n_heads: number of heads in the cross attention
        :param dropout: the dropout used in the cross attention
        :param scaling: how to scale the output
        """
        super().__init__()

        self.in_proj = nn.Sequential(
            Transpose(1, 2),
            nn.Linear(d_in + d_state, d_hidden, bias=False),
            nn.LayerNorm(d_hidden),
        )

        self.cross_attn = nn.MultiheadAttention(d_hidden, n_heads, dropout, kdim=d_cond, vdim=d_cond, batch_first=True)

        self.mid_proj = nn.Sequential(
            Transpose(1, 2),
            nn.SiLU(inplace=True),
        )

        self.res = nn.Sequential(
            Conv1DGnSiLU(d_hidden, d_hidden, 3, 1, 1, gn_groups=8),
            nn.Conv1d(d_hidden, d_hidden, 3, 1, 1)
        )

        self.state_proj = CGRU(d_hidden, d_state)

        self.out_proj = nn.Sequential(
            nn.Upsample(scale_factor=2) if scaling == "up" else nn.Identity(),
            GnSiLUConv1D(d_hidden, d_out, 3, 1 + int(scaling == "down"), 1, gn_groups=8),
        )

    def forward(self, x, h, cond):
        """
        
        :param x: the input hidden features
        :param f: the state feature for this block
        :param cond: the conditional features
        """
        x = self.in_proj(torch.cat([x, h], dim=1))
        x = x + self.cross_attn(x, cond, cond)[0]
        x = self.mid_proj(x)
        x = x + self.res(x)
        return self.out_proj(x), self.state_proj(x, h)