from JimmyTorch.Models import *

class MultiLevelStagePropagator(nn.Module):
    def __init__(self, d_state: int, d_cond: int, n_heads: int = 4):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond, d_state),
            nn.LayerNorm(d_state),
            nn.SiLU(inplace=True),
            nn.Linear(d_state, d_state),
        )

        self.cross_attn = nn.MultiheadAttention(d_state, n_heads, batch_first=True)

    def forward(self, h, cond):
        """
        :param h: the state features n_stages * (B, d_state, L_state)
        :param cond: the conditional feature (B, L_cond, d_cond)
        """
        state_len_list = [h_i.shape[2] for h_i in h]
        cat_h = torch.cat(h, dim=2).transpose(1, 2)     # (B, L_all_state, d_state)
        cond = self.cond_proj(cond)     # (B, L_cond, d_state)

        cat_h = cat_h + self.cross_attn(cat_h, cond, cond)[0]     # (B, L_all_state, d_state)
        cat_h = cat_h.transpose(1, 2)     # (B, d_state, L_all_state)

        return list(torch.split(cat_h, state_len_list, dim=2))     # n_stages * (B, d_state, L_state)


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