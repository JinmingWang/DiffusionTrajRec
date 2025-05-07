from JimmyTorch.Models import *
import torch.nn as nn
from math import sqrt, log


class NumberEmbedder(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, max_num: int, hidden_dim: int = 256, embed_dim: int = 128) -> None:
        super().__init__()

        # --- Diffusion step Encoding ---
        position = torch.arange(max_num, dtype=torch.float32, device=self.device).unsqueeze(1)  # (max_time, 1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32, device=self.device) * -(
                log(1.0e4) / hidden_dim))  # (feature_dim / 2)
        self.encodings = torch.zeros((max_num, hidden_dim), dtype=torch.float32,
                                     device=self.device)  # (max_time, feature_dim)
        self.encodings[:, 0::2] = torch.sin(position * div_term)
        self.encodings[:, 1::2] = torch.cos(position * div_term)

        self.proj = nn.Linear(hidden_dim, embed_dim)  # (B, embed_dim)

    def forward(self, num: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        time_embed = self.encodings[num, :]  # (B, hidden_dim)
        return self.proj(time_embed)  # (B, embed_dim, 1)


class GRU_Conv(nn.Module):
    """
    This is actually a variant of GRU, but the state is in shape (B, state_c, L),
    The linear operations are replaced with 1D convolutions.
    """
    def __init__(self, in_c: int, state_c: int, max_t: int):
        super().__init__()

        self.t_embedder = nn.Sequential(
            nn.Embedding(max_t, 128),
            nn.Unflatten(-1, (-1, 1))
        )
        self.embed_1 = nn.Conv1d(128, state_c * 3, 1, 1, 0)
        self.embed_2 = nn.Conv1d(128, state_c * 2, 1, 1, 0)

        self.input_proj = nn.Conv1d(in_c, state_c * 3, 3, 1, 1)

        self.state_proj = nn.Conv1d(state_c, state_c * 2, 3, 1, 1)

        self.state_r_proj = nn.Conv1d(state_c, state_c, 3, 1, 1)


    def forward(self, x: Tensor, prev_h: Tensor, t: Tensor):
        # input_state: (B, state_c, L)
        # prev_hidden_state: (B, state_c, L)
        # output: (B, state_c, L) the next hidden state

        t_embed = self.t_embedder(t)

        Wz_xt, Wr_xt, Wh_xt = torch.chunk(self.input_proj(x) + self.embed_1(t_embed), 3, dim=1)

        Uz_ht, Ur_ht = torch.chunk(self.state_proj(prev_h) + self.embed_2(t_embed), 2, dim=1)

        z = torch.sigmoid(Wz_xt + Uz_ht)

        r = torch.sigmoid(Wr_xt + Ur_ht)

        h_tilde = torch.tanh(Wh_xt + self.state_r_proj(r * prev_h))

        h = (1 - z) * prev_h + z * h_tilde

        return h


class GRU_Linear(nn.Module):
    """
    This is actually a variant of GRU, but the state is in shape (B, state_c, L),
    The linear operations are replaced with 1D convolutions.
    """
    def __init__(self, in_c: int, state_c: int, max_t: int):
        super().__init__()

        self.t_embedder = nn.Sequential(
            nn.Embedding(max_t, 128),
            nn.Unflatten(-1, (-1, 1))
        )
        self.embed_1 = nn.Conv1d(128, state_c * 3, 1, 1, 0)
        self.embed_2 = nn.Conv1d(128, state_c * 2, 1, 1, 0)

        self.input_proj = nn.Conv1d(in_c, state_c * 3, 1, 1, 0)

        self.state_proj = nn.Conv1d(state_c, state_c * 2, 1, 1, 0)

        self.state_r_proj = nn.Conv1d(state_c, state_c, 1, 1, 0)


    def forward(self, x: Tensor, prev_h: Tensor, t: Tensor):
        # input_state: (B, state_c, L)
        # prev_hidden_state: (B, state_c, L)
        # output: (B, state_c, L) the next hidden state

        t_embed = self.t_embedder(t)

        Wz_xt, Wr_xt, Wh_xt = torch.chunk(self.input_proj(x) + self.embed_1(t_embed), 3, dim=1)

        Uz_ht, Ur_ht = torch.chunk(self.state_proj(prev_h) + self.embed_2(t_embed), 2, dim=1)

        z = torch.sigmoid(Wz_xt + Uz_ht)

        r = torch.sigmoid(Wr_xt + Ur_ht)

        h_tilde = torch.tanh(Wh_xt + self.state_r_proj(r * prev_h))

        h = (1 - z) * prev_h + z * h_tilde

        return h


class GNSwishConv(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, k: int, s: int, p: int):
        super().__init__(
            nn.GroupNorm(32, in_c),
            nn.SiLU(inplace=True),
            nn.Conv1d(in_c, out_c, k, s, p),
        )


class TW8_Res(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 state_c: int,
                 embed_c: int,
                 expand: int,
                 resize: Literal["up", "down", "same"] = "same") -> None:
        super().__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.state_c = state_c

        mid_c = in_c * expand

        self.embed_proj = nn.Conv1d(embed_c, in_c, 1, 1, 0)

        self.res_block = nn.Sequential(
            GNSwishConv(in_c + state_c, mid_c, 3, 1, 1),
            SELayer1D(mid_c, 1),
            GNSwishConv(mid_c, in_c + state_c, 3, 1, 1),
        )

        nn.init.zeros_(self.res_block[-1][-1].weight)
        nn.init.zeros_(self.res_block[-1][-1].bias)

        if resize == "up":
            self.resize = nn.Upsample(scale_factor=2, mode="linear")
        elif resize == "down":
            self.resize = nn.MaxPool1d(2)
        else:
            self.resize = nn.Identity()

        self.x_proj = nn.Conv1d(in_c, out_c, 1, 1, 0)

    def forward(self, x, state_feature, mix_embed):
        # x: (B, C, L)
        # mix_embed: (B, C_m, 1)
        identity = x
        x = x + self.embed_proj(mix_embed)
        x = torch.cat([x, state_feature], dim=1)
        x = self.res_block(x)
        x, state_feature = x[:, :self.in_c], x[:, self.in_c:]
        return self.x_proj(self.resize(x + identity)), state_feature


class MidAttnBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, expend: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.H = expend * 2
        self.d_qk = in_c // 2 * self.H

        self.qkv_proj = GNSwishConv(in_c, in_c * self.H * 2, 1, 1, 0)

        self.scale = nn.Parameter(torch.tensor(1 / sqrt(in_c)))

        self.reshaper = Rearrange("B (H C) L -> (B H) C L", H=self.H)

        self.out_proj = nn.Sequential(
            Rearrange("(B H) C L -> B (H C) L", H=self.H),
            nn.GroupNorm(32, in_c * self.H),
            nn.Dropout(dropout),
            nn.Conv1d(in_c * self.H, out_c, 1, 1, 0),
            GNSwishConv(out_c, out_c, 3, 1, 1)
        )

        nn.init.zeros_(self.out_proj[-1][-1].weight)
        nn.init.zeros_(self.out_proj[-1][-1].bias)

        self.shortcut = nn.Conv1d(in_c, out_c, 1, 1, 0) if in_c != in_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = torch.chunk(self.qkv_proj(x), 3, dim=1)
        q = self.reshaper(q)    # (B*H, C, L)
        k = self.reshaper(k)    # (B*H, C, L)
        v = self.reshaper(v)    # (B*H, C, L)

        attn = torch.softmax((k.transpose(1, 2) @ q) * self.scale, dim=1)  # (B*H, L, L)

        return self.shortcut(x) + self.out_proj(v @ attn)