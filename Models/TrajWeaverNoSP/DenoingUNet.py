from JimmyTorch.Models import *
    
class UNetBlock(nn.Module):
    def __init__(self, 
                 d_in: int,
                 d_hidden: int, 
                 d_out: int,
                 d_cond: int,
                 n_heads: int,
                 scaling: Literal["down", "same", "up"]):
        """
        :param d_in: input dim
        :param d_state: the dim of state feature
        :param d_hidden: the hidden dimension
        :param d_out: the output dimension
        :param scaling: how to scale the output
        """
        super().__init__()

        self.ln = nn.LayerNorm(d_in)
        self.ca = nn.MultiheadAttention(d_in, n_heads, 0.0, kdim=d_cond, vdim=d_cond, batch_first=True)

        self.res1 = nn.Sequential(
            GnSiLUConv1D(d_in, d_hidden, 3, 1, 1, gn_groups=16),
            GnSiLUConv1D(d_hidden, d_in, 3, 1, 1, gn_groups=16),
        )

        self.res2 = nn.Sequential(
            GnSiLUConv1D(d_in, d_hidden, 3, 1, 1, gn_groups=16),
            GnSiLUConv1D(d_hidden, d_in, 3, 1, 1, gn_groups=16),
        )

        self.self_attn = nn.Sequential(
            Transpose(1, 2),
            PosEncoderSinusoidal(d_in, 513, "add"),
            nn.LayerNorm(d_in),
            MHSA(d_in, n_heads),
            Transpose(1, 2),
        )

        self.out_proj = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest") if scaling == "up" else nn.Identity(),
            GnSiLUConv1D(d_in, d_out, 3, 1 + int(scaling == "down"), 1, gn_groups=16),
        )

    def forward(self, x, cond):
        """
        
        :param x: the input hidden features
        :param f: the state feature for this block
        :param cond: the conditional features
        """
        x = x + self.ca(self.ln(x.transpose(1, 2)), cond, cond)[0].transpose(1, 2)
        x = x + self.res1(x)
        x = x + self.res2(x)
        x = x + self.self_attn(x)
        return self.out_proj(x)



class MidTransformerBlock(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_hidden: int,
                 n_heads: int,
                 ):
        super().__init__()

        self.transformer = nn.Sequential(
            Transpose(1, 2),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_in,
                    nhead=n_heads,
                    dim_feedforward=d_hidden,
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=2,
            ),
            Transpose(1, 2),
        )

        self.out_proj = GnSiLUConv1D(d_in, d_in, 3, 1, 1, gn_groups=16)

    def forward(self, x):
        return self.out_proj(self.transformer(x))


class DenoisingUNet(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 d_cond: int,
                 d_list: list[int],
                 n_heads: int,
                 ):
        super().__init__()

        self.d_list = d_list
        self.n_stages = len(d_list) - 1

        self.stem = nn.Sequential(  # (B, L, d_in)
            Transpose(1, 2),
            Conv1DGnSiLU(d_in, d_list[0], k=3, s=1, p=1, gn_groups=16),
            nn.Conv1d(d_list[0], d_list[0], 3, 1, 1)
        )

        self.down_blocks = nn.ModuleList()
        for i in range(self.n_stages):
            self.down_blocks.append(UNetBlock(d_in=d_list[i], d_cond=d_cond,
                                              d_hidden=d_list[i] * 2, d_out=d_list[i + 1], n_heads=n_heads,
                                              scaling="down"))

        self.mid_block = MidTransformerBlock(d_list[-1], d_list[-1] * 2, n_heads=n_heads)

        self.merge_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(self.n_stages, 0, -1):
            self.merge_blocks.append(nn.Conv1d(d_list[i] * 2, d_list[i], 3, 1, 1))
            self.up_blocks.append(UNetBlock(d_in=d_list[i], d_cond=d_cond,
                                            d_hidden=d_list[i] * 2, d_out=d_list[i - 1], n_heads=n_heads,
                                            scaling="up"))

        self.head = nn.Sequential(
            GnSiLUConv1D(d_list[0], d_list[0], k=3, s=1, p=1, gn_groups=16),
            GnSiLUConv1D(d_list[0], d_out, k=3, s=1, p=1, gn_groups=16),
            Transpose(1, 2)  # (B,L, d_out)
        )

    def forward(self, x, cond_embed):
        # During eval, the embed is computed only once
        # so it will be passed as a parameter
        # if self.training:

        x = self.stem(x)  # (B, L, d_in) -> (B, d0, L)

        x_list = []
        for block in self.down_blocks:
            x = block(x, cond_embed)
            x_list.append(x)

        x = self.mid_block(x)  # (B, d-1, l)

        for i, block in enumerate(self.up_blocks):
            x = self.merge_blocks[i](torch.cat([x, x_list.pop()], dim=1))
            x = block(x, cond_embed)

        return self.head(x)