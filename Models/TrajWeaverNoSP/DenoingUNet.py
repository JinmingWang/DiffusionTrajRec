from JimmyTorch.Models import *
    
class UNetBlock(nn.Module):
    def __init__(self, 
                 d_in: int,
                 d_hidden: int, 
                 d_out: int,
                 d_cond: int,
                 scaling: Literal["down", "same", "up"]):
        """
        :param d_in: input dim
        :param d_state: the dim of state feature
        :param d_hidden: the hidden dimension
        :param d_out: the output dimension
        :param scaling: how to scale the output
        """
        super().__init__()

        self.ca = nn.MultiheadAttention(d_in, 8, 0.0, kdim=d_cond, vdim=d_cond, batch_first=True)

        self.se_res = nn.Sequential(
            GnSiLUConv1D(d_in, d_hidden, 3, 1, 1, gn_groups=16),
            SELayer1D(d_hidden, 4),
            GnSiLUConv1D(d_hidden, d_hidden, 3, 1, 1, gn_groups=16),
            GnSiLUConv1D(d_hidden, d_in, 3, 1, 1, gn_groups=16),
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
        x = x + self.ca(x.transpose(1, 2), cond, cond)[0].transpose(1, 2)
        x = x + self.se_res(x)
        return self.out_proj(x)


class DenoisingUNet(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 d_cond: int,
                 d_list: list[int],
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
                                              d_hidden=d_list[i] * 2, d_out=d_list[i + 1],
                                              scaling="down"))

        self.mid_block = UNetBlock(d_list[-1], d_list[-1] * 2, d_list[-1], d_cond=d_cond, scaling="same")

        self.merge_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(self.n_stages, 0, -1):
            self.merge_blocks.append(nn.Conv1d(d_list[i] * 2, d_list[i], 3, 1, 1))
            self.up_blocks.append(UNetBlock(d_in=d_list[i], d_cond=d_cond,
                                            d_hidden=d_list[i] * 2, d_out=d_list[i - 1],
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

        x = self.mid_block(x, cond_embed)  # (B, d-1, l)

        for i, block in enumerate(self.up_blocks):
            x = self.merge_blocks[i](torch.cat([x, x_list.pop()], dim=1))
            x = block(x, cond_embed)

        return self.head(x)