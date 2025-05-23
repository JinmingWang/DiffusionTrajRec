from JimmyTorch.Models import *

    
class UNetBlock(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_state: int,
                 d_hidden: int,
                 d_out: int,
                 num_heads: int,
                 scaling: Literal["down", "same", "up"]):
        """
        :param d_in: input dim
        :param d_state: the dim of state feature
        :param d_hidden: the hidden dimension
        :param d_out: the output dimension
        :param scaling: how to scale the output
        """
        super().__init__()

        d_comb = d_in + d_state

        self.res1 = nn.Sequential(
            GnSiLUConv1D(d_comb, d_hidden, 3, 1, 1, gn_groups=16),
            GnSiLUConv1D(d_hidden, d_comb, 3, 1, 1, gn_groups=16),
        )

        self.res2 = nn.Sequential(
            GnSiLUConv1D(d_comb, d_hidden, 3, 1, 1, gn_groups=16),
            GnSiLUConv1D(d_hidden, d_comb, 3, 1, 1, gn_groups=16),
        )

        self.self_attn = nn.Sequential(
            Transpose(1, 2),
            PosEncoderSinusoidal(d_comb, 513, "add"),
            nn.LayerNorm(d_comb),
            MHSA(d_comb, num_heads),
            Transpose(1, 2),
        )

        self.state_proj = GnSiLUConv1D(d_comb, d_state, 1, 1, 0, gn_groups=16)
        nn.init.zeros_(self.state_proj[2].weight)
        nn.init.zeros_(self.state_proj[2].bias)

        self.out_proj = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest") if scaling == "up" else nn.Identity(),
            GnSiLUConv1D(d_comb, d_out, 3, 1 + int(scaling == "down"), 1, gn_groups=16),
        )

    def forward(self, x, state_feature):
        """
        
        :param x: the input hidden features
        :param f: the state feature for this block
        :param cond: the conditional features
        """
        x = torch.cat([x, state_feature], dim=1)
        x = x + self.res1(x)
        x = x + self.res2(x)
        x = x + self.self_attn(x)
        return self.out_proj(x), self.state_proj(x) + state_feature


class MidTransformerBlock(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_hidden: int,
                 n_heads: int,
                 ):
        super().__init__()

        d_comb = d_in + d_state

        self.transformer = nn.Sequential(
            Transpose(1, 2),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_comb,
                    nhead=n_heads,
                    dim_feedforward=d_hidden,
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=2,
            ),
            Transpose(1, 2),
        )

        self.state_proj = GnSiLUConv1D(d_comb, d_state, 3, 1, 1, gn_groups=16)
        self.out_proj = GnSiLUConv1D(d_comb, d_in, 3, 1, 1, gn_groups=16)

    def forward(self, x, state_feature):
        x = torch.cat([x, state_feature], dim=1)
        x = self.transformer(x)
        return self.out_proj(x), self.state_proj(x)


class DenoisingUNet(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 d_state: int,
                 n_heads: int,
                 d_list: list[int],
                 ):
        super().__init__()

        self.d_list = d_list
        self.d_state = d_state
        self.n_stages = len(d_list) - 1

        self.stem = nn.Sequential(  # (B, L, d_in)
            Transpose(1, 2),
            Conv1DGnSiLU(d_in, d_list[0], k=3, s=1, p=1, gn_groups=16),
            nn.Conv1d(d_list[0], d_list[0], 3, 1, 1)
        )

        self.down_blocks = nn.ModuleList()
        for i in range(self.n_stages):
            self.down_blocks.append(UNetBlock(d_in=d_list[i], d_state=d_state, d_hidden=d_list[i] * 2,
                                              d_out=d_list[i + 1], num_heads=n_heads, scaling="down"))

        self.mid_block = MidTransformerBlock(d_list[-1], d_state, d_list[-1] * 2, n_heads)

        self.merge_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(self.n_stages, 0, -1):
            self.merge_blocks.append(nn.Conv1d(d_list[i] * 2, d_list[i], 3, 1, 1))
            self.up_blocks.append(UNetBlock(d_in=d_list[i], d_state=d_state, d_hidden=d_list[i] * 2,
                                            d_out=d_list[i - 1], num_heads=n_heads, scaling="up"))

        self.head = nn.Sequential(
            GnSiLUConv1D(d_list[0], d_list[0], k=3, s=1, p=1, gn_groups=16),
            GnSiLUConv1D(d_list[0], d_out, k=3, s=1, p=1, gn_groups=16),
            Transpose(1, 2)  # (B,L, d_out)
        )

    def forward(self, x, state_features):
        # During eval, the embed is computed only once
        # so it will be passed as a parameter
        # if self.training:

        x = self.stem(x)  # (B, L, d_in) -> (B, d0, L)

        x_list = []
        new_state_features = []
        for block in self.down_blocks:
            x, h = block(x, state_features.pop(0))
            x_list.append(x)
            new_state_features.append(h)

        x, h = self.mid_block(x, state_features.pop(0))  # (B, d-1, l)
        new_state_features.append(h)

        for i, block in enumerate(self.up_blocks):
            x = self.merge_blocks[i](torch.cat([x, x_list.pop()], dim=1))
            x, h = block(x, state_features.pop(0))
            new_state_features.append(h)

        return self.head(x), new_state_features

    def getStateShapes(self, input_len: int):
        L = input_len
        down_state_shapes = [(self.d_state, L // (2 ** i)) for i in range(self.n_stages)]
        mid_state_shape = (self.d_state, L // (2 ** self.n_stages))
        up_state_shapes = list(reversed([(self.d_state, L // (2 ** i)) for i in range(1, self.n_stages + 1)]))
        return down_state_shapes + [mid_state_shape] + up_state_shapes