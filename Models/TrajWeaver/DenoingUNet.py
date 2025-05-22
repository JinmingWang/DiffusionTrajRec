from JimmyTorch.Models import *
    
class UNetBlock(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_state: int,
                 d_hidden: int,
                 d_time: int,
                 d_out: int,
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

        d_comb = d_in + d_state

        self.t_proj = nn.Linear(d_time, d_comb)

        self.se_res = nn.Sequential(
            GnSiLUConv1D(d_comb, d_hidden, 3, 1, 1, gn_groups=16),
            SELayer1D(d_hidden, 4),
            GnSiLUConv1D(d_hidden, d_comb, 3, 1, 1, gn_groups=16),
        )

        self.transformer = nn.TransformerEncoderLayer(d_comb, n_heads, d_hidden, dropout=0.0,
                                                      activation=nn.SiLU(inplace=True), batch_first=True,
                                                      norm_first=True)

        self.state_proj = GnSiLUConv1D(d_comb, d_state, 3, 1, 1, gn_groups=16)

        self.out_proj = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest") if scaling == "up" else nn.Identity(),
            GnSiLUConv1D(d_comb, d_out, 3, 1 + int(scaling == "down"), 1, gn_groups=16),
        )

    def forward(self, x, state_feature, t):
        """
        
        :param x: the input hidden features
        :param f: the state feature for this block
        :param cond: the conditional features
        """
        x = torch.cat([x, state_feature], dim=1) + self.t_proj(t).unsqueeze(2)
        x = x + self.se_res(x)
        x = self.transformer(x.transpose(1, 2)).transpose(1, 2)  # (B, L, d_in + d_state)
        return self.out_proj(x), self.state_proj(x)


class DenoisingUNet(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 d_state: int,
                 ddm_T: int,
                 d_time: int,
                 n_heads: int,
                 d_list: list[int],
                 ):
        super().__init__()

        self.t_embed = nn.Sequential(
            nn.Embedding(ddm_T, d_time),
            FCLayers([d_time, d_time, d_time], act=nn.SiLU(inplace=True), final_act=nn.SiLU(inplace=True)),
        )

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
                                              d_time=d_time, d_out=d_list[i + 1], n_heads=n_heads,
                                              scaling="down"))

        self.mid_block = UNetBlock(d_list[-1], d_state, d_list[-1] * 2, d_time, d_list[-1], n_heads=n_heads, scaling="same")

        self.merge_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(self.n_stages, 0, -1):
            self.merge_blocks.append(nn.Conv1d(d_list[i] * 2, d_list[i], 3, 1, 1))
            self.up_blocks.append(UNetBlock(d_in=d_list[i], d_state=d_state, d_hidden=d_list[i] * 2,
                                            d_time=d_time, d_out=d_list[i - 1], n_heads=n_heads,
                                            scaling="up"))

        self.head = nn.Sequential(
            GnSiLUConv1D(d_list[0], d_list[0], k=3, s=1, p=1, gn_groups=16),
            GnSiLUConv1D(d_list[0], d_out, k=3, s=1, p=1, gn_groups=16),
            Transpose(1, 2)  # (B,L, d_out)
        )

    def forward(self, x, ddm_t, state_features):
        # During eval, the embed is computed only once
        # so it will be passed as a parameter
        # if self.training:

        x = self.stem(x)  # (B, L, d_in) -> (B, d0, L)

        t = self.t_embed(ddm_t)

        x_list = []
        new_state_features = []
        for block in self.down_blocks:
            x, h = block(x, state_features.pop(0), t)
            x_list.append(x)
            new_state_features.append(h)

        x, h = self.mid_block(x, state_features.pop(0), t)  # (B, d-1, l)
        new_state_features.append(h)

        for i, block in enumerate(self.up_blocks):
            x = self.merge_blocks[i](torch.cat([x, x_list.pop()], dim=1))
            x, h = block(x, state_features.pop(0), t)
            new_state_features.append(h)

        return self.head(x), new_state_features

    def getStateShapes(self, input_len: int):
        L = input_len
        down_state_shapes = [(self.d_state, L // (2 ** i)) for i in range(self.n_stages)]
        mid_state_shape = (self.d_state, L // (2 ** self.n_stages))
        up_state_shapes = list(reversed([(self.d_state, L // (2 ** i)) for i in range(1, self.n_stages + 1)]))
        return down_state_shapes + [mid_state_shape] + up_state_shapes