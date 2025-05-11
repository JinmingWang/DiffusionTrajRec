from JimmyTorch.Models import *
    
class UNetBlock(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_state: int,
                 d_hidden: int, 
                 d_out: int,
                 scaling: Literal["down", "same", "up"]):
        """
        :param d_in: input dim
        :param d_state: the dim of state feature
        :param d_hidden: the hidden dimension
        :param d_out: the output dimension
        :param scaling: how to scale the output
        """
        super().__init__()

        self.in_proj = Conv1DGnSiLU(d_in + d_state, d_hidden, 1, 1, 0, gn_groups=8)

        self.res = nn.Sequential(
            Conv1DGnSiLU(d_hidden, d_hidden, 3, 1, 1, gn_groups=8),
            Conv1DGnSiLU(d_hidden, d_hidden, 3, 1, 1, gn_groups=8),
            nn.Conv1d(d_hidden, d_hidden, 3, 1, 1)
        )

        self.state_proj = GnSiLUConv1D(d_hidden, d_state, 3, 1, 1, gn_groups=8)

        self.out_proj = nn.Sequential(
            nn.Upsample(scale_factor=2) if scaling == "up" else nn.Identity(),
            GnSiLUConv1D(d_hidden, d_out, 3, 1 + int(scaling == "down"), 1, gn_groups=8),
        )

    def forward(self, x, state_feature):
        """
        
        :param x: the input hidden features
        :param f: the state feature for this block
        :param cond: the conditional features
        """
        x = self.in_proj(torch.cat([x, state_feature], dim=1))
        x = x + self.res(x)
        return self.out_proj(x), self.state_proj(x)


class DenoisingUNet(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 d_state: int,
                 d_list: list[int],
                 ):
        super().__init__()

        self.d_list = d_list
        self.d_state = d_state
        self.n_stages = len(d_list) - 1

        self.stem = nn.Sequential(  # (B, L, d_in)
            Transpose(1, 2),
            Conv1DGnSiLU(d_in, d_list[0], k=3, s=1, p=1, gn_groups=8),
            nn.Conv1d(d_list[0], d_list[0], 3, 1, 1)
        )

        self.down_blocks = nn.ModuleList()
        for i in range(self.n_stages):
            self.down_blocks.append(UNetBlock(d_list[i], d_state, d_list[i] * 2, d_list[i + 1], scaling="down"))

        self.mid_block = UNetBlock(d_list[-1], d_state, d_list[-1] * 2, d_list[-1], scaling="same")

        self.merge_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(self.n_stages, 0, -1):
            self.merge_blocks.append(
                GnSiLUConv1D(d_list[i] * 2, d_list[i], 3, 1, 1)
            )

            self.up_blocks.append(UNetBlock(d_list[i], d_state, d_list[i] * 2, d_list[i - 1], scaling="up"))

        self.head = nn.Sequential(
            GnSiLUConv1D(d_list[0], d_out, k=3, s=1, p=1),
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