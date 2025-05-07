from .components import *
from .Embedder import MixedCondEmbedder

"""
Total Training Params:                                                  23.26 M 
fwd MACs:                                                               1.52 GMACs
fwd FLOPs:                                                              3.04 GFLOPS
fwd+bwd MACs:                                                           4.55 GMACs
fwd+bwd FLOPs:                                                          9.13 GFLOPS
"""
class TrajWeaver(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 d_list: list[int],
                 d_embed: int,
                 n_heads: int,
                 embedder: nn.Module,
                 dropout: float,
                 ):
        super().__init__()

        self.d_list = d_list
        self.n_stages = len(d_list) - 1

        self.embedder = embedder

        self.stem = nn.Sequential(  # (B, L, d_in)
            Transpose(1, 2),
            Conv1DGnSiLU(d_in, d_list[0], k=3, s=1, p=1, gn_groups=8),
            nn.Conv1d(d_list[0], d_list[0], 3, 1, 1)
        )

        self.down_blocks = nn.ModuleList()
        for i in range(self.n_stages):
            self.down_blocks.append(TrajWeaverBlock(
                d_list[i], d_list[i], d_embed, d_list[i] * 2, d_list[i+1], 
                n_heads, dropout, scaling="down"))
            
        self.mid_block = TrajWeaverBlock(
            d_list[-1], d_list[-1], d_embed, d_list[-1] * 2, d_list[-1],
            n_heads, dropout, scaling="same"
        )

        self.merge_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(self.n_stages, 0, -1):
            self.merge_blocks.append(
                GnSiLUConv1D(d_list[i]*2, d_list[i], 3, 1, 1)
            )

            self.up_blocks.append(TrajWeaverBlock(
                d_list[i], d_list[i], d_embed, d_list[i] * 2, d_list[i - 1],
                n_heads, dropout, scaling="up"
            ))

        self.head = nn.Sequential(
            GnSiLUConv1D(d_list[0], d_out, k=3, s=1, p=1),
            Transpose(1, 2)     # (B,L, d_out)
        )

    def forward(self, x, ddm_t, h_list, embed=None, **kw_cond):
        # During eval, the embed is computed only once
        # so it will be passed as a parameter
        # if self.training:
        embed = self.embedder(ddm_t, **kw_cond)

        x = self.stem(x)    # (B, L, d_in) -> (B, d0, L)

        x_list = []
        new_h_list = []
        for block in self.down_blocks:
            x, h = block(x, h_list.pop(0), embed)
            x_list.append(x)
            new_h_list.append(h)

        x, h = self.mid_block(x, h_list.pop(0), embed)     # (B, d-1, l)
        new_h_list.append(h)

        for i, block in enumerate(self.up_blocks):
            x = self.merge_blocks[i](torch.cat([x, x_list.pop()], dim=1))
            x, h = block(x, h_list.pop(0), embed)
            new_h_list.append(h)

        return self.head(x), new_h_list

    def getStateShapes(self, input_len: int):
        L = input_len
        down_state_shapes = [(self.d_list[i], L//(2**i)) for i in range(self.n_stages)]
        mid_state_shape = (self.d_list[-1], L//(2**self.n_stages))
        up_state_shapes = list(reversed([(self.d_list[i], L//(2**i)) for i in range(1, self.n_stages + 1)]))
        return down_state_shapes + [mid_state_shape] + up_state_shapes
        