from Models.TrajWeaverOld.components import *

class TrajWeaverOld(nn.Module):
    def __init__(self,
                 init_c: int,
                 result_c: int,
                 diffusion_steps: int,
                 c_list: List[int],
                 embed_c: int,
                 expend: int,
                 dropout: float = 0.0,
                 ) -> None:
        super().__init__()

        self.c_list = c_list
        self.stages = len(c_list) - 1
        self.embed_c = embed_c
        self.state_shapes = []

        # This block adds positional encoding and trajectory length encoding to the input trajectory
        self.stem = nn.Conv1d(init_c, c_list[0], 5, 1, 2)

        # This obtains the time embedding of the input trajectory (but not yet added)
        self.time_embedder = NumberEmbedder(max_num=diffusion_steps, hidden_dim=256, embed_dim=embed_c)

        # Create Encoder (Down sampling) Blocks for UNet
        in_c = c_list[:-1]
        out_c = c_list[1:]
        length = 512
        self.down_blocks = nn.ModuleList()
        for i in range(self.stages):
            self.state_shapes.append((in_c[i], length))
            if in_c[i] != out_c[i]:
                length //= 2
                resize = "down"
            else:
                resize = "same"
            self.down_blocks.append(TW8_Res(in_c[i], out_c[i], in_c[i], embed_c, expend, resize))

        # Create Middle Attention Block for UNet
        self.mid_attn_block = MidAttnBlock(c_list[-1], c_list[-1], expend, dropout)

        # Create Decoder (Up sampling) Blocks for UNet
        self.up_blocks = nn.ModuleList()
        # reverse the channel schedule
        in_c = c_list[-1:0:-1]
        out_c = c_list[-2::-1]
        for i in range(self.stages):
            self.state_shapes.append((in_c[i], length))
            if in_c[i] != out_c[i]:
                length *= 2
                resize = "up"
            else:
                resize = "same"
            self.up_blocks.append(TW8_Res(in_c[i] * 2, out_c[i], in_c[i], embed_c, expend//2, resize))

        self.head = nn.Conv1d(c_list[0], result_c, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


    def forward(self, x: torch.Tensor, diffusion_t: torch.Tensor, s_list: List[Tensor]):
        # Embeddings
        t_embed = self.time_embedder(diffusion_t).unsqueeze(-1)
        x = self.stem(x.transpose(-1, -2))

        # Encoder
        states = []
        x_list = []
        for di, down_stage in enumerate(self.down_blocks):
            x, s = down_stage(x, s_list[len(states)], t_embed)
            x_list.append(x)
            states.append(s)

        # Middle Attention Block
        x = self.mid_attn_block(x)  # (B, C', L//2**i)

        # Decode
        for i, up_stage in enumerate(self.up_blocks):
            x, s = up_stage(torch.cat([x, x_list[-1 - i]], dim=1), s_list[len(states)], t_embed)
            states.append(s)

        return self.head(x).transpose(-1, -2), states

    def getStateShapes(self):
        return self.state_shapes