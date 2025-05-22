from JimmyTorch.Models import *

class MixedCondEmbedder(nn.Module):
    def __init__(self, l_embed: int, d_embed: int):
        super().__init__()
        self.non_seq_embedders = nn.ModuleDict()

        self.seq_embedders = nn.ModuleDict()

        self.l_embed = l_embed
        self.d_embed = d_embed
        self.total_non_seq_dim = 0
        self.total_seq_dim = 0

        self.aggregators = nn.ModuleDict({
            "non_sequential": nn.Identity(),
            "sequential": nn.Identity()
        })

    def addCategorical(self, name: str, n_embed: int, d_hidden: int):
        self.non_seq_embedders[name] = nn.Embedding(n_embed, d_hidden)
        self.total_non_seq_dim += d_hidden
        self.__updateAggregator()

    def addVector(self, name: str, dim: int, d_hidden: int):
        self.non_seq_embedders[name] = nn.Linear(dim, d_hidden)
        self.total_non_seq_dim += d_hidden
        self.__updateAggregator()

    def addSequence(self, name: str, dim: int, d_hidden: int, use_conv: bool = True):
        kernel_size = 3 if use_conv else 1
        padding = 1 if use_conv else 0
        self.seq_embedders[name] = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(dim, d_hidden, kernel_size, 1, padding),
            nn.SiLU(inplace=True),
            nn.Conv1d(d_hidden, d_hidden, kernel_size, 1, padding),
            Transpose(1, 2),
        )
        self.total_seq_dim += d_hidden
        self.__updateAggregator()

    def __updateAggregator(self):
        d_out = self.d_embed * self.l_embed
        self.aggregators["non_sequential"] = nn.Sequential(FCLayers(
            [self.total_non_seq_dim, self.total_non_seq_dim, d_out, d_out], 
            act=nn.SiLU(inplace=True)
            ), nn.Unflatten(-1, (self.l_embed, -1)))
        
        self.aggregators["sequential"] = FCLayers(
            [self.total_seq_dim, self.total_seq_dim, self.d_embed],
            act=nn.SiLU(inplace=True)
        )

    def forward(self, **kw_cond):
        non_seq_embed_list = []
        seq_embed_list = []

        for key in self.non_seq_embedders.keys():
            value = kw_cond[key]
            non_seq_embed_list.append(self.non_seq_embedders[key](value))

        for key in self.seq_embedders.keys():
            seq_embed_list.append(self.seq_embedders[key](non_seq_embed_list[-1]))
        
        non_seq_embed = self.aggregators["non_sequential"](torch.cat(non_seq_embed_list, dim=-1))

        if self.total_seq_dim == 0:
            return non_seq_embed

        non_seq_embed = torch.sum(non_seq_embed, dim=-1, keepdim=True)
        seq_embed = self.aggregators["sequential"](torch.cat(seq_embed_list, dim=-1))
        return seq_embed + non_seq_embed