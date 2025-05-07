from JimmyTorch.Models import *

class MixedCondEmbedder(nn.Module):
    def __init__(self, ddm_T: int, t_dim: int, d_embed: int):
        super().__init__()
        self.non_seq_embedders = nn.ModuleDict({
            "ddm_t": nn.Embedding(ddm_T, t_dim)
        })

        self.seq_embedders = nn.ModuleDict()

        self.d_embed = d_embed
        self.total_non_seq_dim = t_dim
        self.total_seq_dim = 0

        self.aggregators = nn.ModuleDict({
            "non_sequential": FCLayers([t_dim, t_dim, d_embed], act=nn.SiLU(inplace=True)),
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
        d_out = self.d_embed * len(self.non_seq_embedders)
        self.aggregators["non_sequential"] = nn.Sequential(FCLayers(
            [self.total_non_seq_dim, self.total_non_seq_dim, d_out, d_out], 
            act=nn.SiLU(inplace=True)
            ), nn.Unflatten(-1, (len(self.non_seq_embedders), -1)))
        
        self.aggregators["sequential"] = FCLayers(
            [self.total_seq_dim, self.total_seq_dim, self.d_embed],
            act=nn.SiLU(inplace=True)
        )

    def forward(self, ddm_t, **kw_cond):
        non_seq_embed_list = [self.non_seq_embedders["ddm_t"](ddm_t)]
        seq_embed_list = []
        for key, data in kw_cond.items():
            if key in self.non_seq_embedders:
                non_seq_embed_list.append(self.non_seq_embedders[key](data))
            elif key in self.seq_embedders:
                seq_embed_list.append(self.seq_embedders[key](data))
            else:
                raise KeyError(f"Condition of name {key} is not registered in {self.__class__.__name__}")
        
        non_seq_embed = self.aggregators["non_sequential"](torch.cat(non_seq_embed_list, dim=-1))

        if self.total_seq_dim == 0:
            return non_seq_embed

        non_seq_embed = torch.sum(non_seq_embed, dim=-1, keepdim=True)
        seq_embed = self.aggregators["sequential"](torch.cat(seq_embed_list, dim=-1))
        return seq_embed + non_seq_embed