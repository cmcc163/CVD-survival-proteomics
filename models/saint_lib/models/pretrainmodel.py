from .model import *


class sep_MLP(nn.Module):
    
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        
        y_pred = list([])

class SAINT(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=0,
            attn_dropout=0.,
            ff_dropout=0.,
            cont_embeddings='MLP',
            scalingfactor=10,
            attentiontype='col',
            final_mlp_style='common',
            y_dim=2,
            mlp_hiddle = 100,
            final_hiddle = 1000,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        
        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, mlp_hiddle, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, mlp_hiddle, self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
        elif attentiontype in ['row', 'colrow']:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype
            )

        
        self.embeds = nn.Embedding(self.total_tokens, self.dim)

        
        self.pos_encodings = nn.Embedding(self.num_categories + self.num_continuous, self.dim)

        
        
        self.mlpfory = simple_MLP([dim, final_hiddle, y_dim])


    def forward(self, x_categ, x_cont):
        
        
        
        x = self.transformer(x_categ, x_cont)

        
        
        
        cls_reps = x[:, 0, :] # shape: (Batch, dim)

        
        output = self.mlpfory(cls_reps)
        
        return output
