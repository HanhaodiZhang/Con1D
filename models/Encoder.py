import logging

import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.transformer import Transformer
from models.wav_embedding import Embedding


class AuT(nn.Module):
    def __init__(self, *, sample_rate=36000, time_length=10, embedding_kernel=16, embedding_stride=8, embedding_depth=4,
                 embedding_glu=True, embedding_res=True, time_step=0.5, num_classes=527, dim=1024, depth=6, heads=16,
                 mlp_dim=1024, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super(AuT, self).__init__()

        self.num_patches = int(time_length / time_step)
        # embedding the wav
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (n v) -> (b n) 1 v', v=int(time_step*sample_rate)),
            Embedding(depth=embedding_depth, kernel_size=embedding_kernel, stride=embedding_stride, glu=embedding_glu,
                      res=embedding_res),
            Rearrange('(b n) c v -> b n (c v)', n=self.num_patches)
        )
        # transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, wav):
        x = self.to_patch_embedding(wav)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    input = torch.randn((4, 360000))
    aut = AuT()
    output = aut(input)
    print(output.shape)
