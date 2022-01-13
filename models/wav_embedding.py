import torch
import torch.nn as nn
import logging


class EmbeddingBlock(nn.Module):
    def __init__(self, index, kernel_size, stride, glu=True, res=True):
        super(EmbeddingBlock, self).__init__()
        self.conv1D_1 = nn.Conv1d(in_channels=4 ** index, out_channels=4 ** (index + 1), kernel_size=kernel_size,
                                  stride=stride, padding=int(stride / 2))
        self.act_1 = nn.ReLU()
        if glu:
            self.activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            self.activation = nn.ReLU()
            ch_scale = 1
        self.conv1D_2 = nn.Conv1d(in_channels=4 ** (index + 1), out_channels=ch_scale * (4 ** (index + 1)),
                                  kernel_size=1,
                                  stride=1)
        self.res = res

    def forward(self, x):
        res = self.act_1(self.conv1D_1(x))
        # logging.debug(f"res {res.shape}")
        output = self.conv1D_2(res)
        # logging.debug(f"output {output.shape}")
        if self.res:
            output += res.repeat(1, 2, 1)
        output = self.activation(output)
        return output


class Embedding(nn.Module):

    def __init__(self, depth=5, kernel_size=8, stride=4, glu=True, res=True):
        super(Embedding, self).__init__()
        self.embedding = nn.ModuleList()
        for index in range(depth):
            self.embedding.append(EmbeddingBlock(index, kernel_size=kernel_size, stride=stride, glu=glu, res=res))

    def forward(self, x):
        output = x
        for embedding_layer in self.embedding:
            output = embedding_layer(output)

        return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    input = torch.randn((4, 1, 16000))
    embedding = Embedding(depth=6, kernel_size=16, stride=4)
    output = embedding(input)
    print(output.shape)
