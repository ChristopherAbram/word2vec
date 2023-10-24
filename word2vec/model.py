import torch
import torch.nn as nn


class CBOWModel(nn.Module):
    """
    Implementation of Word2Vec model described in paper: https://arxiv.org/abs/1301.3781
    """

    def __init__(self, vocab_size: int, embed_size: int):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
        )
        self.linear = nn.Linear(
            in_features=embed_size,
            out_features=vocab_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_: torch.Tensor = self.embeddings(x)
        x_ = x_.mean(axis=1)
        x_ = self.linear(x_)
        return x_


class SkipGramModel(nn.Module):
    """
    Implementation of skip-gram word2vec model.
    """

    def __init__(self, vocab_size: int, embed_size: int):
        """
        Create Skip-gram network from https://arxiv.org/pdf/1310.4546.pdf.

        :param vocab_size: Size of vocabulary
        :param embed_size: Size of low-dimensional embeddings.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # Create embedding layers for input and output words
        self.in_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
        )
        self.out_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
        )

        # Init weight of embedding layers with uniform distribution U(-1, 1)
        # NOTE: by default weight are init with dist N(0, 1)
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def in_forward(self, x: torch.Tensor) -> torch.Tensor:
        out_: torch.FloatTensor = self.in_embed(x)
        return out_

    def out_forward(self, x: torch.Tensor) -> torch.Tensor:
        out_ = self.out_embed(x)
        return out_


class NegativeSamplingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, in_embed: torch.Tensor, out_embed: torch.Tensor, neg_embed: torch.Tensor) -> torch.Tensor:
        # Reshape in/out embeddings to batch of column/row vectors
        in_embed = in_embed.unsqueeze(2)
        out_embed = out_embed.unsqueeze(1)
        # Compute the first term of the loss function, i.e., log-sigmoid of dot-product of out_embed and in_embed samples
        out_loss = torch.bmm(out_embed, in_embed).sigmoid().log().squeeze()
        # Compute the second term of the loss function, i.e., log-sigmoid of dot-product of neg_embed
        # and in_embed summed across negative samples
        noise_loss_sum = torch.bmm(neg_embed.neg(), in_embed).sigmoid().log().squeeze().sum(1)
        # Final loss is a batch average of out_loss and noise_loss_sum
        return -1 * (out_loss + noise_loss_sum).mean()
