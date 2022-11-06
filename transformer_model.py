import math
import torch
import os

class Embedder(torch.nn.Module):

    def __init__(self, vocab_sz: int, hidden_dim: int):

        super().__init__()

        self._hidden_dim = hidden_dim

        self._model = torch.nn.Embedding(vocab_sz, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        return self._model(x)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_seq_len: int = 512):

        super().__init__()

        self._dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))

        pos_enc = torch.zeros(1, max_seq_len, hidden_dim)

        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)

        # register_buffer, not part of state dict, is not going to be exposed to opti,
        # but be part of transfer to device etc..
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        x = x + self.pos_enc[:, :x.size(1), :].repeat(x.size(0), 1, 1)

        x = self._dropout(x)

        return x


class LayerNorm(torch.nn.Module):

    def __init__(self, hidden_dim: int, epsilon: float = 1e-6):

        super().__init__()

        self._hidden_dim = hidden_dim

        self._alpha = torch.nn.Parameter(torch.ones(self._hidden_dim))

        self._beta = torch.nn.Parameter(torch.zeros(self._hidden_dim))

        self._epsilon = epsilon

    def forward(self, x: torch.Tensor):

        sigma = (x.std(dim=-1, keepdim=True) + self._epsilon)

        offset = x - x.mean(dim=-1, keepdim=True)

        y = self._alpha * ((offset / sigma) + self._beta)

        return y


class Embeddings(torch.nn.Module):

    def __init__(self, vocab_sz: int, hidden_dim: int, dropout: float = 0.1, max_seq_len: int = 512):

        super().__init__()

        self._embedder = Embedder(vocab_sz=vocab_sz, hidden_dim=hidden_dim)

        self._pos_encoder = PositionalEncoding(
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self._dropout = torch.nn.Dropout(p=dropout)

        self._layer_norm = LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):

        y = self._embedder(x)
        y = self._pos_encoder(y)
        y = self._dropout(y)
        y = self._layer_norm(y)

        return y


class TimeEmbeddings(torch.nn.Module):

    def __init__(self, timesteps: int, hidden_dim: int, dropout: float = 0.1):

        super().__init__()

        self._embedder = Embedder(vocab_sz=timesteps, hidden_dim=hidden_dim)

        self._dropout = torch.nn.Dropout(p=dropout)

        self._layer_norm = LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):

        y = self._embedder(x)
        y = self._dropout(y)
        y = self._layer_norm(y)

        return y



class TransformerModel(torch.nn.Module):

    def __init__(
        self,
        nlayers: int,
        decoder: torch.nn.Module = None,
        hidden_dim: int = 768,
        timesteps: int = 1000,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):

        super().__init__()

        self._pos_encoder = PositionalEncoding(
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self._time_embedder = TimeEmbeddings(
            timesteps=timesteps,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self._encoder_layer = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(nlayers)
            ]
        )

        self._decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor
    ):

        src_key_padding_mask = (~attention_mask.bool())

        y = self._pos_encoder(x)

        # unsqueeze to make it work with broadcasting
        yt = self._time_embedder(t).unsqueeze(1)

        for layer in self._encoder_layer:
            y = layer(src=y, src_key_padding_mask=src_key_padding_mask) + yt

        y_decoded = None

        if self._decoder is not None:

            y_decoded = y

            for layer in self._decoder:
                y_decoded = layer(y_decoded)

        return y, y_decoded

    def save(self, folder: str, name: str):

        if not os.path.exists(folder):
            os.makedirs(folder)

        path = os.path.join(folder, name)

        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)

        return path

    def load(self, folder: str, name: str):

        path = os.path.join(folder, name)
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])