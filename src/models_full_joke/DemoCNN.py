import torch.nn as nn
import src.schema as S
import torch


class AttentivePooling(nn.Module):
    def __init__(self, dim=-1, input_size=64, hidden_size=128):
        super(self.__class__, self).__init__()
        self.dim = dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nn_attn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return (x * torch.transpose(self.nn_attn(torch.transpose(x, 2, 1)), 1, 2).softmax(dim=self.dim)).sum(
            dim=self.dim)


class JokeRankPredictor(nn.Module):
    def __init__(self, n_tokens, hid_size=64):
        super().__init__()
        self.embedder = nn.Embedding(n_tokens, hid_size)

        self.joke_encoder = nn.Sequential(
            nn.Conv1d(hid_size, hid_size * 2, kernel_size=2),
            nn.Tanh(),
            nn.Conv1d(hid_size * 2, hid_size * 2, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv1d(hid_size * 2, hid_size, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Dropout(p=0.25),
            nn.LeakyReLU(),
            AttentivePooling(input_size=hid_size)
        )

        self.final_predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.BatchNorm1d(hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 5),
            nn.Softmax(dim=-1)
        )

    def forward(self, batch):
        joke_embeddings = self.embedder(batch[S.JOKE]).permute(0, 2, 1)
        joke_features = self.joke_encoder(joke_embeddings).squeeze()

        return self.final_predictor(joke_features).squeeze()