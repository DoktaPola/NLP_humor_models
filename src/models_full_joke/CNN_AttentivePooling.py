import torch.nn as nn
import src.schema as S
import torch


class AttentivePooling(nn.Module):
    def __init__(self, dim=-1, input_size=128, hidden_size=32):
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
    def __init__(self, n_tokens, hid_size=8):
        super().__init__()
        self.embedder = nn.Embedding(n_tokens, hid_size)

        self.joke_encoder = nn.Sequential(
            nn.Conv1d(hid_size, hid_size * 2, kernel_size=3),
            nn.Tanh(),
            AttentivePooling(input_size=hid_size * 2),
        )

        self.joke_encoder_2 = nn.Sequential(
            nn.Conv1d(hid_size, hid_size * 2, kernel_size=3),
            nn.ELU(),
            AttentivePooling(input_size=hid_size * 2),
        )

        self.joke_encoder_3 = nn.Sequential(
            nn.Conv1d(hid_size, hid_size, kernel_size=5),
            nn.ELU(),
            AttentivePooling(input_size=hid_size),
        )

        self.joke_encoder_4 = nn.Sequential(
            nn.Conv1d(hid_size, hid_size, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            AttentivePooling(input_size=hid_size),
        )

        self.final_predictor = nn.Sequential(
            nn.Linear(hid_size * 6, hid_size),
            nn.BatchNorm1d(hid_size),
            nn.LeakyReLU(),
            nn.Linear(hid_size, 5),
            nn.Softmax(dim=-1)
        )

    def forward(self, batch):
        joke_embeddings = self.embedder(batch[S.JOKE]).permute(0, 2, 1)
        joke_features_1 = self.joke_encoder(joke_embeddings).squeeze()
        joke_features_2 = self.joke_encoder_2(joke_embeddings).squeeze()
        joke_features_3 = self.joke_encoder_3(joke_embeddings).squeeze()
        joke_features_4 = self.joke_encoder_4(joke_embeddings).squeeze()
        joke_features = torch.cat([joke_features_1, joke_features_2, joke_features_3, joke_features_4], dim=1)

        return self.final_predictor(joke_features).squeeze()