import torch.nn as nn
import src.schema as S


class GlobalMaxPooling(nn.Module):
    def __init__(self, dim=-1):
        super(self.__class__, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim)[0]


class CNNGlobalMaxPooling(nn.Module):
    def __init__(self, n_tokens, hid_size=256):
        super().__init__()
        self.embedder = nn.Embedding(n_tokens, hid_size)

        self.joke_encoder = nn.Sequential(
            nn.Conv1d(hid_size, 128, kernel_size=3),
            nn.Sigmoid(),
            GlobalMaxPooling(),
        )

        self.final_predictor = nn.Sequential(
            nn.Linear(128, 50),
            nn.Dropout(p=0.5),
            nn.Sigmoid(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 5),
            nn.Softmax(dim=-1)
        )

    def forward(self, batch):
        joke_embeddings = self.embedder(batch[S.JOKE]).permute(0, 2, 1)
        joke_features = self.joke_encoder(joke_embeddings).squeeze()

        return self.final_predictor(joke_features).squeeze()
