import torch.nn as nn
import src.schema as S


class CNNSimple(nn.Module):
    def __init__(self, n_tokens, hid_size=8):
        super().__init__()
        self.embedder = nn.Embedding(n_tokens, hid_size)

        self.joke_encoder = nn.Sequential(
            nn.Conv1d(hid_size, hid_size, kernel_size=2),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
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
