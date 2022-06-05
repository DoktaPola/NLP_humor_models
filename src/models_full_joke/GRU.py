import torch.nn as nn
import src.schema as S
import torch


class GRUPredictor(nn.Module):
    def __init__(self, n_tokens, hid_size=8):
        super().__init__()
        self.embedder = nn.Embedding(n_tokens, hid_size)

        self.title_gru = nn.GRU(hid_size, hid_size, batch_first=True)
        self.dr1 = nn.Dropout(p=0.25)
        self.af1 = nn.ELU()

        self.final_predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.BatchNorm1d(hid_size),
            nn.LeakyReLU(),
            nn.Linear(hid_size, 5),
            nn.Softmax(dim=-1)
        )

    def forward(self, batch):
        title_embeddings = self.embedder(batch[S.JOKE])
        output, hidden_title = self.title_gru(title_embeddings)
        hidden_title = torch.cat([hidden_title[i, :, :] for i in range(hidden_title.shape[0])], dim=1)
        out_title = self.dr1(hidden_title)
        out_title = self.af1(out_title)

        return self.final_predictor(out_title).squeeze()