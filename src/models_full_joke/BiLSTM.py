import torch.nn as nn
import src.schema as S
import torch


class BiLSTMPredictor(nn.Module):
    def __init__(self, n_tokens, hid_size=8):
        super().__init__()
        self.embedder = nn.Embedding(n_tokens, hid_size)

        self.title_lstm = nn.LSTM(hid_size,
                                  hid_size,
                                  num_layers=2,
                                  bidirectional=True,
                                  batch_first=True)
        self.dr1 = nn.Dropout(p=0.25)
        self.af1 = nn.LeakyReLU()

        self.final_predictor = nn.Sequential(
            nn.Linear(hid_size * 4, hid_size),
            nn.BatchNorm1d(hid_size),
            nn.LeakyReLU(),
            nn.Linear(hid_size, 5),
            nn.Softmax(dim=-1)
        )

    def forward(self, batch):
        title_embeddings = self.embedder(batch[S.JOKE])
        output, (h_n_title, c_n_title) = self.title_lstm(title_embeddings)
        h_n_title = torch.cat([h_n_title[i, :, :] for i in range(h_n_title.shape[0])], dim=1)
        out_title = self.dr1(h_n_title)
        out_title = self.af1(out_title)

        return self.final_predictor(out_title).squeeze()