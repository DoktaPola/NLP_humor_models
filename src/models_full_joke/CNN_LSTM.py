import torch.nn as nn
import src.schema as S
import torch


class CNNLSTMPredictorV1(nn.Module):
    def __init__(self, n_tokens, hid_size=8):
        super().__init__()
        self.embedder = nn.Embedding(n_tokens, hid_size)

        self.title_conv1 = nn.Conv1d(hid_size, hid_size * 2, kernel_size=3)
        self.af11 = nn.ReLU()
        self.title_lstm = nn.LSTM(hid_size * 2, hid_size, batch_first=True)
        self.dr1 = nn.Dropout(p=0.25)
        self.af12 = nn.ReLU()

        self.final_predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.BatchNorm1d(hid_size),
            nn.LeakyReLU(),
            nn.Linear(hid_size, 5),
            nn.Softmax(dim=-1)
        )

    def forward(self, batch):
        title_embeddings = self.embedder(batch[S.JOKE]).permute(0, 2, 1)
        out1 = self.title_conv1(title_embeddings)
        out1 = self.af11(out1).permute(0, 2, 1)
        output, (h_n_title, c_n_title) = self.title_lstm(out1)
        h_n_title = torch.squeeze(h_n_title, 0)
        out_title = self.dr1(h_n_title)
        out_title = self.af12(out_title)

        return self.final_predictor(out_title).squeeze()


class CNNLSTMPredictorV2(nn.Module):
    def __init__(self, n_tokens, hid_size=8):
        super().__init__()
        self.embedder = nn.Embedding(n_tokens, hid_size)

        self.description_conv1 = nn.Conv1d(hid_size, hid_size * 2, kernel_size=2)
        self.af21 = nn.ReLU()
        self.description_conv2 = nn.Conv1d(hid_size * 2, hid_size * 2, kernel_size=3)
        self.af22 = nn.ReLU()
        self.description_conv3 = nn.Conv1d(hid_size * 2, hid_size, kernel_size=5)
        self.af23 = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.description_lstm = nn.LSTM(hid_size, hid_size, batch_first=True)
        self.dr2 = nn.Dropout(p=0.25)
        self.af24 = nn.ReLU()

        self.final_predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.BatchNorm1d(hid_size),
            nn.ELU(),
            nn.Linear(hid_size, 5),
            nn.Softmax(dim=-1)
        )

    def forward(self, batch):
        description_embeddings = self.embedder(batch[S.JOKE]).permute(0, 2, 1)
        out2 = self.description_conv1(description_embeddings)
        out2 = self.af21(out2)
        out2 = self.description_conv2(out2)
        out2 = self.af22(out2)
        out2 = self.description_conv3(out2)
        out2 = self.af23(out2)
        out2 = self.pool(out2).permute(0, 2, 1)
        output, (h_n_description, c_n_description) = self.description_lstm(out2)
        h_n_description = torch.squeeze(h_n_description, 0)
        out_description = self.dr2(h_n_description)
        out_description = self.af24(out_description)

        return self.final_predictor(out_description).squeeze()