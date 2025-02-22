import torch
from torch import nn

class LSTM_model(nn.Module):
    def __init__(self, n_features, n_hidden, n_veh, n_out, n_layers, device):
        super(LSTM_model, self).__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_out = n_out
        self.n_veh = n_veh
        self.n_hidden_fc = 256

        self.lstm = nn.LSTM(
          input_size=n_features,
          hidden_size=n_hidden,
          num_layers=n_layers,
          dropout=0,
          batch_first=True
        )

        self.linear = nn.Linear(in_features=self.n_hidden, out_features=self.n_out*self.n_veh)
        self.activation = nn.ReLU()

        self.device = device

    def forward(self, sequences):
        x = sequences.permute(0,2,1)
        h0 = torch.zeros(self.n_layers, x.shape[0], self.n_hidden).to(self.device)
        c0 = torch.zeros(self.n_layers, x.shape[0], self.n_hidden).to(self.device)
        lstm_out, (h_out, c0) = self.lstm(x, (h0, c0))
        h_out = h_out[0].view(-1, self.n_hidden)
        y_pred = self.linear(h_out)
        y_pred = self.activation(y_pred)
        y_pred = y_pred.view(-1, self.n_veh, self.n_out)
        return y_pred
    