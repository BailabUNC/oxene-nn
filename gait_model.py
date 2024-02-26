import torch
import torch.nn as nn

class GaitGRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int, output_dim: int,
                 reduced_chnl: int, kernel_size: int, stride: int):
        super().__init__()

        # channel weighting layer
        self.channel_weights = nn.Linear(input_dim, reduced_chnl)

        # sequence length reduction layer
        self.seq_reduction_conv = nn.Conv1d(reduced_chnl, reduced_chnl,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=1)

        self.gru = nn.GRU(reduced_chnl, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1) # permute to [batch, seq_len, num_channels]

        batch_size, seq_len, num_channels = x.shape
        x = x.reshape([batch_size * seq_len, num_channels])
        x = self.channel_weights(x)
        x = x.reshape([batch_size, -1, seq_len])
        x = self.seq_reduction_conv(x)
        x = x.permute(0, 2, 1) # permute back to (batch, seq_len, num_channels) for rnn

        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        out = self.softmax(x)
        return out