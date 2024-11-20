import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x tem a forma [batch_size, seq_len, num_channels]
        batch_size, seq_len, num_channels = x.size()

        # Aplicar codificação posicional nas amostras temporais
        x = self.pos_encoder(x)

        # Passar pelo Transformer Encoder
        x = self.transformer_encoder(x)

        # Agregar ao longo da dimensão do comprimento da sequência
        x = x.mean(dim=1)  # Média ao longo da sequência temporal

        # Passar pela camada fully connected
        x = self.fc(x)

        return x

