import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4, dropout=0.1, max_len=512):
        super(TransformerEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # [batch_size, seq_len]

        token_embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        position_embeddings = self.position_embedding(position_ids)

        embeddings = self.dropout(token_embeddings + position_embeddings)
        output = self.transformer_encoder(embeddings)  # [batch_size, seq_len, embed_dim]

        return output