import torch
import torch.nn as nn
import torch.nn.functional as F


class HANClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(HANClassifier, self).__init__()

        self.word_attention = nn.Linear(embed_dim, embed_dim)
        self.word_context_vector = nn.Linear(embed_dim, 1, bias=False)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, encoder_outputs, attention_mask=None):
        # encoder_outputs: [batch_size, seq_len, embed_dim]
        u = torch.tanh(self.word_attention(encoder_outputs))  # [batch_size, seq_len, embed_dim]
        attn_scores = self.word_context_vector(u).squeeze(-1)  # [batch_size, seq_len]

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, seq_len]

        # Apply attention weights to encoder outputs
        attended_representation = torch.sum(encoder_outputs * attn_weights.unsqueeze(-1),
                                            dim=1)  # [batch_size, embed_dim]

        logits = self.fc(attended_representation)  # [batch_size, num_classes]
        return {
            "logits": logits,
            "attention_weights": attn_weights  # for XAI
        }


class HANPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        return: pooled vector [batch, embed_dim], attention weights [batch, seq_len]
        """
        attn_scores = self.attention(x).squeeze(-1)                  # [batch, seq_len]
        attn_weights = torch.softmax(attn_scores, dim=1)             # [batch, seq_len]
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch, embed_dim]
        return pooled, attn_weights

