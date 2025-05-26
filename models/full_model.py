import torch
import torch.nn as nn
from models.transformer_encoder import TransformerEncoder
from models.han_classifier import HANPooling
from models.diffusion_module import SimpleDiffusion
from config import Config  # num_classes_per_level 딕셔너리 포함

class FullModel(nn.Module):
    def __init__(self, vocab_size, num_classes_per_level, embed_dim, num_heads, num_layers, dropout, max_len):
        super().__init__()
        self.levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        self.encoder = TransformerEncoder(vocab_size, embed_dim, num_heads, num_layers, dropout, max_len)
        self.diffusion = SimpleDiffusion(dim=embed_dim)
        self.pooling = HANPooling(embed_dim)  # attention-based pooling only
        self.classifier_heads = nn.ModuleDict({
            lvl: nn.Linear(embed_dim, num_classes_per_level[lvl])
            for lvl in self.levels
        })

    def forward(self, input_ids):
        x = self.encoder(input_ids)  # [B, L, D]
        diffusion_loss = self.diffusion.compute_loss(x)
        t = torch.randint(0, self.diffusion.timesteps, (x.size(0),), device=x.device)
        x_denoised = self.diffusion.forward(x, t)

        pooled, attention = self.pooling(x_denoised)  # [B, D], [B, L]

        logits_dict = {
            lvl: self.classifier_heads[lvl](pooled) for lvl in self.levels
        }

        return {
            'logits': logits_dict,
            'attention': attention,
            'diffusion_loss': diffusion_loss
        }
