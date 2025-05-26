import torch
import argparse
import matplotlib.pyplot as plt
from models.full_model import FullModel
from utils.kmer_tokenizer import KmerTokenizer
from config import Config
from utils.label_encoder import LabelEncoder


def visualize_attention(sequence: str, k: int, top_k: int = 10):
    tokenizer = KmerTokenizer(k=k, vocab_path=Config.vocab_path)
    input_ids = tokenizer.encode(sequence, max_len=Config.max_seq_len)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]

    label_encoder = LabelEncoder()
    label_encoder.load(Config.label_path)
    num_classes = label_encoder.num_classes

    model = FullModel(
        vocab_size=Config.vocab_size,
        num_classes=num_classes,
        embed_dim=Config.embed_dim,
        num_heads=Config.num_heads,
        num_layers=Config.num_layers,
        dropout=Config.dropout,
        max_len=Config.max_seq_len
    )
    model.load_state_dict(torch.load(Config.model_save_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        attention = output['attention'].squeeze(0).tolist()  # [seq_len]

    kmer_ids = input_ids
    id2kmer = tokenizer.id2kmer
    kmers = [id2kmer.get(i, '[UNK]') for i in kmer_ids]

    paired = list(zip(kmers, attention))
    paired = sorted(paired, key=lambda x: x[1], reverse=True)[:top_k]

    print("Top attention k-mers:")
    for i, (kmer, weight) in enumerate(paired):
        print(f"{i+1}. {kmer:10s} → weight: {weight:.4f}")

    # Optional: barplot
    plt.figure(figsize=(10, 4))
    plt.bar([x[0] for x in paired], [x[1] for x in paired])
    plt.xlabel('k-mer')
    plt.ylabel('Attention Weight')
    plt.title(f'Top {top_k} Attention k-mers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=str, required=True, help='DNA sequence input')
    parser.add_argument('--kmer', type=int, default=6)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    visualize_attention(args.sequence, args.kmer, args.top_k)
