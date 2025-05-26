import torch
import argparse
import matplotlib.pyplot as plt
from models.full_model import FullModel
from utils.kmer_tokenizer import KmerTokenizer
from utils.label_encoder import LabelEncoder
from config import Config

FALLBACK_ORDER = ['species', 'genus', 'family']
THRESHOLD = 0.7  # species 예측 확률이 이보다 낮으면 fallback


def predict(sequence: str, k: int = 6, top_k: int = 10):
    tokenizer = KmerTokenizer(k=k, vocab_path=Config.vocab_path)
    label_encoder = LabelEncoder()
    label_encoder.load(Config.label_path)
    num_classes_per_level = label_encoder.num_classes

    input_ids = tokenizer.encode(sequence, max_len=Config.max_seq_len)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    model = FullModel(
        vocab_size=Config.vocab_size,
        num_classes_per_level=num_classes_per_level,
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
        logits_dict = output['logits']
        attention = output['attention'].squeeze(0).tolist()

    result = {}
    fallback_result = None

    for level in FALLBACK_ORDER:
        logits = logits_dict[level]
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        label = label_encoder.decode(level, pred_idx)

        result[level] = (label, confidence)

        if confidence >= THRESHOLD and fallback_result is None:
            fallback_result = (level, label, confidence)

    print("\n[Prediction Result with Fallback]")
    if fallback_result:
        level, label, conf = fallback_result
        print(f"Final Prediction: {level.upper()} = {label} (confidence: {conf:.4f})")
    else:
        print("Low confidence at all levels. Unable to classify reliably.")

    print("\n[All Predictions]")
    for lvl in FALLBACK_ORDER:
        label, conf = result[lvl]
        print(f"{lvl:<8s}: {label:<30s} (conf: {conf:.4f})")

    # Attention top-k
    print("\n[Top Attention k-mers]")
    kmers = [tokenizer.id2kmer.get(i, '[UNK]') for i in input_ids]
    attn_scores = list(zip(kmers, attention))
    attn_scores.sort(key=lambda x: x[1], reverse=True)
    for i, (kmer, score) in enumerate(attn_scores[:top_k]):
        print(f"{i+1}. {kmer:10s} → {score:.4f}")

    # Optional plot
    plt.figure(figsize=(10, 4))
    plt.bar([x[0] for x in attn_scores[:top_k]], [x[1] for x in attn_scores[:top_k]])
    plt.xlabel('k-mer')
    plt.ylabel('Attention Weight')
    plt.title(f'Top {top_k} Attention k-mers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("inference.py")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--kmer', type=int, default=6)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    predict(args.sequence, k=args.kmer, top_k=args.top_k)
