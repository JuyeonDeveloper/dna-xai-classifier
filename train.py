import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from models.full_model import FullModel
from config import Config
from tqdm import tqdm
import os
from utils.kmer_tokenizer import KmerTokenizer
from utils.label_encoder import LabelEncoder
from evaluate import evaluate_classification


class SimpleDNADataset(Dataset):
    def __init__(self, csv_path, tokenizer, label_encoder, max_len=512):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = row['sequence']
        tax_str = row['taxonomy']
        input_ids = self.tokenizer.encode(seq, self.max_len)
        labels = self.label_encoder.encode_all(tax_str)
        return torch.tensor(input_ids, dtype=torch.long), {
            level: torch.tensor(label, dtype=torch.long) for level, label in labels.items()
        }


def train():
    # ✅ 디바이스 설정 (GPU 우선)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = KmerTokenizer(k=Config.kmer, vocab_path=Config.vocab_path)
    if not Config.vocab_path.exists():
        df = pd.read_csv(Config.kmer_processed_path)
        tokenizer.build_vocab(df['sequence'].tolist())

    vocab_size = tokenizer.vocab_size

    label_encoder = LabelEncoder()
    if not Config.label_path.exists():
        df = pd.read_csv(Config.kmer_processed_path)
        label_encoder.build(df['taxonomy'].tolist())
        label_encoder.save(Config.label_path)
    else:
        label_encoder.load(Config.label_path)

    num_classes_per_level = label_encoder.num_classes

    full_dataset = SimpleDNADataset(
        csv_path=Config.kmer_processed_path,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_len=Config.max_seq_len
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)

    model = FullModel(
        vocab_size=vocab_size,
        num_classes_per_level=num_classes_per_level,
        embed_dim=Config.embed_dim,
        num_heads=Config.num_heads,
        num_layers=Config.num_layers,
        dropout=Config.dropout,
        max_len=Config.max_seq_len
    ).to(device)  # ✅ 모델 GPU로

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

    model.train()
    for epoch in range(Config.num_epochs):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}")
        for input_ids, label_dicts in loop:
            input_ids = input_ids.to(device)  # ✅ 입력 GPU로

            output = model(input_ids)
            logits_dict = output['logits']
            diffusion_loss = output['diffusion_loss']

            loss = 0
            for lvl in model.levels:
                target = label_dicts[lvl].to(device)  # ✅ 레이블 GPU로
                loss += criterion(logits_dict[lvl], target)

            loss += Config.diffusion_weight * diffusion_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

        # ✅ Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for input_ids, label_dicts in val_loader:
                input_ids = input_ids.to(device)
                output = model(input_ids)
                preds = torch.argmax(output['logits']['species'], dim=1)
                targets = label_dicts['species'].to(device)
                all_preds.extend(preds.tolist())
                all_labels.extend(targets.tolist())

        metrics = evaluate_classification(all_labels, all_preds)
        print(f"Validation Metrics (Epoch {epoch+1}):")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        model.train()

    os.makedirs(Config.model_save_path.parent, exist_ok=True)
    torch.save(model.state_dict(), Config.model_save_path)
    print("Model saved.")


if __name__ == '__main__':
    train()
