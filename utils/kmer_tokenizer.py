import json
from collections import Counter
from pathlib import Path

class KmerTokenizer:
    def __init__(self, k=6, vocab_path=None):
        self.k = k
        self.vocab_path = vocab_path
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.vocab = {}
        self.id2kmer = {}

        if vocab_path and Path(vocab_path).exists():
            self.load_vocab()

    def build_vocab(self, sequences, max_vocab_size=None):
        kmer_counts = Counter()
        for seq in sequences:
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i:i + self.k]
                kmer_counts[kmer] += 1

        most_common = kmer_counts.most_common(max_vocab_size)
        self.vocab = {
            self.pad_token: 0,
            self.unk_token: 1
        }
        for idx, (kmer, _) in enumerate(most_common, start=2):
            self.vocab[kmer] = idx

        self.id2kmer = {idx: kmer for kmer, idx in self.vocab.items()}

        if self.vocab_path:
            with open(self.vocab_path, 'w') as f:
                json.dump(self.vocab, f)

    def load_vocab(self):
        if self.vocab_path and Path(self.vocab_path).exists():
            with open(self.vocab_path, 'r') as f:
                self.vocab = json.load(f)

            self.vocab = {k: int(v) for k, v in self.vocab.items()}
            self.id2kmer = {v: k for k, v in self.vocab.items()}

            # ✅ 필수 토큰 강제 보장
            if self.pad_token not in self.vocab:
                self.vocab[self.pad_token] = 0
            if self.unk_token not in self.vocab:
                self.vocab[self.unk_token] = 1
        else:
            raise FileNotFoundError("Vocab file not found.")

    def encode(self, sequence, max_len):
        tokens = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            tokens.append(self.vocab.get(kmer, self.vocab[self.unk_token]))

        tokens = tokens[:max_len]
        tokens += [self.vocab[self.pad_token]] * (max_len - len(tokens))
        return tokens

    @property
    def vocab_size(self):
        return len(self.vocab)
