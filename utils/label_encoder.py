import json
from pathlib import Path
from collections import defaultdict

class LabelEncoder:
    def __init__(self):
        self.levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        self.label2idx = {lvl: {} for lvl in self.levels}
        self.idx2label = {lvl: {} for lvl in self.levels}

    def build(self, taxonomy_list):
        for tax_str in taxonomy_list:
            parts = [x.strip() for x in tax_str.split(';')]
            for i, lvl in enumerate(self.levels):
                if i < len(parts):
                    label = parts[i]
                    if label not in self.label2idx[lvl]:
                        idx = len(self.label2idx[lvl])
                        self.label2idx[lvl][label] = idx
                        self.idx2label[lvl][idx] = label

    def encode_all(self, tax_str):
        """ taxonomy 문자열 → 계층별 int label 딕셔너리 """
        parts = [x.strip() for x in tax_str.split(';')]
        result = {}
        for i, lvl in enumerate(self.levels):
            if i < len(parts):
                label = parts[i]
                result[lvl] = self.label2idx[lvl].get(label, -1)
        return result

    def decode(self, level, index):
        return self.idx2label[level].get(index, '[UNK]')

    def save(self, path: Path):
        obj = {lvl: self.label2idx[lvl] for lvl in self.levels}
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    def load(self, path: Path):
        with open(path, 'r') as f:
            obj = json.load(f)
        self.label2idx = {}
        self.idx2label = {}
        for lvl in self.levels:
            self.label2idx[lvl] = obj.get(lvl, {})
            self.idx2label[lvl] = {int(v): k for k, v in self.label2idx[lvl].items()}

    @property
    def num_classes(self):
        return {lvl: len(self.label2idx[lvl]) for lvl in self.levels}
