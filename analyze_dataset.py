import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from utils.label_encoder import LabelEncoder
from config import Config
from pathlib import Path

def analyze_dataset(csv_path):
    df = pd.read_csv(csv_path)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print(f"\n총 샘플 수: {len(df):,}")

    levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    for i, lvl in enumerate(levels):
        df[lvl] = df['taxonomy'].apply(lambda x: x.split(';')[i].strip() if len(x.split(';')) > i else '[MISSING]')

    print("\n[계층별 클래스 수]")
    for lvl in levels:
        print(f"{lvl:<8}: {df[lvl].nunique()}개")

    df['length'] = df['sequence'].str.len()
    print("\n[시퀀스 길이 통계]")
    print(df['length'].describe())

    print("\n[종별 샘플 수 (상위 15개)]")
    top_species = df['species'].value_counts().head(15)
    print(top_species)

    # 시각화 1: 시퀀스 길이 분포
    plt.figure(figsize=(8, 4))
    plt.hist(df['length'], bins=50, color='skyblue')
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Length')
    plt.ylabel('Sample Count')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(results_dir / "sequence_length_distribution.png")
    plt.close()

    # 시각화 2: 종별 샘플 수 (상위 15개)
    plt.figure(figsize=(10, 5))
    top_species.plot(kind='bar')
    plt.title('Top 15 Most Frequent Species')
    plt.xlabel('Species')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(results_dir / "top15_species_distribution.png")
    plt.close()

    print("\n[✓] 이미지가 results/ 폴더에 저장되었습니다.")

if __name__ == "__main__":
    analyze_dataset(Config.kmer_processed_path)
