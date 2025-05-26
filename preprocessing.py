import argparse
import pandas as pd
from pathlib import Path

def parse_fasta_with_taxonomy(fasta_path):
    records = []
    with open(fasta_path, 'r') as f:
        current_id, current_tax, current_seq = None, None, []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    records.append((current_id, ''.join(current_seq), current_tax))
                header = line[1:]
                parts = header.split(maxsplit=1)
                current_id = parts[0]
                current_tax = parts[1] if len(parts) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            records.append((current_id, ''.join(current_seq), current_tax))
    return pd.DataFrame(records, columns=["id", "sequence", "taxonomy"])

def main(input_path, output_path):
    df = parse_fasta_with_taxonomy(input_path)

    # Bacteria만 필터링
    df = df[df['taxonomy'].str.startswith("Bacteria")]
    df.to_csv(output_path, index=False)
    print(f"[✓] 전처리 완료: {len(df):,}개 샘플 → {output_path}")

if __name__ == '__main__':
    print("preprocessing.py")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='FASTA 입력 경로')
    parser.add_argument('--output', type=str, required=True, help='CSV 출력 경로')
    args = parser.parse_args()

    main(Path(args.input), Path(args.output))
