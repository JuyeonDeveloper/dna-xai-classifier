from pathlib import Path

class Config:
    # 데이터 경로
    raw_data_path = Path("data/SILVA_138.2_SSURef_NR99_tax_silva.fasta")
    parsed_csv_path = Path("data/silva_parsed.csv")
    kmer_processed_path = Path("data/silva_kmer.csv")
    label_path = Path("data/label_encoder.json")

    # K-mer 설정
    kmer = 6
    vocab_path = Path("data/kmer_vocab.json")
    max_seq_len = 512

    # 모델 설정
    embed_dim = 128
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    diffusion_weight = 0.5  # 예: classifier loss와 diffusion loss를 1:0.5 비율로 합산

    # 학습 설정
    batch_size = 32
    num_epochs = 10
    learning_rate = 3e-4
    weight_decay = 1e-5

    # 출력 경로
    model_save_path = Path("checkpoints/model.pt")
    log_dir = Path("logs/")

    # 기타
    seed = 42

