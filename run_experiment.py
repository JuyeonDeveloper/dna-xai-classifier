import os
import json
from config import Config
from train import train  # train()이 함수 형태로 정의되어 있어야 함
from pathlib import Path

# 실험 결과 저장 경로
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# 실험 조합 목록
experiments = [
    {"kmer": 5, "embed_dim": 128, "num_layers": 2},
    {"kmer": 6, "embed_dim": 128, "num_layers": 2},
    {"kmer": 6, "embed_dim": 256, "num_layers": 3},
    {"kmer": 7, "embed_dim": 256, "num_layers": 4}
]

# 실행
for i, exp in enumerate(experiments):
    print(f"\n[Experiment {i+1}/{len(experiments)}] - {exp}")

    # 실험마다 고유한 모델/결과 파일 지정
    Config.kmer = exp["kmer"]
    Config.embed_dim = exp["embed_dim"]
    Config.num_layers = exp["num_layers"]
    Config.model_save_path = Path(f"models/checkpoint_k{exp['kmer']}_d{exp['embed_dim']}_l{exp['num_layers']}.pt")

    # Config와 결과 로그 저장용
    result_record = {
        "experiment": exp,
        "model_path": str(Config.model_save_path),
    }

    try:
        metrics = train(return_metrics=True)  # train.py에서 return_metrics=True일 때 평가값 반환하도록 되어 있어야 함
        result_record.update(metrics)
    except Exception as e:
        print(f"Experiment failed: {e}")
        result_record["error"] = str(e)

    # 결과 저장
    with open(results_dir / f"result_exp{i+1}.json", "w") as f:
        json.dump(result_record, f, indent=2)

    print(f"Finished Experiment {i+1}")
