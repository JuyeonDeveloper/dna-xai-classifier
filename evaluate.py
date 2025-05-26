from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import List, Dict

def evaluate_classification(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0)
    }
    return metrics


def hierarchical_accuracy(y_true_list, y_pred_list, levels=None):
    """
    각 샘플의 taxonomy 계층별 예측 결과와 정답을 비교하여
    얼마나 위 계층까지 일치하는지를 평균으로 계산.

    y_true_list, y_pred_list: List[Dict[str, int]]
    levels: 비교할 계층 리스트 (ex. ['domain', ..., 'species'])

    Returns: 평균 계층 일치율 (0.0 ~ 1.0)
    """
    if levels is None:
        levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    total = 0
    correct = 0

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        matched = 0
        for lvl in levels:
            if y_true.get(lvl) == y_pred.get(lvl):
                matched += 1
            else:
                break  # 상위 계층부터 비교하므로 처음 틀리면 중단
        correct += matched
        total += len(levels)

    return correct / total if total > 0 else 0.0
