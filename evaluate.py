"""

evaluate.py
Gera a tabela de métricas no período de TESTE (dados que o modelo nunca viu).
Uso:  python evaluate.py      (rode train.py antes)

"""
import joblib
import pandas as pd

from modeling import (BASE_MA5, BASE_ZERO, MODEL_FILES, OUTPUTS_DIR,
                      compute_metrics, predict_returns)
from preprocess import get_dataset, split_by_date


def evaluate_all(test, models):
    names = [BASE_ZERO, BASE_MA5] + list(models)
    table, preds = {}, test[["date", "close", "next_close"]].copy()
    for name in names:
        ret = predict_returns(name, models.get(name), test)
        table[name] = compute_metrics(test, ret)
        preds[name] = test["close"].to_numpy() * (1 + ret)
    return pd.DataFrame(table).T, preds


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    _, test = split_by_date(get_dataset())
    models = {n: joblib.load(p) for n, p in MODEL_FILES.items()}
    metrics, preds = evaluate_all(test, models)

    metrics.round(4).to_csv(OUTPUTS_DIR / "metricas.csv")
    preds.to_csv(OUTPUTS_DIR / "predicoes_teste.csv", index=False)
    print("\nMétricas no conjunto de teste:\n")
    print(metrics.round(4).to_string())
    print("\nArquivos gerados em outputs/: metricas.csv, predicoes_teste.csv")


if __name__ == "__main__":
    main()
