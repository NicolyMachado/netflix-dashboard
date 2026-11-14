"""
train.py
Treina os modelos com o período de TREINO, faz validação cruzada temporal e salva
os modelos em arquivos .joblib.   Uso:  python train.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from modeling import MODEL_FACTORIES, MODEL_FILES, MODELS_DIR, OUTPUTS_DIR
from preprocess import FEATURES, get_dataset, split_by_date


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    d = get_dataset()
    train, test = split_by_date(d)
    X_tr, y_tr = train[FEATURES], train["target_ret"]
    print(f"Treino: {len(train)} dias ({train['date'].min().date()} a {train['date'].max().date()})")
    print(f"Teste : {len(test)} dias ({test['date'].min().date()} a {test['date'].max().date()})")

    # Validação cruzada temporal (cada fold treina no passado e valida no futuro)
    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    for name, factory in MODEL_FACTORIES.items():
        for i, (tr, va) in enumerate(tscv.split(X_tr), start=1):
            m = factory().fit(X_tr.iloc[tr], y_tr.iloc[tr])
            rmse = np.sqrt(mean_squared_error(y_tr.iloc[va], m.predict(X_tr.iloc[va])))
            rows.append({"modelo": name, "fold": i, "RMSE do retorno (validação)": rmse})
    cv = pd.DataFrame(rows)
    cv.to_csv(OUTPUTS_DIR / "validacao_cruzada.csv", index=False)
    print("\nValidação cruzada temporal (média por modelo):")
    print(cv.groupby("modelo")["RMSE do retorno (validação)"].mean().round(5).to_string())

    # Treino final (todo o período de treino) e persistência
    for name, factory in MODEL_FACTORIES.items():
        model = factory().fit(X_tr, y_tr)
        joblib.dump(model, MODEL_FILES[name])
        print(f"Modelo salvo: {MODEL_FILES[name].name}")

    meta = {
        "features": FEATURES,
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
