"""
modeling.py
Definição dos modelos, previsão e métricas (usado por train.py, evaluate.py e app.py).
"""
import warnings
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import FEATURES

# Número fixo de épocas (max_iter) funciona como regularização; o aviso de
# "não convergiu" é esperado em séries financeiras ruidosas.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

RF_NAME = "Random Forest"
MLP_NAME = "Rede Neural (MLP)"
BASE_ZERO = "Baseline: persistência"
BASE_MA5 = "Baseline: média móvel 5d"

MODEL_FILES = {
    RF_NAME: MODELS_DIR / "random_forest.joblib",
    MLP_NAME: MODELS_DIR / "mlp.joblib",
}


def make_rf():
    """Árvores de decisão em conjunto. Parâmetros conservadores para evitar overfitting."""
    return RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=20,
        n_jobs=-1, random_state=42,
    )


def make_mlp():
    """Perceptron multicamadas: padroniza as entradas e também o alvo."""
    net = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3,
            learning_rate_init=1e-3, max_iter=500, random_state=42,
        )),
    ])
    return TransformedTargetRegressor(regressor=net, transformer=StandardScaler())


MODEL_FACTORIES = {RF_NAME: make_rf, MLP_NAME: make_mlp}


def predict_returns(name, model, d):
    """Retorno previsto para o próximo pregão, para cada linha de d."""
    if name == BASE_ZERO:          # "amanhã = hoje"
        return np.zeros(len(d))
    if name == BASE_MA5:           # "amanhã = média dos últimos 5 dias"
        return (d["ma_5"] / d["close"] - 1).to_numpy()
    return model.predict(d[FEATURES])


def compute_metrics(d, ret_pred):
    """Métricas de regressão no PREÇO + acerto de direção (sobe/desce)."""
    price_pred = d["close"].to_numpy() * (1 + ret_pred)
    y = d["next_close"].to_numpy()
    mse = mean_squared_error(y, price_pred)
    nonzero = ret_pred != 0
    hit = (np.sign(ret_pred[nonzero]) == np.sign(d["target_ret"].to_numpy()[nonzero])).mean() \
        if nonzero.any() else np.nan
    return {
        "MAE": mean_absolute_error(y, price_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2 (preço)": r2_score(y, price_pred),
        "R2 (retorno)": r2_score(d["target_ret"], ret_pred),
        "Acerto de direção": hit,
    }
