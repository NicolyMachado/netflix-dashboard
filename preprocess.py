"""
preprocess.py
Leitura, limpeza e engenharia de atributos do dataset de ações da Netflix (NFLX).

Problema: prever o fechamento do PRÓXIMO pregão a partir de informações até hoje.
Estratégia: o alvo do modelo é o RETORNO do dia seguinte (variação percentual),
e o preço previsto é calculado depois: preço_previsto = fechamento * (1 + retorno_previsto).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "netflix.csv"
TEST_FRAC = 0.20  # últimos 20% do período ficam para teste

PRICE_COLS = ["open", "high", "low", "close", "adj_close"]

FEATURES = [
    "ret_lag_0", "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4",
    "ma_5_ratio", "ma_10_ratio", "ma_20_ratio",
    "vol_10", "mom_10", "hl_range", "vol_rel",
]


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Lê o CSV e padroniza os nomes das colunas (minúsculas, sem espaços)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo {path} não encontrado. Baixe o dataset do Kaggle e salve "
            "como data/netflix.csv (veja o README)."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza: datas, tipos numéricos, duplicatas, valores ausentes e inválidos."""
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("O CSV precisa ter uma coluna 'Date'.")

    # 1) Datas válidas, em ordem cronológica e sem repetição
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset="date")

    # 2) Algumas versões do dataset não trazem 'adj_close'
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # 3) Tudo numérico; preço/volume <= 0 não existe no mercado -> vira ausente
    num_cols = PRICE_COLS + ["volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].where(df[c] > 0)

    # 4) Valores ausentes: repete o último valor válido (no máximo 3 dias seguidos)
    df[num_cols] = df[num_cols].ffill(limit=3)
    df = df.dropna(subset=num_cols).reset_index(drop=True)

    # 5) Sanidade: um salto diário > 50% indica desdobramento (split) sem ajuste
    if df["close"].pct_change().abs().max() > 0.5:
        warnings.warn("Variação diária > 50% detectada: verifique splits nos dados.")
    return df


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cria os atributos (features) e o alvo. Usa SOMENTE informação até o dia t."""
    d = df[["date", "open", "high", "low", "close", "volume"]].copy()

    d["ret"] = d["close"].pct_change()
    for k in range(5):                                   # retornos dos últimos 5 dias
        d[f"ret_lag_{k}"] = d["ret"].shift(k)

    for w in (5, 10, 20):                                # distância do preço às médias móveis
        d[f"ma_{w}"] = d["close"].rolling(w).mean()
        d[f"ma_{w}_ratio"] = d["close"] / d[f"ma_{w}"] - 1

    d["vol_10"] = d["ret"].rolling(10).std()             # volatilidade recente
    d["mom_10"] = d["close"].pct_change(10)              # momento de 10 dias
    d["hl_range"] = (d["high"] - d["low"]) / d["close"]  # amplitude do dia
    d["vol_rel"] = np.log(d["volume"] / d["volume"].rolling(20).mean())  # volume relativo

    # Alvo: fechamento e retorno do pregão SEGUINTE
    d["next_close"] = d["close"].shift(-1)
    d["target_ret"] = d["next_close"] / d["close"] - 1

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURES).reset_index(drop=True)
    return d


def split_by_date(d: pd.DataFrame, test_frac: float = TEST_FRAC):
    """Divisão temporal: treino = passado, teste = futuro (nunca embaralhar)."""
    labeled = d.dropna(subset=["target_ret"]).reset_index(drop=True)
    cut = int(len(labeled) * (1 - test_frac))
    return labeled.iloc[:cut].copy(), labeled.iloc[cut:].copy()


def get_dataset():
    """Atalho: lê, limpa e monta o dataset completo."""
    return build_dataset(clean(load_raw()))
