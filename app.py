"""

app.py
Dashboard em Streamlit.   Uso:  streamlit run app.py

"""
import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from evaluate import evaluate_all
from modeling import MODEL_FILES, MODELS_DIR, BASE_ZERO, predict_returns
from preprocess import FEATURES, get_dataset, split_by_date

st.set_page_config(page_title="Netflix (NFLX) – Previsão", layout="wide")


@st.cache_data
def load_data():
    d = get_dataset()
    train, test = split_by_date(d)
    return d, train, test


@st.cache_resource
def load_models():
    return {n: joblib.load(p) for n, p in MODEL_FILES.items() if p.exists()}


d, train, test = load_data()
models = load_models()

st.title("📈 Netflix (NFLX) – Previsão do próximo fechamento")
st.caption("Apoio à tomada de decisão com séries temporais · UFMS Três Lagoas · Dados: Kaggle")

if not models:
    st.error("Nenhum modelo encontrado. Rode `python train.py` antes de abrir o dashboard.")
    st.stop()

# ---------------- Barra lateral ----------------
st.sidebar.header("Configurações")
model_name = st.sidebar.selectbox("Modelo", list(models))
model = models[model_name]
d_min, d_max = d["date"].min().date(), d["date"].max().date()
start_default = max(d_min, (d["date"].max() - pd.Timedelta(days=730)).date())
period = st.sidebar.date_input("Período do histórico", (start_default, d_max),
                               min_value=d_min, max_value=d_max)
mas = st.sidebar.multiselect("Médias móveis", [5, 10, 20], default=[20])
n_days = st.sidebar.slider("Dias exibidos no gráfico de teste", 30, len(test), min(120, len(test)))

# ---------------- Previsão do próximo pregão ----------------
last = d.iloc[[-1]]
ret_pred = float(predict_returns(model_name, model, last)[0])
price_now = float(last["close"].iloc[0])
price_pred = price_now * (1 + ret_pred)
test_ret_pred = predict_returns(model_name, model, test)
sd = float((test["target_ret"] - test_ret_pred).std())       # erro típico no teste
low, high = price_now * (1 + ret_pred - 1.96 * sd), price_now * (1 + ret_pred + 1.96 * sd)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Último fechamento ({last['date'].iloc[0].date()})", f"US$ {price_now:,.2f}")
c2.metric("Previsão do próximo pregão", f"US$ {price_pred:,.2f}", f"{ret_pred * 100:+.2f}%")
c3.metric("Faixa provável (95%)", f"{low:,.0f} – {high:,.0f}")
c4.metric("Direção prevista", "Alta ▲" if ret_pred > 0 else "Queda ▼")

tab1, tab2, tab3, tab4 = st.tabs(["Histórico", "Previsão vs Real (teste)", "Métricas", "Insights"])

# ---------------- Aba 1: histórico ----------------
with tab1:
    if isinstance(period, tuple) and len(period) == 2:
        h = d[(d["date"].dt.date >= period[0]) & (d["date"].dt.date <= period[1])]
    else:
        h = d
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["date"], y=h["close"], name="Fechamento", line=dict(width=2)))
    for w in mas:
        fig.add_trace(go.Scatter(x=h["date"], y=h[f"ma_{w}"], name=f"Média móvel {w}d",
                                 line=dict(width=1, dash="dot")))
    fig.update_layout(height=450, yaxis_title="US$", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig)
    st.caption(f"{len(h)} pregões no período selecionado.")

# ---------------- Aba 2: previsão vs real ----------------
with tab2:
    metrics, preds = evaluate_all(test, models)
    p = preds.tail(n_days)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=p["date"], y=p["next_close"], name="Real (pregão seguinte)"))
    fig2.add_trace(go.Scatter(x=p["date"], y=p[model_name], name=f"Previsto – {model_name}"))
    fig2.update_layout(height=450, yaxis_title="US$", margin=dict(l=10, r=10, t=30, b=10),
                       xaxis_title="Data de referência (previsão feita neste dia)")
    st.plotly_chart(fig2)
    st.caption("Período de TESTE: dados que o modelo nunca viu durante o treino.")

# ---------------- Aba 3: métricas ----------------
with tab3:
    st.subheader("Métricas no período de teste")
    st.dataframe(metrics.round(4))
    st.caption("Linhas 'Baseline' são regras simples de comparação. "
               "Um modelo só é útil se superar o baseline.")
    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        st.write(f"Treino: {meta['train_start']} a {meta['train_end']} · "
                 f"Teste: {meta['test_start']} a {meta['test_end']}")

# ---------------- Aba 4: insights ----------------
with tab4:
    row = last.iloc[0]
    vol_med = d["vol_10"].median()
    m = metrics.loc[model_name]
    base = metrics.loc[BASE_ZERO]
    st.subheader("O que os dados indicam")
    st.markdown(f"""
- **Tendência de curto prazo:** o preço está **{abs(row['ma_20_ratio']) * 100:.1f}% {'acima' if row['ma_20_ratio'] > 0 else 'abaixo'}** da média móvel de 20 dias.
- **Volatilidade recente:** {'**alta**' if row['vol_10'] > vol_med else '**normal/baixa**'} ({row['vol_10'] * 100:.2f}% ao dia, contra mediana histórica de {vol_med * 100:.2f}%).
- **Previsão do modelo:** variação de **{ret_pred * 100:+.2f}%** no próximo pregão, com faixa provável de US$ {low:,.0f} a US$ {high:,.0f}.
- **Confiabilidade:** o modelo acertou a direção (sobe/desce) em **{m['Acerto de direção'] * 100:.1f}%** dos dias de teste.
- **Comparação com o baseline:** RMSE do modelo = {m['RMSE']:.2f} contra {base['RMSE']:.2f} da persistência — {'o modelo **superou** o baseline.' if m['RMSE'] < base['RMSE'] else 'o baseline **não foi superado**: no curto prazo, o preço de amanhã é muito parecido com o de hoje.'}
""")
    st.warning("Projeto acadêmico. Previsões de bolsa têm grande incerteza e **não** são recomendação de investimento.")
