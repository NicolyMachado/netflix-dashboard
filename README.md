# Netflix (NFLX) – Dashboard de previsão de preço

Trabalho Bimestral – Sistemas de Apoio à Decisão · UFMS Câmpus de Três Lagoas.

Modelo de regressão em série temporal que prevê o fechamento do **próximo pregão** das ações da Netflix
e um dashboard em **Streamlit** para explorar histórico, previsões, métricas e insights.

## Dataset
Netflix Stock Price (All Time) – Kaggle: https://www.kaggle.com/akpmpr/updated-netflix-stock-price-all-time

## Como reproduzir
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py        # treina e salva os modelos em models/*.joblib
python evaluate.py     # gera outputs/metricas.csv e outputs/predicoes_teste.csv
streamlit run app.py   # abre o dashboard
```

## Estrutura
| Arquivo | Função |
|---|---|
| `preprocess.py` | leitura, limpeza, atributos e divisão treino/teste por data |
| `modeling.py` | modelos (Random Forest, MLP), baselines, previsão e métricas |
| `train.py` | validação cruzada temporal, treino final e salvamento (.joblib) |
| `evaluate.py` | tabela de métricas MAE, MSE, RMSE e R² no teste |
| `app.py` | dashboard Streamlit |

## Modelos
- **Random Forest** (algoritmo baseado em árvores) e **MLP** (rede neural artificial).
- Comparados com dois *baselines*: persistência e média móvel de 5 dias.
- Alvo: retorno do dia seguinte; o preço previsto = fechamento × (1 + retorno previsto).
