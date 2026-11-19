# tech_challenge_churn

Modelo de previsão de churn de clientes, do EDA ao deploy em produção na AWS.
Projeto da Fase 1 do Tech Challenge (Pós-Tech FIAP – Machine Learning Engineering).

## Contexto

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. Este projeto constrói
um modelo que classifica clientes com risco de cancelamento, desde a análise exploratória até o
modelo servido via API REST, seguindo boas práticas de engenharia de software.

- **Dataset:** [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Modelos:** Regressão Logística (baseline), Random Forest e MLPClassifier (scikit-learn)
- **API:** FastAPI com `/health` e `/predict`
- **Testes:** pytest
- **Deploy:** Docker + AWS

## Estrutura do repositório

```
tech_challenge_churn/
├── .github/workflows/ci.yml   # Lint (ruff) + testes (pytest) a cada push/PR
├── data/
│   ├── raw/                   # Dataset original (imutável)
│   ├── interim/               # Dados intermediários
│   └── processed/             # Dados prontos para modelagem
├── docs/
│   ├── ml_canvas.md           # Etapa 1: ML Canvas
│   ├── experiments.md         # Etapa 2: registro de experimentos
│   ├── model_card.md          # Etapa 4: Model Card
│   └── architecture.md        # Arquitetura e deploy na AWS
├── models/                    # Modelo campeão serializado (.joblib)
├── notebooks/                 # Experimentação (EDA, baseline, comparação)
├── reports/figures/           # Gráficos gerados
├── src/churn/                 # Código produtivo
│   ├── config.py              # Caminhos, SEED e lista de features
│   ├── data.py                # Carga e limpeza
│   ├── features.py            # Pipeline de pré-processamento
│   ├── train.py               # Treino e comparação de modelos
│   ├── evaluate.py            # Métricas
│   ├── predict.py             # Carga do modelo e inferência
│   └── api/
│       ├── main.py            # FastAPI (/health, /predict)
│       └── schemas.py         # Contratos Pydantic
├── tests/                     # Testes pytest
├── Dockerfile                 # Imagem da API
├── pyproject.toml             # Dependências e configuração de ferramentas
└── README.md
```

## Setup

Pré-requisito: Python 3.11+.

```bash
# 1. Ambiente virtual
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

# 2. Dependências (pacote em modo editável + ferramentas de desenvolvimento)
pip install -e ".[dev]"

# 3. Hooks de qualidade (opcional)
pre-commit install
```

Depois, baixe o dataset conforme [data/README.md](data/README.md).

## Execução

```bash
# Treinar e salvar o modelo campeão em models/churn_model.joblib
python -m churn.train

# Subir a API localmente (docs interativas em http://localhost:8000/docs)
uvicorn churn.api.main:app --reload

# Rodar os testes
pytest

# Lint e formatação
ruff check . && ruff format .
```

### Exemplo de chamada

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,
       "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
       "OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No",
       "StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":79.85,"TotalCharges":958.2}'
```

### Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Etapas do projeto

| Etapa | Foco | Entregável | Status |
|-------|------|------------|--------|
| 1 | Entendimento e preparação | Notebook de EDA + baseline (Regressão Logística) | ⬜ |
| 2 | Modelagem e avaliação | Tabela comparativa + modelo campeão salvo | ⬜ |
| 3 | Engenharia de software e API | Código em `src/` + testes + API FastAPI | ⬜ |
| 4 | Documentação e entrega | Model Card + README final + vídeo STAR | ⬜ |
| — | Deploy (opcional) | API em produção na AWS | ⬜ |

## Reprodutibilidade

- Seed global fixada em `src/churn/config.py` (`SEED = 42`).
- Pré-processamento e modelo salvos juntos em um único `Pipeline` do scikit-learn.
- Dependências declaradas no `pyproject.toml`.

## Documentação

- [ML Canvas](docs/ml_canvas.md)
- [Registro de experimentos](docs/experiments.md)
- [Model Card](docs/model_card.md)
- [Arquitetura e deploy](docs/architecture.md)
