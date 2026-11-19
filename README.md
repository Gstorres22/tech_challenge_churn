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
├── .github/
│   ├── scripts/
│   │   ├── versioning.py          # Cálculo sequencial de versão (vX.Y.Z)
│   │   └── model_quality_gate.py  # Quality gate do modelo e métricas
│   └── workflows/
│       ├── dev-pipeline.yml       # Dev (develop): testes rápidos + PR Release
│       ├── homol-pr.yml           # Homol (PR -> main): validação completa
│       ├── _homol-checks.yml      # Checks reutilizáveis de Homol/Prod
│       └── prod-release.yml       # Prod (main): validação final + Tag/Release
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

## Esteira CI/CD (GitHub Actions — Dev, Homol e Prod)

A esteira do projeto implementa um fluxo contínuo em três camadas (**Dev → Homol → Prod**) com controle de qualidade, cobertura mínima, empacotamento, containerização e versionamento automático por tags.

### Visão geral do fluxo

```text
push em develop (DEV)
  ├─► dev-checks: ruff check/format + pytest
  └─► release-pr: se houver commits à frente da main,
      abre ou atualiza o PR "Release vX.Y.Z" (develop -> main)
           │
           ▼
PR "Release vX.Y.Z" -> main (HOMOL)
  ├─► homol-lint-and-coverage: ruff + pytest com cobertura mínima (>= 70%)
  ├─► homol-package-and-ml-gate: pip install . + smoke import + ML Quality Gate
  └─► homol-docker-smoke-test: docker build + teste de /health e /predict
           │
           ▼  (aprovação e merge manual com "Create a merge commit")
push/merge em main (PROD)
  ├─► prod-checks: reexecuta todos os checks de Homol na main
  └─► publish-release: cria tag anotada vX.Y.Z + GitHub Release automática
```

### Regra de versionamento sequencial

- O versionamento é calculado por [.github/scripts/versioning.py](.github/scripts/versioning.py) de forma sequencial no componente `patch`: `v0.1.0` → `v0.1.1` → `v0.1.2` → ...
- Enquanto o PR `Release vX.Y.Z` estiver aberto, novos pushes em `develop` **atualizam o mesmo PR** (com a lista atualizada de commits) sem abrir PRs duplicados.
- Após o merge na `main` e publicação da tag `vX.Y.Z`, o próximo commit na `develop` abrirá automaticamente um novo PR para `vX.Y.(Z+1)`.
- **Salto manual de versão:** para iniciar uma nova minor/major (ex.: `v0.2.0`), atualize `[project].version` no [pyproject.toml](pyproject.toml) para `"0.2.0"` — se for maior que a última tag existente, a esteira adotará `v0.2.0` e seguirá somando `+1` no patch a partir dela.

### O que cada arquivo da esteira faz

| Arquivo | Camada | Gatilho (`on:`) | Responsabilidade |
|---------|--------|-----------------|------------------|
| [.github/workflows/dev-pipeline.yml](.github/workflows/dev-pipeline.yml) | **Dev** | `push` em `develop` (ou manual) | Roda `ruff` + `pytest`, abre/atualiza o PR `Release vX.Y.Z` e dispara o Homol na criação do PR. |
| [.github/workflows/homol-pr.yml](.github/workflows/homol-pr.yml) | **Homol** | `pull_request` para `main` (`opened`, `synchronize`, `reopened`) ou manual | Dispara o workflow reutilizável de Homol sempre que o PR recebe novos commits. |
| [.github/workflows/_homol-checks.yml](.github/workflows/_homol-checks.yml) | **Homol / Prod** | `workflow_call` | Centraliza os 3 jobs pesados de homologação (cobertura, pacote/ML e Docker). |
| [.github/workflows/prod-release.yml](.github/workflows/prod-release.yml) | **Prod** | `push` em `main` (ou manual) | Revalida toda a `main`, cria a tag `vX.Y.Z` e publica a GitHub Release com release notes. |
| [.github/scripts/versioning.py](.github/scripts/versioning.py) | Script CI | Chamado pelos workflows | Lê tags Git e `[project].version` para determinar `vX.Y.Z` e `Release vX.Y.Z`. |
| [.github/scripts/model_quality_gate.py](.github/scripts/model_quality_gate.py) | Script CI | Chamado em Homol/Prod | Valida carga do `.joblib`, schema Pydantic de entrada, `predict_proba` e métricas mínimas. |

### O que cada teste e quality gate valida

1. **Qualidade de código (`ruff check` / `ruff format --check`):** garante estilo uniforme, ausência de imports/variáveis não usadas e conformidade com as regras do [pyproject.toml](pyproject.toml).
2. **Testes unitários e de contrato (`pytest` em `tests/`):**
   - [tests/test_preprocessing.py](tests/test_preprocessing.py): carga do CSV, limpeza determinística, imputação e ausência de `NaN`s.
   - [tests/test_api.py](tests/test_api.py): endpoints `/health` (200), validação Pydantic em `/predict` (422), indisponibilidade do modelo (503) e inferência com probabilidade válida (200).
   - [tests/test_evaluate_and_pipeline.py](tests/test_evaluate_and_pipeline.py): métricas técnicas (`roc_auc`, `f1`, `precision`, `recall`, `accuracy`), limiar customizado, famílias de modelos exigidas com `SEED = 42` e construção do `Pipeline`.
   - [tests/test_versioning.py](tests/test_versioning.py): regras de incremento sequencial, ordenação numérica (`v0.1.9` < `v0.1.10`), filtro de tags inválidas e leitura do `pyproject.toml`.
   - [tests/test_model_quality_gate.py](tests/test_model_quality_gate.py): comportamento de *skip* antes da Etapa 2, bloqueio quando obrigatório e validação de artefato `.joblib` + `metrics.json`.
3. **Gate de cobertura (`--cov-fail-under=70`):** exige no mínimo 70% de cobertura sobre o pacote `src/churn/` em Homol e Prod.
4. **Empacotamento limpo (`pip install .`):** instala apenas dependências de produção (sem `[dev]`) e verifica a importação do módulo e rotas da API FastAPI.
5. **Quality Gate de ML (`model_quality_gate.py`):** enquanto `models/churn_model.joblib` não existir (Etapa 1), registra `SKIP` sem bloquear a esteira; após o treino (Etapa 2+), valida inferência e métricas mínimas (`roc_auc >= 0.75`, `f1 >= 0.50`).
6. **Build Docker e Smoke Test (`homol-docker-smoke-test`):** constrói a imagem pelo [Dockerfile](Dockerfile), sobe o container, valida `/health` (HTTP 200) e `/predict` com payload inválido (HTTP 422) — e, quando houver modelo carregado, valida uma predição completa (HTTP 200).

### Onde ajustar testes, limiares e regras

| O que deseja ajustar | Arquivo | Onde editar |
|----------------------|---------|-------------|
| Adicionar ou alterar testes automatizados | `tests/test_*.py` | Crie funções `test_*` seguindo as fixtures de [tests/conftest.py](tests/conftest.py). |
| Alterar cobertura mínima, limiares de ML ou exigir modelo | [.github/workflows/_homol-checks.yml](.github/workflows/_homol-checks.yml) | Bloco `on.workflow_call.inputs` (`min_coverage`, `require_model`, `min_roc_auc`, `min_f1`). Após concluir a Etapa 2, mude `require_model` para `true`. |
| Mudar regras do `ruff`, `pytest` ou cobertura | [pyproject.toml](pyproject.toml) | Seções `[tool.pytest.ini_options]`, `[tool.ruff]` e `[tool.coverage.run]`. |
| Mudar versão inicial ou forçar salto de versão | [pyproject.toml](pyproject.toml) | Campo `version` em `[project]`. |
| Alterar fluxo de triggers ou nomes de branches | `.github/workflows/*.yml` | Seção `on:` dos workflows (`develop` / `main`). |

### Configuração única necessária no GitHub

Para que a esteira consiga abrir automaticamente o PR de `develop` para `main` e criar tags/releases:

1. No repositório no GitHub, acesse **Settings → Actions → General**.
2. Em **Workflow permissions**, habilite **Allow GitHub Actions to create and approve pull requests** e clique em **Save**.
3. Ao aprovar e subir um PR `Release vX.Y.Z` para a `main`, utilize **Create a merge commit** (evite *Squash and merge* para que `develop` e `main` mantenham histórico alinhado entre releases).
