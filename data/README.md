# Dados

| Pasta        | Conteúdo                                             | Versionado? |
|--------------|------------------------------------------------------|-------------|
| `raw/`       | Dataset original, imutável                            | Só o CSV oficial |
| `interim/`   | Dados intermediários (limpeza parcial)                | Não |
| `processed/` | Dados prontos para modelagem                          | Não |

## Dataset

**Telco Customer Churn (IBM)**: 7.043 clientes, 21 colunas, alvo `Churn` (Yes/No).

### Download automatizado (recomendado)

Defina `KAGGLE_API_TOKEN` no arquivo `.env` (não versionado) ou nas variáveis de ambiente e execute:

```bash
python -m churn.download
```

No Windows com o ambiente virtual local, utilize `.\.venv\Scripts\python.exe -m churn.download`. O módulo `src/churn/download.py` baixa o ZIP oficial do Kaggle e extrai o arquivo em `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (ignorando o download se o CSV já existir).

### Download manual (alternativa)

1. Baixe em <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>.
2. Salve como `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

O arquivo bruto nunca deve ser editado; toda transformação fica em `src/churn/data.py`.
