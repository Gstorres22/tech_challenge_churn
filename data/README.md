# Dados

| Pasta        | Conteúdo                                             | Versionado? |
|--------------|------------------------------------------------------|-------------|
| `raw/`       | Dataset original, imutável                            | Só o CSV oficial |
| `interim/`   | Dados intermediários (limpeza parcial)                | Não |
| `processed/` | Dados prontos para modelagem                          | Não |

## Dataset

**Telco Customer Churn (IBM)**: 7.043 clientes, 21 colunas, alvo `Churn` (Yes/No).

1. Baixe em <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>.
2. Salve como `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

O arquivo bruto nunca deve ser editado; toda transformação fica em `src/churn/data.py`.
