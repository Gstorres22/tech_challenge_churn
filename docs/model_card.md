# Model Card — Churn Classifier

> Etapa 4. Baseado em *Model Cards for Model Reporting* (Mitchell et al., 2019).

## Detalhes do modelo
- **Tipo:** _a definir (modelo campeão)_
- **Versão:** 0.1.0
- **Framework:** scikit-learn (Pipeline: pré-processamento + estimador)
- **Seed:** 42

## Uso pretendido
- Priorizar clientes para ações de retenção.
- **Fora de escopo:** decisões automáticas sem revisão humana; uso em outros setores/datasets.

## Dados
- **Treino/Avaliação:** Telco Customer Churn (IBM), split estratificado 80/20 + CV 5-fold.

## Métricas
| Modelo | AUC-ROC | F1 | Precision | Recall |
|--------|---------|----|-----------|--------|
| Baseline (Regressão Logística) | | | | |
| Random Forest | | | | |
| MLP | | | | |

## Limitações
- _a preencher_

## Vieses e considerações éticas
- Avaliar desempenho por `gender` e `SeniorCitizen`.

## Recomendações
- _a preencher_
