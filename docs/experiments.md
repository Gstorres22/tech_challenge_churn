# Registro de Experimentos

> Etapa 2. Rastreabilidade completa de validação cruzada (`StratifiedKFold(5)` em `X_train`, $n=5.634$), calibração *out-of-fold* (OOF) de threshold e avaliação final no holdout (`X_test`, $n=1.409$), reproduzindo exatamente `models/metrics.json` e `reports/model_comparison.csv`.

## Configuração Experimental Padrão

- **Dataset:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (7.043 linhas, 19 features após limpeza em `churn.data.clean`, taxa global de churn 26,54%).
- **Split:** Estratificado 80/20 com `SEED = 42` (`train_rows = 5.634`, `test_rows = 1.409`, `churn_rate = 26,54%` em ambos).
- **Pré-processamento (fit apenas no treino de cada fold / `X_train`):** `SimpleImputer(median)` + `StandardScaler` em numéricas; `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")` em categóricas (`churn.features.build_preprocessor`).
- **Premissas de negócio (`churn.config`):** `LTV = 6 meses`, `taxa de sucesso = 30%`, `custo da ação = US$ 30,00`.
- **Critério de seleção do campeão:** Maior `cv_roc_auc_mean` em 5-fold CV no treino (desempate por `oof_net_retained_revenue` e `oof_f1_opt`), com seleção de threshold no OOF de treino sujeita aos pisos do Quality Gate (`F1 >= 0,55` e `Recall >= 0,60`).

---

## Tabela 1 — Validação Cruzada (5-Fold em `X_train`) e Otimização OOF de Threshold

| Data | ID | Rank | Modelo | Hiperparâmetros | CV Train AUC | CV Val AUC (média ± dp) | Overfit Gap | CV PR-AUC | CV F1 (@0,50) | Opt Thr (OOF) | OOF F1 (@Opt) | OOF Recall (@Opt) | OOF Net Rev (@Opt) | OOF ROI (@Opt) | Campeão? |
|------|----|------|--------|-----------------|--------------|-------------------------|-------------|-----------|-------------|---------------|---------------|-------------------|--------------------|----------------|----------|
| 2026-11-21 | exp-001 | 2 | `logistic_regression` | `max_iter=1000, C=1.0, random_state=42` | 0.8496 | 0.8461 ± 0.0125 | 0.0034 | 0.6614 | 0.5930 | 0.28 | 0.6358 | 0.7900 | US$ 99.188,82 | 1.49x | Não |
| 2026-11-21 | exp-002 | 1 | `random_forest` | `n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42` | 0.9202 | **0.8465 ± 0.0090** | 0.0737 | **0.6618** | 0.5810 | **0.23** | 0.6217 | **0.8502** | US$ 96.656,34 | 1.24x | **Sim** |
| 2026-11-21 | exp-003 | 3 | `mlp` | `hidden_layer_sizes=(64, 32), alpha=0.1, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, max_iter=500, random_state=42` | 0.8659 | 0.8410 ± 0.0119 | 0.0249 | 0.6471 | 0.5911 | 0.24 | 0.6218 | 0.8060 | US$ 96.841,20 | 1.36x | Não |

> Nota de referência: no conjunto de treino (`n = 5.634`), o baseline *Target-All* gera **US$ 32.428,71** de receita retida líquida; todos os modelos triplicam esse resultado na calibração OOF.

---

## Tabela 2 — Avaliação Final no Holdout (`X_test`, $n = 1.409$, 374 *churners*)

### 2A. Comparação com Limiar Padrão (`0.50`) vs. Limiar Ótimo Calibrado no OOF de Treino (`Opt Thr`)

| Modelo | Holdout ROC-AUC | Holdout PR-AUC | F1 (@0,50 / @Opt) | Precision (@0,50 / @Opt) | Recall (@0,50 / @Opt) | Accuracy (@0,50 / @Opt) | Net Rev (@0,50) | Net Rev (@Opt) | ROI (@0,50 / @Opt) | Ganho vs. Target-All (@Opt) |
|--------|-----------------|----------------|-------------------|--------------------------|-----------------------|-------------------------|-----------------|----------------|--------------------|-----------------------------|
| `random_forest` (campeão, `@0,23`) | **0.8430** | **0.6559** | 0.5848 / 0.6142 | 0.6748 / 0.4860 | 0.5160 / **0.8342** | 0.8055 / 0.7218 | US$ 18.101,22 | US$ 22.371,93 | 2.11x / 1.16x | US$ 15.655,11 |
| `logistic_regression` (`@0,28`) | 0.8419 | 0.6338 | 0.6040 / 0.6226 | 0.6572 / 0.5177 | 0.5588 / 0.7807 | 0.8055 / 0.7488 | US$ 20.045,34 | **US$ 23.335,65** | 2.10x / **1.38x** | **US$ 16.618,83** |
| `mlp` (`@0,24`) | 0.8417 | 0.6356 | 0.5870 / **0.6293** | 0.6760 / 0.5144 | 0.5187 / 0.8102 | 0.8062 / 0.7466 | US$ 19.018,83 | US$ 23.006,76 | 2.21x / 1.30x | US$ 16.289,94 |

### 2B. Detalhamento Operacional no Holdout (Abordagens, Acertos e Receitas)

| Modelo | Threshold | Clientes Abordados | TP Capturados (de 374) | Falsos Positivos | Receita Bruta Salva | Custo da Campanha | Receita Retida Líquida | Baseline Target-All | Ganho vs. Target-All |
|--------|-----------|--------------------|------------------------|------------------|---------------------|-------------------|------------------------|---------------------|----------------------|
| `random_forest` | `0.50` | 286 | 193 | 93 | US$ 26.681,22 | US$ 8.580,00 | US$ 18.101,22 | US$ 6.716,82 | US$ 11.384,40 |
| `random_forest` (**ótimo**) | **`0.23`** | **642** | **312** | **330** | **US$ 41.631,93** | **US$ 19.260,00** | **US$ 22.371,93** | **US$ 6.716,82** | **US$ 15.655,11** |
| `logistic_regression` | `0.50` | 318 | 209 | 109 | US$ 29.585,34 | US$ 9.540,00 | US$ 20.045,34 | US$ 6.716,82 | US$ 13.328,52 |
| `logistic_regression` (ótimo) | `0.28` | 564 | 292 | 272 | US$ 40.255,65 | US$ 16.920,00 | US$ 23.335,65 | US$ 6.716,82 | US$ 16.618,83 |
| `mlp` | `0.50` | 287 | 194 | 93 | US$ 27.628,83 | US$ 8.610,00 | US$ 19.018,83 | US$ 6.716,82 | US$ 12.302,01 |
| `mlp` (ótimo) | `0.24` | 589 | 303 | 286 | US$ 40.676,76 | US$ 17.670,00 | US$ 23.006,76 | US$ 6.716,82 | US$ 16.289,94 |

---

## Conclusão e Decisão Técnica

- **Modelo campeão escolhido:** `random_forest` (`models/churn_model.joblib`, `3.190,60 KB`), selecionado estritamente por liderar a **CV ROC-AUC no treino (`0,8465 ± 0,0090`)**, com menor variância entre folds (`0,0090`) e maior `PR-AUC` tanto em CV (`0,6618`) quanto no holdout (`0,6559`).
- **Impacto do threshold calibrado (`0,23` vs `0,50`):** eleva o `Recall` no holdout de **51,60%** para **83,42%** (+119 *churners* capturados) e adiciona **+US$ 4.270,71** de receita retida líquida sobre o corte padrão `0,50`.
- **Competitividade da `logistic_regression`:** embora fique apenas `0,0004` abaixo em CV ROC-AUC (`0,8461`), apresenta *overfitting gap* mínimo (`0,0034` vs `0,0737`) e gerou receita líquida e ROI ligeiramente maiores no holdout (`US$ 23.335,65`, ROI `1,38x`). Seguindo boas práticas anti-*peeking*, não alteramos o vencedor após observar o holdout; registramos a **calibração de probabilidades** (Platt/isotônica) e a seleção orientada diretamente a retorno líquido em CV como próximos passos de evolução (v0.2).
