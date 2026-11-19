# Notebooks (experimentação)

Notebooks são para **exploração**. Código reaproveitável vai para `src/churn/` e é importado aqui
(`from churn.data import load_raw, clean`).

Convenção de nomes: `NN_descricao.ipynb`.

| Notebook                  | Etapa | Conteúdo |
|---------------------------|-------|----------|
| `01_eda.ipynb`            | 1     | Volume, qualidade e distribuição das features |
| `02_baseline.ipynb`       | 1     | Regressão Logística + definição de métricas |
| `03_model_comparison.ipynb` | 2   | Random Forest, MLP, validação cruzada e escolha do campeão |

Boas práticas: fixar `SEED` de `churn.config` e limpar outputs antes do commit (hook `nbstripout`).
