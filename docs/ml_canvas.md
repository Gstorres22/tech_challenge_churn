# ML Canvas — Previsão de Churn (Telco)

> Etapa 1. Preencher durante o entendimento do problema.

| Bloco | Descrição |
|-------|-----------|
| **Proposta de valor** | Identificar clientes com alto risco de cancelamento para ações de retenção antes do churn. |
| **Stakeholders** | Diretoria, time de retenção/CRM, marketing, time de dados. |
| **Decisões** | Quais clientes recebem ação de retenção (oferta, contato, desconto). |
| **Tarefa de ML** | Classificação binária supervisionada (`Churn` = 1/0), saída em probabilidade. |
| **Fontes de dados** | Telco Customer Churn (IBM): perfil, serviços contratados, contrato e cobrança. |
| **Features** | _a definir após EDA_ |
| **Métrica técnica** | _a definir_ (candidatas: AUC-ROC, F1, Recall da classe churn) |
| **Métrica de negócio** | _a definir_ (ex.: receita retida líquida = churn evitado × LTV − custo da ação) |
| **Predições** | Online via API `/predict`, por cliente. |
| **Avaliação offline** | Validação cruzada estratificada + holdout. |
| **Monitoramento** | _a definir_ (drift de features, taxa de churn real vs. prevista). |
| **Riscos** | Desbalanceamento de classes, vieses por gênero/idade, mudança de comportamento ao longo do tempo. |
