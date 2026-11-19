# Arquitetura

## Visão geral

```
data/raw (CSV) ──► src/churn/data.py (limpeza)
                         │
                         ▼
               src/churn/features.py (ColumnTransformer)
                         │
                         ▼
               src/churn/train.py (CV + comparação) ──► models/churn_model.joblib
                                                              │
                                                              ▼
                                        src/churn/api/main.py (FastAPI: /health, /predict)
                                                              │
                                                              ▼
                                               Docker ──► AWS (deploy opcional)
```

O pré-processamento e o estimador são salvos juntos em um único `Pipeline`, o que garante que
treino e inferência apliquem exatamente as mesmas transformações.

## Deploy na AWS (opcional no enunciado)

Fluxo proposto, a confirmar na Etapa 3/4:

1. Build da imagem com o `Dockerfile` da raiz.
2. Push para o **Amazon ECR**.
3. Execução no **Amazon ECS (Fargate)** atrás de um Application Load Balancer, com health check em `/health`.
4. Logs no **CloudWatch**.

Alternativa de menor custo: **AWS Lambda** com imagem de container + adaptador ASGI (Mangum) e API Gateway.

```bash
# exemplo (a ajustar)
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin <conta>.dkr.ecr.$AWS_REGION.amazonaws.com
docker build -t churn-api .
docker tag churn-api:latest <conta>.dkr.ecr.$AWS_REGION.amazonaws.com/churn-api:latest
docker push <conta>.dkr.ecr.$AWS_REGION.amazonaws.com/churn-api:latest
```
