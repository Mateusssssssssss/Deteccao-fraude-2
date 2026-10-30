# Modelo escolhido

Random Forest de metricas_smotec: 160 árvores, max_depth=14,
min_samples_leaf=2, SMOTE (razão 0,10, k=3), 14 features, threshold **0,70**.

- best_model.joblib: pipeline completo, features, classe positiva e threshold.
- metadata.json: parâmetros, versões, hashes e origem do ajuste.
- requirements.txt: versões necessárias ao pipeline.
- request_example.json: requisição sintética para a API.
- mlflow_run.json: referência da execução local no MLflow.

O pacote reutiliza o ajuste no TRAIN completo já existente em
models/standardized_selected_pipeline.joblib. A receita coincide com a escolha
de metricas_smotec/model_select; o ajuste dos folds de validação é distinto.
A exportação não executa fit nem ajusta o threshold.

Reproduzir pela raiz:

    python -m fraud_detection.deployment

Avaliação e gráficos: notebooks/best_model_metrics.ipynb.
API: python -m uvicorn fraud_detection.api:app --host 127.0.0.1 --port 8000.
Tracking: python -m fraud_detection.tracking.

O binário e o armazenamento MLflow são locais e ignorados pelo Git. Os metadados
e instruções podem ser versionados. O artefato deve vir de uma origem confiável:
a API é configurada pelo operador para carregar este arquivo local.
