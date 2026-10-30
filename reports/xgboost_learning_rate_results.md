# XGBoost + SMOTE — GridSearchCV

Grade: 162 combinações × 3 folds temporais. Seleção por F1; threshold 0,50.

Melhores parâmetros na grade: `{"learning_rate": 0.01, "max_depth": 3, "n_estimators": 100, "reg_alpha": 1.0, "reg_lambda": 1.0}`.

F1 médio da melhor combinação na CV usada na seleção: **82.72%**.

## Validação temporal dentro do TRAIN, fora desta busca

| Receita | F1 | Recall | Precision | AP | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost inicial + SMOTE | 41.46% | 77.27% | 28.33% | 0.7276 | 34 | 86 | 10 |
| XGBoost GridSearch + SMOTE | 74.73% | 77.27% | 72.34% | 0.7054 | 34 | 13 | 10 |
| Dummy (prevalência) | 0.00% | 0.00% | 0.00% | 0.0008 | 0 | 0 | 44 |

Esta janela já foi explorada em rodadas anteriores. Não é confirmação independente de todas as decisões do projeto. O TEST não foi pontuado; o Random Forest escolhido permanece preservado.

Notebook de treino: notebooks/xgboost_gridsearch.ipynb. Estatísticas: notebooks/xgboost_model_metrics.ipynb. Resultados completos: reports/metrics/xgboost_learning_rate_results.csv.
