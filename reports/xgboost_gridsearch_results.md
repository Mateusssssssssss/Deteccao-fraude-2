# XGBoost + SMOTE — GridSearchCV

Grade: 54 combinações × 3 folds temporais. Seleção por F1; threshold 0,50.

Melhores parâmetros na grade: `{"max_depth": 6, "n_estimators": 400, "reg_alpha": 0.0, "reg_lambda": 0.0}`.

F1 médio da melhor combinação na CV usada na seleção: **81.24%**.

## Validação temporal dentro do TRAIN, fora desta busca

| Receita | F1 | Recall | Precision | AP | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost inicial + SMOTE | 41.46% | 77.27% | 28.33% | 0.7276 | 34 | 86 | 10 |
| XGBoost GridSearch + SMOTE | 62.39% | 77.27% | 52.31% | 0.7059 | 34 | 31 | 10 |
| Dummy (prevalência) | 0.00% | 0.00% | 0.00% | 0.0008 | 0 | 0 | 44 |

Esta janela já foi explorada em rodadas anteriores. Não é confirmação independente de todas as decisões do projeto. O TEST não foi pontuado; o Random Forest escolhido permanece preservado.

Notebook: notebooks/xgboost_gridsearch.ipynb. Resultados completos: reports/metrics/xgboost_gridsearch_results.csv.
