# SMOTE isolado e undersampling isolado — F1 principal

Médias dos mesmos folds temporais do TRAIN, pesos uniformes e threshold comum 0,50. F1 equilibra Precision e Recall com pesos simétricos. O relatório atual não usa F2.

As previsões e contagens já calculadas foram preservadas. Trocar a métrica de seleção não muda Precision ou Recall no mesmo threshold. Não houve novo treino nem reteste para atualizar estes relatórios.

| Receita | F1 médio | Recall médio | Precision média | AP média |
|---|---:|---:|---:|---:|
| random_forest__smote | 79.55% | 76.31% | 83.74% | 0.7557 |
| xgboost__none | 77.64% | 69.00% | 90.68% | 0.7684 |
| random_forest__none | 76.53% | 72.14% | 82.86% | 0.7494 |
| xgboost__smote | 64.42% | 76.00% | 62.23% | 0.7646 |
| random_forest__under | 60.08% | 79.65% | 49.45% | 0.7427 |
| xgboost__under | 55.54% | 77.90% | 46.64% | 0.7531 |

## Mudança frente ao controle sem reamostragem

| Modelo | Sampler | Delta F1 (p.p.) | Delta Recall (p.p.) | Delta Precision (p.p.) |
|---|---|---:|---:|---:|
| random_forest | smote | +3.02 | +4.17 | +0.88 |
| random_forest | under | -16.45 | +7.50 | -33.41 |
| xgboost | smote | -13.21 | +6.99 | -28.46 |
| xgboost | under | -22.10 | +8.90 | -44.04 |

## Gráficos e notebooks

Cada gráfico principal contém dois grupos (Random Forest e XGBoost), com três barras por grupo: F1, Recall e Precision. Os controles aparecem nas tabelas. A sensibilidade por threshold também usa barras agrupadas dessas três métricas.

Notebooks: notebooks/metricas_smotec.ipynb e notebooks/metricas_under.ipynb. SMOTEC é o nome de arquivo solicitado; o sampler é SMOTE numérico.

Os gráficos de treino/validação usam F1. PR/AP, ROC, matriz OOF, curvas AP/Log Loss do XGBoost, correlação/importance e diversidade de erros continuam complementares.

**Limites:** mesmos folds reutilizados; diferenças de médias não provam superioridade estatística. Threshold 0,50 é comum e não foi otimizado nesta atualização. OOF agregado não é a média dos folds. O artefato histórico com limiar 0,70 continua preservado como histórico; nenhum novo modelo orientado por F1 foi promovido nem avaliado no TEST.
