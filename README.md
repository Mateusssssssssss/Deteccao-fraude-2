# Detecção de fraude — projeto reformulado

Este repositório é uma **reformulação do [Projeto-Deteccao-De-Fraude](https://github.com/Mateusssssssssss/Projeto-Deteccao-De-Fraude)**,
criado no início da minha caminhada em Ciência de Dados e Machine Learning.
Ao revisitar o trabalho, identifiquei vazamento de dados (*data leakage*) no fluxo
de balanceamento. Esta versão registra essa evolução: corrige a metodologia,
organiza o código e torna a análise e a avaliação reproduzíveis.

Na versão anterior, undersampling e SMOTE eram aplicados à base inteira antes do
split; a seleção por correlação também via toda a base, e a validação cruzada
recebia dados já reamostrados. Além disso, Average Precision era calculada com
rótulos, em vez de probabilidades. Por isso, **as métricas antigas não são
evidência válida da capacidade de generalização** e não são diretamente
comparáveis aos resultados desta reformulação.

Os arquivos originais foram preservados como registro em `docs/legacy/`.
As duas imagens antigas diretamente em `reports/` também são históricas.
Os resultados do novo protocolo ficam em `reports/metrics/` e `reports/figures/`.

## Referência principal: F2

Decisão de 27/10/2026: **manter F2 como objetivo principal** das próximas
comparações, com Precision, Recall, F1 e AP como métricas complementares.
A primeira rodada por AP permanece registrada como histórico.

O pipeline de referência atual é **Random Forest + StandardScaler + SMOTE +
seleção conjunta de 14 features**, com threshold **0,70** escolhido no TRAIN por F2.
Artefato: `models/standardized_selected_pipeline.joblib`.
O notebook `04_final_test_metrics.ipynb` apresenta sua avaliação retrospectiva
no TEST histórico. F3 permanece documentado como experimento separado e não
substitui essa referência. As próximas técnicas serão avaliadas em outra sessão.

## Objetivo e dados

Classificar uma transação no momento da autorização: **Class=1 é fraude**;
Class=0 é legítima. Um falso negativo deixa passar fraude; um falso positivo
pode gerar bloqueio ou revisão indevida. Não foram informados custos financeiros
nem capacidade de revisão; o threshold inicial é fixo em 0,5.

A base é `creditcard.csv`, descrita em `context.txt`: transações europeias de
dois dias de setembro de 2013, features numéricas Time, Amount e V1–V28
(componentes PCA anônimos). O script valida o schema e registra o hash SHA-256
da cópia utilizada. Não inferimos significados semânticos para as PCs.

Coloque o arquivo **original, sem reamostragem** em `data/raw/creditcard.csv`.
A cópia entregue na raiz foi preservada e copiada para esse destino. Dados e
modelo binário são ignorados pelo Git; figuras e métricas agregadas podem ser
versionadas. Verifique a licença da fonte antes de redistribuir a base.
Origem de referência: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Protocolo corrigido

1. Validar dados, retirar apenas cópias exatas e ordenar por Time, preservando raw.
2. Reservar o último quinto da **duração temporal**, sem separar empates de Time.
   A proporção de linhas não precisa ser 20%. O objetivo é avaliar transações futuras.
3. Usar só TRAIN para EDA e CV temporal expansiva em três janelas.
4. Comparar **Random Forest e XGBoost**, cada um sem reamostragem e com
   RandomUnderSampler + SMOTE. DummyClassifier é referência ingênua.
5. Ajustar imputador, scaler (nas receitas SMOTE), samplers e estimador
   **dentro de cada treino de fold** usando imblearn.Pipeline.
6. Selecionar por **Average Precision média na CV**. Precision, Recall, F1,
   ROC-AUC e Balanced Accuracy complementam; threshold=0,5 em todas as receitas.
7. Congelar receita e configuração, reajustar no TRAIN completo e avaliar
   **somente o vencedor no TEST**, que mantém a prevalência original.

RF sem amostragem usa pesos de classe; RF com SMOTE usa pesos uniformes.
XGBoost usa pesos uniformes nas duas receitas. Não acumulamos SMOTE e pesos
arbitrários. A comparação é de receitas completas, sem isolar causalmente o efeito
do balanceamento. Parâmetros prefixados estão em `configs/experiment.yaml`.
Não há busca de hiperparâmetros, calibração ou ajuste de threshold nesta rodada.
AP usa probabilidades e difere da área trapezoidal da curva PR.

## Estrutura

```text
configs/experiment.yaml       # receitas, CV, seed, amostragem e threshold
data/raw/                    # arquivo original local, ignorado no Git
data/interim/                # reservado para derivados, se necessários
data/processed/              # reservado para derivados, se necessários
notebooks/01_eda.ipynb        # EDA somente TRAIN
notebooks/02_model_training.ipynb  # protocolo, configuração e seleção
notebooks/03_model_metrics.ipynb   # CV/OOF, diagnóstico e experimentos
notebooks/04_final_test_metrics.ipynb  # todas as métricas do TEST histórico, referência F2
src/fraud_detection/
  data.py                    # schema e split temporal
  preprocess.py              # imputação e sampling dentro do pipeline
  model.py                   # RF, XGBoost e referência Dummy
  validation.py              # folds temporais e cross_validate multiscorer
  eda.py                     # tabelas e figuras exploratórias
  evaluation.py              # métricas e gráficos
  train.py                   # comparação, congelamento, refit e TEST
  predict.py                 # inferência com o pipeline salvo
tests/                       # regressões contra leakage, schema e inferência
models/                      # pipeline persistido localmente
reports/metrics/             # resultados agregados
reports/figures/             # figuras do protocolo corrigido
reports/experiments/         # manifesto, folds e seleção
docs/methodology.md           # justificativas e limitações
docs/legacy/                 # registro da versão anterior
```

## Instalação e execução

Python 3.11 ou superior (ambiente desta execução: Python 3.12). No PowerShell,
a partir da raiz:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m fraud_detection.eda
.\.venv\Scripts\python.exe -m fraud_detection.train
.\.venv\Scripts\python.exe scripts/summarize_results.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

O ambiente efetivamente executado fica em `requirements-lock.txt`; para reproduzi-lo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Arquivos de configuração alternativos: `--config configs/experiment.yaml`.
A execução interrompe se um fold não tiver as duas classes ou se os ratios de
SMOTE/under não forem válidos. Nenhum downsampling global ou fallback silencioso
é aplicado.

Para reproduzir numa pasta separada:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.train --output reports/reproductions/run_02
```

A saída padrão bloqueia sobrescrita quando já existe avaliação TEST. Reproduzir
o mesmo protocolo não cria uma nova avaliação independente. Se mudar decisões
depois de ver TEST, reserve outro período externo.

## Notebooks

Os quatro notebooks leem relatórios já calculados; não treinam modelos de novo.
Para abri-los:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks
```

Escolha o kernel da .venv. Se precisar registrá-lo, use
`python -m ipykernel install --user --name fraud-detection --display-name "Fraud Detection"`
com o Python desse ambiente. Para executar todos e salvar as saídas:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
```

Se faltarem artefatos, os notebooks informam quais scripts executar.
A EDA inclui qualidade, distribuições, outliers, correlações, taxas por quantis,
tempo e interação exploratória. O notebook de métricas mostra barras agrupadas
de F1/Recall/Precision, média e dispersão por fold, AP, OOF, erros complementares,
curvas por número de árvores, associação versus importância e TEST em seção própria.

## Nova rodada: threshold com ênfase em Recall

Após a primeira avaliação, foi adicionada uma busca específica de threshold
para a receita XGBoost sem reamostragem. **GridSearchCV** percorre os limiares
em `configs/threshold.yaml`, otimizando **F2 (beta=2)**, conforme escolhido:
Recall tem mais peso, mas falsos positivos ainda são penalizados.
O threshold 0 é incluído para demonstrar o limite de maximizar Recall puro.

O modelo é ajustado no prefixo da janela de treino. Um
`FixedThresholdClassifier(FrozenEstimator(pipeline))` permite ao GridSearch
variar apenas o threshold, pontuando numa janela interna posterior que o modelo
não viu. Três folds temporais externos comparam a política ajustada com 0,5.
Depois, o threshold final é escolhido numa janela interna do TRAIN completo
e o pipeline é reajustado em todo o TRAIN.

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.threshold
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
```

Resultados desta rodada: [reports/threshold_results.md](reports/threshold_results.md).
Tabelas: `threshold_grid.csv` (seleção interna),
`threshold_outer_folds.csv` e `threshold_comparison.csv` (validação externa).
O notebook de métricas inclui essa análise antes da seção de TEST histórico.
O artefato novo é `models/recall_pipeline.joblib`; use esse caminho na função
`predict` para aplicar a nova política. O artefato `final_pipeline.joblib`
continua representando a primeira rodada com threshold 0,5.

**O TEST já havia sido avaliado e não foi reutilizado nesta busca.**
A CV externa avalia o ajuste de threshold condicionado à receita escolhida antes;
não é avaliação independente de toda a seleção de modelos. O score da janela
interna final é de seleção, não teste. A nova política precisa de outro período
externo para avaliação independente. Melhor limiar na grade não é ótimo universal
nem threshold de custo financeiro conhecido.

## Resultados

Os resultados reais desta execução e a interpretação resumida estão em
[reports/results.md](reports/results.md). Tabelas completas:
`reports/metrics/model_comparison.csv` (CV),
`cv_folds.csv`, `oof_metrics.csv` e `test_metrics.json` (só vencedor).
O gráfico de seleção é `reports/figures/model_comparison_f1_recall_precision.png`.

OOF cobre apenas as janelas de validação, sem o prefixo inicial; métricas OOF
agregadas diferem da média dos folds. Desvio entre folds não é intervalo de
confiança. Não foram calculados intervalos formais do TEST nesta rodada.

## Inferência

```python
from pathlib import Path
import pandas as pd
from fraud_detection.predict import predict

# Execute pela raiz; transacoes deve conter Time, V1–V28 e Amount numéricos.
transacoes = pd.read_csv("novas_transacoes.csv")
# Pipeline principal: F2, threshold 0,70, 14 features mantidas.
resultado = predict(transacoes, Path("models/standardized_selected_pipeline.joblib"))
```

O artefato contém o pipeline inteiro, a ordem das features e o threshold.
Reamostragem ocorre no fit, nunca durante predict. Carregue apenas artefatos
joblib de origem confiável.

## Limitações e próximos passos

A base tem só dois dias; não há IDs de cartão para separar clientes, nem dados
originais para auditar o PCA fornecido. É assumido que as features existem na
autorização. Como o projeto anterior já explorou a base, esta é uma reavaliação
retrospectiva e não evidência prospectiva.

Sem custos de FP/FN não há threshold operacional ótimo demonstrado; as
probabilidades não foram calibradas. Permutação usa validação dentro do TRAIN e
é descritiva após seleção. Correlação/importance não implicam causalidade.
Mais períodos são necessários para estabilidade e incerteza temporal.

Antes de introduzir complexidade, priorize validação prospectiva e definição dos
custos. Threshold tuning/calibração podem então ajudar, mas precisam de CV interna
ou janela adicional no TRAIN. Early stopping requer validação interna por fold;
Voting/Stacking só se justificam com ganho validado. Não há ganho presumido de
novas técnicas nem alegação de prontidão de produção. Detalhes em
[docs/methodology.md](docs/methodology.md).

Referências técnicas:
[imbalanced-learn: leakage](https://imbalanced-learn.org/stable/common_pitfalls.html)
e [scikit-learn: Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

### Correlação com a target e importância na EDA

O notebook 01 inclui dois painéis com as mesmas 30 features: Pearson com Class no
TRAIN e importância por permutação (queda de AP) na última validação do TRAIN.
Os valores e a dispersão ficam indicados no gráfico e na tabela. É um diagnóstico
pós-modelagem, sem TEST, com escalas distintas e sem interpretação causal.

Para gerar esse painel após o treinamento, execute novamente
`python -m fraud_detection.eda`, que apenas lê as importâncias já calculadas.
Figura: `reports/figures/eda_target_correlation_importance.png`.

As curvas Precision–Recall já estão no notebook 03, nos diagnósticos OOF do TRAIN
e na seção própria de TEST histórico. A legenda informa **AP (Average Precision)**,
que difere da área trapezoidal da curva PR. Alterar o threshold muda o ponto de
decisão, não a curva nem AP quando as probabilidades permanecem iguais.

## StandardScaler, SMOTE e regra conjunta de seleção

A rodada em configs/ablation.yaml compara dez receitas (cinco por família):
todas as features sem scaling; todas padronizadas; todas com scaling+SMOTE;
filtradas+scaling; filtradas+scaling+SMOTE. As métricas principais usam threshold
comum 0,5 nos mesmos folds temporais. Em seguida há GridSearch/F2 de threshold
da melhor receita filtrada, com ajuste interno e avaliação externa no TRAIN.

Regra de retirada: **abs(Pearson com Class) < 0,090 E importância por permutação
< 0,100**. Se só uma condição for atendida, a feature permanece. O sinal negativo
não torna uma correlação forte descartável. A importância é queda de AP em
validação interna de uma referência XGBoost, não porcentagem. A regra é reaprendida
em cada treino para evitar vazamento. Controles completos medem a ablação.

StandardScaler aprende média/desvio no treino do fold; árvores normalmente mudam
pouco, mas SMOTE usa distâncias. SMOTE é aplicado só no treino, mantendo todas
as legítimas e definindo minoria/maioria=0,1. SMOTENC é para entradas mistas
numéricas e categóricas; esta base só tem entradas numéricas. Nenhum método
de balanceamento é garantidamente melhor: veja a comparação executada.

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.eda
.\.venv\Scripts\python.exe -m fraud_detection.ablation
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
```

Resultados: [reports/ablation_results.md](reports/ablation_results.md).
Artefato novo: models/standardized_selected_pipeline.joblib, compatível com predict.
O schema de entrada segue Time, V1–V28 e Amount; o ColumnTransformer usa só as
features mantidas. Os modelos/TEST históricos permanecem separados; esta rodada
não volta a pontuar TEST já visto.

[Guia das métricas e denominadores](docs/metrics_guide.md): definições de Precision,
Recall, F1/F2, AP/PR, ROC-AUC, Log Loss, Brier, FP/FN, dispersão de CV, importância,
qualidade dos dados e taxas por faixa. Os notebooks exibem proporções como
percentuais quando apropriado, sem modificar os valores numéricos dos CSVs.

Fraude observada por faixa: reports/metrics/eda_feature_bins.csv. Cada linha
inclui transações, fraudes, legítimas, taxa de fraude (%) e parcela das fraudes
do TRAIN (%). 0,000190 em proporção equivale a 0,0190%.
Erros por faixa na validação: reports/metrics/category_error_metrics.csv, com
TP/FP/FN/TN, Recall, Precision, FPR e parcela de alertas falsos, cada qual com
seu denominador. Essas faixas numéricas não representam tipos reais de fraude.

O código de src/fraud_detection está comentado em português por blocos de
responsabilidade: contratos de dados, split, prevenção de leakage, fit de scaler,
sampling, regra conjunta, seleção de threshold, inferência e relatórios.

## Avaliação solicitada do pipeline final no TEST histórico

O notebook [04_final_test_metrics.ipynb](notebooks/04_final_test_metrics.ipynb)
reúne todas as métricas da avaliação do Random Forest com StandardScaler, SMOTE,
14 features e threshold 0,70. Esse pipeline já estava congelado antes da execução.
Não houve novo fit, reamostragem do TEST, seleção de features ou threshold tuning.

Como esse período já havia sido avaliado na primeira rodada, o resultado é uma
**reavaliação retrospectiva do TEST histórico**, não um novo teste independente.
Resultados antigos e novos ficam separados; não há seleção de candidatos no TEST.

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.final_test
.\.venv\Scripts\python.exe scripts/execute_notebooks.py --notebook 04_final_test_metrics.ipynb
```

A primeira linha é necessária só para gerar os artefatos uma vez. O notebook
apenas lê os resultados. A avaliação existente é protegida contra sobrescrita;
para reproduzir separadamente use --output, lembrando que isso não cria dados novos.

Relatório: [reports/final_test_results.md](reports/final_test_results.md).
Artefatos: reports/metrics/final_test_metrics.json, final_test_metric_definitions.csv,
final_test_classification_report.json, final_test_calibration.csv,
final_test_time_segments.csv e final_test_feature_segments.csv.
O manifesto em reports/experiments/final_test_manifest.json registra hashes,
horário, split e a ausência de treino/tuning nesta avaliação.

O notebook explica Precision/Recall/F1/F2, AP e AUPRC trapezoidal, ROC-AUC,
especificidade, acurácia e balanced accuracy, FPR/FNR, parcela de alertas falsos,
valor preditivo negativo, Log Loss, Brier, MCC/Kappa, matriz de confusão, métricas
por classe e erros por período/faixa. As unidades e denominadores estão explícitos.

## Rodada F3: maior ênfase em Recall

A receita Random Forest com StandardScaler, SMOTE e regra conjunta de features
permanece fixa. O módulo threshold_f3 repondera as contagens da grade interna
anterior com **F3 = 10TP/(10TP+9FN+FP)**, equivalente ao scorer F-beta com beta=3.
A mesma grade de thresholds e as mesmas janelas de TRAIN são preservadas.

Os folds externos com thresholds diferentes são reconstruídos com a receita
original. Antes de pontuar F3, suas contagens com threshold F2 precisam reproduzir
as contagens anteriores. Folds sem mudança de threshold reutilizam suas métricas.
Assim, não há fits internos redundantes nem uso de TEST para escolher limiares.

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.threshold_f3
.\.venv\Scripts\python.exe scripts/execute_notebooks.py --notebook 03_model_metrics.ipynb
```

### Resultado do experimento F3 e decisão

F3 dá mais ênfase ao Recall que F2, mas ainda penaliza falsos alertas.
Na validação temporal externa dentro do TRAIN:

| Política de ajuste | Recall médio | Precision média | F3 médio |
|---|---:|---:|---:|
| F2 — referência principal | 79.41% | 72.24% | 0.7836 |
| F3 — experimento | 80.20% | 51.61% | 0.7499 |

O ganho de Recall foi de apenas 0.79 ponto percentual,
com queda de 20.63 pontos de Precision e piora do
próprio F3 externo. **F3 não apresentou vantagem validada nesta rodada;
F2 continua principal.** Os números são médias da política de ajuste:
cada fold escolhe seu limiar internamente. Não são resultados do TEST.

O threshold final F3 foi 0,50, salvo em artefato experimental separado.
O threshold principal F2 permanece 0,70. Nenhum modelo foi treinado ou
reavaliado para registrar esta decisão; os resultados de TEST foram preservados.

Resultados: [reports/f3_results.md](reports/f3_results.md).
Novo artefato: models/recall_f3_pipeline.joblib. O pipeline treinado é o mesmo;
o threshold e a indicação de objetivo passam a F3. O artefato F2 fica preservado.

O notebook 03 inclui a comparação F2/F3 em CV e a grade interna final.
O notebook 04 continua mostrando exclusivamente o TEST histórico com threshold
F2=0,70. **Não houve nova avaliação de TEST para F3.** O experimento com F3 foi
realizado após observar esse teste; uma confirmação independente requer outro período.
