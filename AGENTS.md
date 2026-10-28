# AGENTS.md — Padrão para projetos de Ciência de Dados, Machine Learning e IA

**Atualizado em:** 27/10/2026

> Instruções permanentes para assistentes de programação (incluindo Codex) ao atuar neste repositório.
> Base arquitetural: **Cookiecutter Data Science** para organização e **Made With ML** para qualidade, experimentação, reprodutibilidade e evolução rumo à produção.

## 1. Papel e princípios

Atue como engenheiro de Machine Learning e cientista de dados experiente. Entregue soluções tecnicamente corretas, legíveis, reproduzíveis, testáveis e compatíveis com o escopo do projeto. **Sugira técnicas relevantes antes de introduzir complexidade**. Justifique escolhas pela natureza dos dados, pelo objetivo do negócio e por evidências de validação.

- Primeiro entenda problema, unidade de observação, variável-alvo, momento da previsão, disponibilidade real de cada feature e custo dos erros.
- Para classificação, explicite qual classe é a positiva e qual métrica orienta as decisões.
- Não escolha algoritmo, proporção de split, encoder, métrica ou threshold por hábito; estabeleça um baseline e compare alternativas quando houver benefício.
- Diferencie resultados de treino, validação cruzada, conjunto de validação e teste final.
- Não invente resultados, parâmetros ótimos, gráficos ou conclusões sem executá-los.
- Evite dependências e camadas arquiteturais desnecessárias para projetos pequenos.
- Preserve os dados originais; não versionar credenciais, dados pessoais ou bases sensíveis.

## 2. Arquitetura de referência

```text
projeto/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example                 # se necessário; nunca incluir segredos
├── Makefile                     # opcional
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── experiment.yaml
├── data/
│   ├── raw/                     # dados de origem, imutáveis
│   ├── interim/                 # transformações intermediárias
│   ├── processed/               # derivados; rastrear sua origem
│   └── external/                # opcional
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb    # baseline, CV e experimentos
│   ├── 03_model_metrics.ipynb     # obrigatório: métricas e diagnóstico
│   └── 04_error_analysis.ipynb    # opcional; análise detalhada
├── src/
│   └── projeto_ml/
│       ├── __init__.py
│       ├── data.py              # leitura, schemas e divisão
│       ├── preprocess.py        # pipelines; X/y
│       ├── features.py          # engenharia de atributos
│       ├── model.py             # criação dos estimadores
│       ├── validation.py        # CV, OOF, tuning e threshold
│       ├── evaluation.py        # métricas, gráficos e relatório final
│       ├── train.py             # orquestração do treinamento
│       └── predict.py           # inferência
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_validation.py
│   └── test_evaluation.py
├── models/                       # artefatos produzidos; política de versionamento
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── experiments/
└── docs/                         # opcional
```

Adapte nomes e pastas ao projeto. Código reutilizável fica em `src/`; notebooks servem para exploração e apresentação, importando funções dos módulos. Prefira imports absolutos, caminhos relativos à raiz via `pathlib`, funções pequenas e configurações explícitas. Use `pyproject.toml` como fonte principal de dependências; `requirements.txt` pode existir por compatibilidade.

## 2.1. Notebooks obrigatórios: EDA e métricas (todos os projetos)

**Sempre criar e manter pelo menos dois notebooks executáveis**, destinados à inspeção humana dos resultados, em todos os projetos de dados/ML em que haja dados para analisar e modelos para avaliar:

1. **`notebooks/01_eda.ipynb` — Análise Exploratória:** descrição e origem dos dados; dimensões, tipos, dicionário de features, duplicatas, ausências e inconsistências; distribuição das variáveis; outliers; análise de classes quando houver target; gráficos de associações/correlações e hipóteses. Explorar relações com target **somente no TRAIN** para orientar modelagem. No não supervisionado, explorar formas, escalas e possíveis grupos sem inventar rótulos.
2. **`notebooks/03_model_metrics.ipynb` — Métricas e resultados:** carregar artefatos já calculados por scripts de `src/`, exibir uma tabela-resumo por modelo e gráficos comparativos; apresentar resultados dos folds de CV (média e dispersão), baseline, escolhas de hiperparâmetros/threshold, curvas de aprendizado/validação quando úteis, inspeção de erros e conclusão com limitações. Em classificação com 2+ modelos, incluir **gráfico obrigatório de barras agrupadas de F1, Recall e Precision**; para um único modelo, mostrar essas métricas em tabela e gráficos diagnósticos pertinentes. Para regressão, MAE/RMSE/R² e resíduos; para clustering, métricas apropriadas (por exemplo, silhouette e Davies–Bouldin, quando válidas), tamanhos/perfis dos clusters e estabilidade. Os resultados do TEST só aparecem em seção final própria, após congelada a seleção do modelo.

**Regras de implementação:**

- Notebooks devem ter títulos, seções em Markdown, interpretações objetivas junto dos gráficos, e células em ordem reproduzível usando caminhos relativos à raiz.
- **Não concentrar lógica de produção em notebooks:** criar cálculos, pré-processamento, treinamento e métricas reutilizáveis em `src/`; notebooks chamam essas funções ou leem resultados persistidos em `reports/metrics/` e figuras em `reports/figures/`.
- Diferenciar claramente **TRAIN, CV/OOF, validação e TEST** em tabelas e figuras. Não misturar métricas nem usar TEST para seleção, tuning, feature engineering ou escolha do threshold.
- Não fabricar números ou gráficos: se dados ou artefatos estiverem ausentes, explicar como reproduzir e não fingir execução. Evitar trabalho pesado repetido ao abrir os notebooks; usar artefatos versionados quando cabível.
- Incluir no README instruções para executar e abrir ambos os notebooks. Se o projeto for apenas exploratório, sem modelo, exigir EDA e justificar a inexistência de notebook de métricas; se for só inferência sem dados de avaliação, documentar a limitação.

## 3. Fluxo obrigatório: dados → treino → teste isolado

1. Defina target, horizonte temporal, unidade de observação, critérios de inclusão, potenciais vazamentos e esquema de dados.
2. Faça um split inicial **TRAIN / TEST** antes de operações que aprendem estatísticas ou padrões dos dados. Para séries temporais use separação cronológica; para pacientes/clientes repetidos use separação por grupo; estratifique quando apropriado.
3. Use **somente TRAIN** para EDA que influencia decisões de modelagem, pré-processamento aprendido, seleção de features, validação cruzada, busca de hiperparâmetros, resampling, calibração e escolha do threshold.
4. Dentro do TRAIN, execute Cross Validation, inclusive em pesquisas de hiperparâmetros ou validação interna quando necessário. Use pipeline para ajustar transformações dentro de cada fold.
5. Escolha modelo, hiperparâmetros e threshold com estimativas de validação; se um mesmo conjunto for reutilizado extensivamente, considere CV aninhada ou holdout de validação para reduzir viés de seleção.
6. Reajuste o pipeline final no TRAIN completo, respeitando a estratégia de threshold/calibração escolhida. **Nunca reajuste threshold com TEST.**
7. Avalie o pipeline uma vez no TEST, reporte métricas e intervalos de incerteza quando viável. Se retornar ao tuning após vê-lo, deixe claro que o TEST deixou de ser avaliação independente e reserve outro teste.

**Regra:** `GridSearchCV` procura a melhor combinação na grade; `Cross Validation` estima desempenho entre folds; `cross_val_predict` gera predições out-of-fold (OOF) e **não substitui** a avaliação final em dados nunca vistos. Fazer CV depois de escolher hiperparâmetros com as mesmas amostras pode produzir avaliação otimista.

### Seleção do esquema de validação

- `StratifiedKFold` para classificação i.i.d. em que o balanceamento por fold é apropriado.
- `KFold` para regressão i.i.d.; `GroupKFold`/equivalentes quando há participantes, pacientes ou entidades repetidas.
- `TimeSeriesSplit`, janela temporal ou backtesting para dados ordenados no tempo, evitando previsões que usam o futuro.
- Em saúde, prevenir vazamento entre registros do mesmo paciente e entre instituições/períodos conforme objetivo de generalização.
- Não imponha `60/20/20`, `80/20` ou outra proporção como regra universal; considere tamanho da base, raridade da classe, grupos e tempo.

## 4. EDA e qualidade dos dados

Gerar relatório exploratório reproduzível em `notebooks/01_eda.ipynb` e salvar figuras significativas em `reports/figures/`:

- Tipo, cardinalidade, distribuição, unidade, valores faltantes, duplicatas, inconsistências, outliers e plausibilidade de cada feature.
- Distribuição do target; frequência e proporção de classes; prevalência por período e subgrupo quando houver base para isso.
- Distribuições e relações feature–target com gráficos adequados: histogramas/KDE, barras, box/violin, scatterplots, hexbin e taxas do desfecho por faixas para classificação.
- Examinar **linearidade, não linearidade, formato quadrático, monotonicidade, interações e comportamento por segmentos**. Use smoothers ou ajustes exploratórios quando úteis, sem interpretar forma visual como prova causal.
- Para categóricas, avaliar taxas do target e contagens por categoria, respeitando grupos de tamanho suficiente.
- Examinar associações entre features e redundância/multicolinearidade quando pertinente.

### Correlação × importância das variáveis — exigência padrão

Sempre que fizer sentido, analisar **(a)** associação entre feature e target e **(b)** importância atribuída pelo modelo, exibindo **gráfico comparativo por feature** e tabela de valores. Dar preferência a dois painéis lado a lado, ou eixos claramente identificados, pois as escalas não são diretamente comparáveis.

- Features numéricas: Pearson para associação linear; Spearman para associação monotônica; outras métricas para relações não lineares, conforme o caso.
- Features categóricas ou target categórico: escolher associação apropriada (por exemplo, Cramér's V, informação mútua ou análise por grupos); evitar Pearson de códigos categóricos arbitrários.
- Importância: preferir permutation importance em validação ou teste ainda não usado para decisões, conforme finalidade; para árvores, reportar também `feature_importances_` com aviso de vieses possíveis; avaliar SHAP quando justificar custo e interpretação.
- Transformações One-Hot podem gerar várias colunas por feature original; agregar importâncias quando apropriado e documentar o método.
- **Correlação alta não implica causalidade ou importância preditiva; importância alta não implica correlação linear alta.** Efeitos conjuntos, colinearidade e importâncias condicionais afetam as leituras.
- Não remover uma variável apenas por baixa correlação, correlação alta com outra feature ou importância baixa: testar a ablação com CV no TRAIN e comparar desempenho/estabilidade.

## 5. Pré-processamento e data leakage

- Use `Pipeline` e `ColumnTransformer` para encoders, imputação, scaling, seleção e estimadores.
- `OneHotEncoder(handle_unknown='ignore')` costuma ser baseline seguro para categóricas nominais de cardinalidade moderada; avaliar ordinal encoding apenas quando justificável, além de codificação nativa ou outras estratégias conforme modelo/cardinalidade.
- `StandardScaler` padroniza média e desvio-padrão, útil para métodos sensíveis à escala (regressões regularizadas, SVM, KNN, redes neurais); árvores geralmente não exigem scaling.
- Para valores ausentes, ajustar imputadores apenas no fold de treino. Criar indicadores de ausência quando houver hipótese útil.
- Reamostragem para classes desbalanceadas (`RandomOverSampler`, `SMOTE`, `SMOTENC`, undersampling) somente no treino de cada fold, preferindo pipeline do `imbalanced-learn`; comparar com class weights e ajuste de threshold.
- Investigar **leakage temporal, pós-desfecho, target encoding sem cross-fitting, identificadores, variáveis proxies e duplicatas entre splits**.
- Documentar o momento em que cada dado estaria disponível na inferência real.

## 6. Baselines, modelos e hiperparâmetros

- Estabeleça baseline ingênuo e modelo simples interpretável antes de técnicas complexas.
- Para classificação, considerar regressão logística, árvore CART, Random Forest, ExtraTrees, HistGradientBoosting, GradientBoosting, XGBoost e modelos complementares quando pertinentes.
- Para regressão, considerar modelos lineares, regularizados, árvores, florestas e boosting conforme estrutura dos dados.
- Usar `GridSearchCV` para grades pequenas e interpretáveis; `RandomizedSearchCV` ou otimização mais eficiente para espaços extensos. Explicitar `scoring`, folds, intervalo pesquisado e seed.
- Testar ablação de features e sensibilidade a hiperparâmetros com CV no TRAIN.

### Regularização

- **L1 / Lasso:** penaliza magnitude absoluta dos coeficientes; pode zerá-los e induzir esparsidade.
- **L2 / Ridge:** penaliza quadrados dos coeficientes; contrai coeficientes, geralmente sem zerá-los.
- **Elastic Net:** mistura L1/L2; útil a considerar em certas estruturas de correlação.
- XGBoost: `reg_alpha` = L1 e `reg_lambda` = L2. Valor `0` desativa a respectiva penalização. Valores como `0.5` são possíveis, mas devem ser validados; não significam universalmente a mesma intensidade entre modelos.

## 7. Árvores CART: Gini, entropia e log loss

Ao utilizar árvores ou florestas para **classificação**, lembrar que CART divide nós buscando reduzir impureza; comparar critérios somente quando houver justificativa:

- `criterion='gini'`: impureza de Gini; padrão de `DecisionTreeClassifier`/`RandomForestClassifier` no scikit-learn, um ótimo baseline.
- `criterion='entropy'`: ganho de informação baseado em entropia.
- `criterion='log_loss'`: nas árvores de classificação do scikit-learn, critério ligado ao ganho de informação/entropia; **não confundir com avaliar probabilidades previstas pela métrica Log Loss fora da árvore**.
- `max_depth` limita profundidade máxima, não quantidade fixa de perguntas em todo caminho; o crescimento pode parar antes por pureza, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes` etc.
- `min_samples_leaf` define mínimo de amostras em cada folha; pode reduzir variância e sobreajuste.
- Avaliar também `ccp_alpha` (poda), `max_features` e outros controles quando apropriado.
- Não assumir que trocar Gini por entropia/log_loss aumentará o desempenho; comparar por CV e objetivo.

## 8. Boosting: AdaBoost, Gradient Boosting e XGBoost

- **AdaBoost:** repondera exemplos difíceis entre aprendizes sequenciais (dependendo da implementação).
- **Gradient Boosting:** adiciona modelos que reduzem uma função de perda seguindo uma direção de otimização; não se resume a dar maior peso aos erros como o AdaBoost.
- `learning_rate` menor reduz a contribuição de cada árvore (não a aumenta), podendo exigir mais estimadores.
- `n_estimators`, `max_depth`, `learning_rate`, regularização e subsampling interagem; ajustar em conjunto quando fizer sentido.

### Escolher número de árvores e diagnosticar generalização

Quando o modelo oferecer, sugerir `staged_predict()` para métricas por iteração e `staged_predict_proba()` para Log Loss/AUPRC etc. Calcular curvas **TRAIN versus validação**, destacar a iteração com melhor métrica de validação e usar o número correspondente de árvores somente após validar a estratégia. Evitar observar TEST para decidir `n_estimators`.

- `warm_start=True` permite continuar/adicionar estimadores em chamadas de `fit` compatíveis, mas **não é necessário** para avaliar todas as etapas de um modelo já treinado com `staged_predict`; este geralmente é mais simples para essa finalidade.
- Recomendar **Early Stopping** quando viável: parar após falta de melhoria na validação, com `n_iter_no_change`, `tol` ou `early_stopping_rounds` conforme biblioteca. Em deep learning, usar `patience`, `min_delta`, e restaurar melhores pesos quando disponível.
- XGBoost binário: `objective='binary:logistic'` otimiza a função logística e fornece probabilidades; `eval_metric='logloss'` acompanha qualidade probabilística. Ajustar conjunto de avaliação sem contaminar TEST.
- Em avaliação temporal ou por grupos, os gráficos devem respeitar o tipo correto de split.

## 9. Diagnóstico de bias, variância e generalização

Incluir quando úteis:

1. **Learning Curve:** métrica de treino e validação contra tamanho de treino, para diagnosticar tendências de alto viés, alta variância e ganho potencial de dados adicionais.
2. **Validation Curve:** treino e validação contra um hiperparâmetro relevante (ex.: `max_depth`, `min_samples_leaf`, `C`).
3. **Curvas por iteração/época:** loss ou métrica de treino e validação ao longo de boosting ou redes neurais.
4. **Comparação de distribuição dos erros:** por classe, período e subgrupo quando aplicável.

Interpretação orientativa: erro elevado em ambos os conjuntos pode sinalizar **underfitting/alto viés**; diferença grande treino–validação pode sinalizar **overfitting/alta variância**. Diminuir variância pode melhorar generalização, mas pode aumentar viés; otimizar erro validado em dados representativos, não apenas tornar curvas próximas.

## 10. Ensemble Learning e diversidade dos erros

Considere ensembles quando modelos individuais cometem erros complementares **e** a combinação oferece ganho validado:

- `VotingClassifier(voting='hard')`: voto majoritário por rótulos.
- `VotingClassifier(voting='soft')`: média de probabilidades; avaliar calibração e, se relevante, pesos dos estimadores.
- `StackingClassifier`: meta-modelo treinado com previsões fora da amostra; usar CV apropriada, impedindo leakage entre modelos base e meta-modelo.
- `RandomForest` e `ExtraTrees`: diversificação de árvores com estratégias distintas; comparar também com Voting e Stacking quando justificado.
- **Bagging**: normalmente bootstrap com reposição (`bootstrap=True`); **Pasting**: subamostragem sem reposição (`bootstrap=False`). São estratégias de amostragem, não sinônimos universais de Random Forest.
- **OOB**: com bootstrap, amostras não selecionadas para uma árvore permitem estimativa out-of-bag (cerca de 36,8% não selecionadas *por árvore*, em expectativa, quando se sorteia `n` vezes de `n`). `oob_score=True` pode ajudar na avaliação de bagging/florestas, mas **não substitui automaticamente CV adequada, tuning cuidadoso nem TEST final**; não funciona para `bootstrap=False`.

### Procedimento para medir erros complementares

1. Obter previsões **OOF alinhadas por amostra** de cada modelo, com folds coerentes e sem leakage, exclusivamente em TRAIN.
2. Avaliar métricas individuais e a correlação/associação entre erros; calcular `disagreement rate` (discordância das previsões), `double-fault` (proporção de exemplos em que ambos erram), matriz de acertos/erros conjuntos e, se útil, diversidade de probabilidades.
3. Priorizar pares/modelos que combinem **qualidade individual e erros complementares**; discordância isolada não prova que o ensemble é melhor.
4. Comparar Hard/Soft Voting, Stacking e Bagging com os baselines por CV, usando as métricas do objetivo; avaliar calibração e latência.
5. Selecionar ensemble somente diante de ganho consistente frente a custo, complexidade e manutenção.

## 11. Avaliação e métricas

### Classificação

- Apresentar matriz de confusão (`TN`, `FP`, `FN`, `TP`) e métricas condizentes com custo de erros: Precision, Recall, F1, Specificity, Accuracy/ Balanced Accuracy.
- Para classe rara, enfatizar **Recall, Precision, F1 e Average Precision (AP/AUPRC)**; apresentar prevalência como baseline da PR curve.
- `scoring='recall'` em classificação binária com classe positiva definida; usar variantes `recall_macro`, `recall_weighted` ou scorer customizado conforme problema.
- ROC-AUC pode complementar PR-AUC, mas não substitui avaliação da classe positiva rara.
- **Log Loss** mede qualidade das probabilidades e penaliza previsões confiantes incorretas; usar também Brier score ou curvas de calibração quando probabilidades importarem.
- Explicitar nível do threshold utilizado; a decisão de threshold não é a mesma tarefa que otimização dos hiperparâmetros.

### Comparação visual de modelos — gráfico obrigatório em classificação

Quando o projeto comparar **dois ou mais classificadores**, criar um **gráfico de barras agrupadas** com **três barras para cada modelo: F1-score, Recall e Precision**, identificando claramente os modelos no eixo X e as métricas na legenda. Para um único classificador, o gráfico é opcional.

- Calcular as métricas sob **mesma estratégia de validação, mesmos folds/dados, mesma definição de classe positiva e mesma política explícita de threshold**, para permitir comparação justa. Se thresholds forem distintos, documentá-los.
- Usar preferencialmente os **resultados de validação cruzada no TRAIN**: uma barra por média de F1, Recall e Precision por modelo, opcionalmente acompanhada de barras de erro (desvio-padrão entre folds). Se usar OOF, rotular explicitamente como OOF e não confundir a métrica agregada com a média dos folds.
- Usar `cross_validate` com múltiplos scorers (`f1`, `recall`, `precision`, ou variantes `macro`/`weighted` para multiclasse). Para multiclasse, explicitar a estratégia de agregação; não usar o padrão binário inadvertidamente.
- Usar Matplotlib ou Seaborn, com nomes legíveis, cores diferenciadas por **métrica**, eixo Y de 0 a 1 (quando aplicável), título, legenda e rótulos numéricos (ex.: 0,84). Ordenar os modelos segundo a **métrica principal do problema**, nunca apenas pela estética.
- Salvar `reports/figures/model_comparison_f1_recall_precision.png` e uma tabela em `reports/metrics/model_comparison.csv` contendo modelo, F1, Recall, Precision, origem das métricas e, se aplicável, dispersão, configuração e threshold.
- Não misturar scores de treino, validação e TEST no mesmo agrupamento sem distinção inequívoca. **Não usar o TEST para escolher o melhor modelo**, nem recalcular o gráfico de seleção com TEST.
- Depois de selecionar o modelo, relatar **separadamente** as métricas no TEST final (e, se fizer sentido, um gráfico final claramente identificado como teste); não reavaliar múltiplos candidatos no TEST para escolher o vencedor.
- Em problemas desbalanceados, complementar o gráfico com AP/AUPRC, matriz de confusão e análise do custo de falsos positivos/negativos, conforme objetivos do negócio.

### Threshold

- `GridSearchCV` busca hiperparâmetros, **não é por si só** o método recomendado para selecionar threshold.
- Para threshold tuning considerar `TunedThresholdClassifierCV` quando aplicável, com pontuação de interesse e CV interna; ou usar probabilidades OOF no TRAIN com procedimento sem vazamento.
- Nunca otimizar o threshold no TEST final. Se a estratégia exige calibração + threshold, alocar validação interna adequada e explicitar a sequência.
- Evitar avaliação otimista: reportar resultado com validação externa ao ajuste do threshold, se necessário via CV aninhada ou validação adicional.

### Regressão

- Considerar MAE, RMSE, R², resíduos, viés por faixas e estabilidade temporal. Não usar métricas de classificação em regressão.
- Escolher métrica primária conforme custo dos erros e tolerância a outliers.

## 12. Aprendizado não supervisionado e clusterização

Para projetos sem target, **não inventar rótulos** apenas para aplicar métricas de classificação. Formular pergunta de negócio e definir unidade de análise, critérios de qualidade e a utilidade esperada dos grupos.

- Explorar escala, assimetria, outliers, tipos de dados, distâncias e distribuições antes de escolher o algoritmo.
- Comparar, se pertinente, `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `HDBSCAN` (se disponível), clustering hierárquico e `GaussianMixture`, explicando hipóteses e diferenças entre clusters rígidos e probabilísticos.
- Para K-Means: explicar centróides, sensibilidade à escala e outliers, `n_init`, inicialização e número de grupos. Não interpretar cada cluster como categoria verdadeira por padrão.
- Para escolher `k`, utilizar curva do cotovelo (inércia), silhouette, Calinski-Harabasz e Davies-Bouldin quando apropriados; não selecionar `k` exclusivamente por um índice automático.
- Analisar tamanho, estabilidade, perfis, atributos característicos e interpretabilidade dos grupos; verificar robustez entre seeds, amostras e períodos. Visualizar projeções 2D com ressalvas.
- Ao combinar não supervisionado com supervisionado, deixar claro que o classificador aprende a **reproduzir os rótulos derivados do agrupamento**, não uma verdade externa. Construir splits e realizar o ajuste do clusterizador nos dados adequados para evitar contaminação.
- Se clusterização fizer parte de pipeline preditivo, ajustar escalonamento, PCA e clusterizador apenas dentro do TRAIN/folds; TEST continua intocado.

## 13. Redução de dimensionalidade e manifolds

- Considerar **PCA**, `TruncatedSVD`, `IncrementalPCA`, `PCA(svd_solver='randomized')`, **Kernel PCA**, **LLE**, UMAP/t-SNE (visualização exploratória, com cautela) somente quando houver benefício claro de desempenho, compressão, ruído, velocidade ou visualização.
- **PCA linear** encontra direções de maior variância, não necessariamente as de maior poder preditivo; não usá-lo automaticamente apenas porque features têm correlações altas, nem evitá-lo somente porque têm correlação baixa.
- Ajustar scaler e transformações aprendidas **apenas em TRAIN ou dentro de cada fold**. Usar `Pipeline`, comparar baseline sem redução versus com redução por CV na mesma estratégia de validação.
- Selecionar `n_components` por validação e/ou variância explicada, conforme objetivo; documentar trade-off entre compressão, interpretabilidade e métrica de negócio.
- `KernelPCA` e LLE podem capturar estruturas não lineares/manifolds, mas têm custos e limitações; comparar em dados reais, não assumir superioridade.
- `TruncatedSVD` pode ser apropriado para matrizes esparsas (ex.: texto). PCA com solver randomized acelera aproximações em cenários adequados e pode introduzir variação numérica.
- Para 3D→2D, explicar que a projeção pode preservar algumas relações e perder outras. Não concluir separação real de clusters ou valor preditivo apenas pela aparência do gráfico.
- `inverse_transform` reconstrói uma aproximação no espaço original quando o estimador disponibiliza o método; analisar erro de reconstrução apenas quando a tarefa o justificar. Nem todo método de manifold possui inversa exata ou implementada.

## 14. Deep Learning, imagens e aplicações end-to-end (quando pertinente)

- Para classificação de imagens (ex.: MNIST), dividir treino/validação/teste respeitando origem, identidade e duplicatas; transformações aleatórias de data augmentation **somente no treino**.
- Estabelecer baseline simples antes de CNNs e transfer learning; comparar desempenho com métricas por classe e matriz de confusão. Não reportar acurácia isoladamente se há desbalanceamento.
- Controlar seeds, versões, checkpoints e callbacks; avaliar `EarlyStopping` com validação apropriada e restauração dos melhores pesos quando desejável.
- Para APIs com FastAPI, carregar **o pipeline completo** treinado; validar schema e tipos de entrada, tratar erros, criar testes de predição e documentar endpoint, versão do modelo e limitações.
- Monitorar, quando houver implantação, drift de dados, drift de desempenho, calibração, latência e distribuição de predições; não alegar monitoramento real se não foi implementado.
- Para LLM, RAG e agentes, utilizar projetos e dependências específicos, com avaliações adequadas (retrieval, qualidade de respostas, segurança e custos). Não introduzir essa infraestrutura em projeto tabular tradicional sem necessidade.

## 15. Experimentos, documentação e reprodutibilidade

- Configurações em `configs/*.yaml` quando houver cenários comparáveis. Não criar YAML apenas por formalidade.
- Registrar dataset e versão, features, target, critérios de exclusão, `random_state`, estratégia de split, parâmetros, ambiente/dependências, métricas por fold e resultado final.
- Separar claramente métricas de validação e de teste, e salvar artefatos relevantes em `reports/`.
- Usar **MLflow** quando número de experimentos, colaboração ou necessidade de rastreabilidade justificarem.
- Usar sementes fixas quando possíveis e documentar fontes inevitáveis de não determinismo.
- Treinar/avaliar a partir de scripts reproduzíveis, não depender de estado oculto de notebook.
- README com objetivo, dados, modo de instalação, execução, validação, resultados, limitações e considerações de uso.

## 16. Qualidade de código e testes

- Escrever testes `pytest` para carregamento e schema, tratamento de ausências/categorias desconhecidas, consistência de transformações, comportamento de pipelines, avaliação e predição.
- Garantir que scripts funcionem com caminhos de projeto e que importações não iniciem treino automaticamente.
- Preferir type hints, docstrings, logging e mensagens de erro claras.
- Verificar se o pipeline de inferência utiliza **exatamente** o pré-processamento ajustado no treino.
- Validar com `python -m pytest`; quando possível incorporar lint/format (`ruff`) e CI simples.
- Se novas alterações afetarem esquema de features, atualizar testes, configs e documentação.

## 17. Segurança, ética e dados de saúde

- Observar LGPD e requisitos locais: minimização, necessidade, consentimento/base legal quando aplicável, segregação de acesso, desidentificação e armazenamento seguro.
- Não expor nomes, documentos, identificadores ou dados clínicos sensíveis em código, logs, commits, gráficos ou artefatos públicos.
- Em dados de saúde, considerar divisão por paciente/tempo/instituição, mudança de distribuição, qualidade dos rótulos, subgrupos e avaliação prospectiva.
- Não inferir causalidade a partir de feature importance, SHAP ou correlação.
- Evitar alegações de prontidão clínica, impacto ou segurança sem validação apropriada.

## 18. Checklist de implementação e comunicação (para Codex)

Ao receber um novo projeto ou solicitação:

- [ ] Explicar problema, target, momento da previsão e riscos de leakage.
- [ ] Propor arquitetura simples e extensível usando esta organização como referência.
- [ ] Separar TRAIN/TEST; justificar CV estratificada, temporal ou por grupos.
- [ ] Implementar baseline e pipeline sem leakage.
- [ ] **Entregar `notebooks/01_eda.ipynb` e `notebooks/03_model_metrics.ipynb` com gráficos, interpretações, comparações e execução reproduzível** (salvo projeto sem modelos/dados avaliáveis, quando justificar).
- [ ] Executar EDA que relacione features e target e investigue relações lineares/não lineares.
- [ ] **Comparar correlação/associação e importância**, incluindo gráfico com valores e observação sobre escalas.
- [ ] Avaliar utilidade de Gini, entropia, Log Loss, poda e regularização conforme modelos escolhidos.
- [ ] Se boosting: considerar `staged_predict`, curvas por iteração e Early Stopping para número ideal de árvores.
- [ ] Se necessário: realizar tuning por GridSearchCV/RandomizedSearchCV exclusivamente no TRAIN.
- [ ] Se necessário: comparar thresholds com CV/OOF; usar `TunedThresholdClassifierCV` quando apropriado.
- [ ] Produzir Learning Curves e/ou Validation Curves quando ajudam a diagnosticar generalização.
- [ ] Se houver vários modelos: medir diversidade de erros OOF e avaliar Voting/Stacking; considerar Bagging/Pasting/OOB quando justificável.
- [ ] Em projetos não supervisionados: avaliar perfis e estabilidade de clusters, com métricas apropriadas e sem criar target artificial.
- [ ] Considerar PCA/SVD/Kernel PCA/LLE apenas quando úteis; comparar com baseline sem redução dentro da validação.
- [ ] Priorizar métricas do problema; para desbalanceamento, considerar AP/AUPRC, Recall e Precision, além de calibração.
- [ ] **Se houver pelo menos dois modelos de classificação, gerar gráfico de barras agrupadas F1, Recall e Precision por modelo com resultados comparáveis de CV no TRAIN**, e salvar PNG e CSV em `reports/`.
- [ ] Executar testes, salvar figuras e métricas, documentar decisões e limites.
- [ ] Avaliar TEST apenas ao final, sem usá-lo para decisões de modelagem.

### Formato das respostas do agente

Ao recomendar técnica adicional, informar brevemente: **quando usar**, **qual problema resolve**, **custo/limitações**, **como validar sem leakage** e **se vale implementá-la agora**. Não ativar todas as técnicas indiscriminadamente. Ao entregar código, mostrar arquivos afetados, comandos para reproduzir e resultados reais de testes executados; se não foi possível executar, declarar isso explicitamente.

---

**Regra final:** prefira evidência de validação e simplicidade a complexidade sem ganho comprovado. Mantenha a avaliação final independente e o projeto reproduzível.
