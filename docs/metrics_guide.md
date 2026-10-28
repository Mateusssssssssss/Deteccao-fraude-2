# Guia das métricas e tabelas

Uma linha é uma transação. Classe positiva: **Class=1, fraude confirmada no dataset**.
Class=0 é transação legítima. O rótulo não distingue tipos de fraude; não existem
categorias como cartão clonado, identidade falsa etc. As categorias mostradas nos
relatórios são **faixas numéricas de features**, não modalidades de fraude.

## Como ler porcentagens

Um valor proporcional de 0,000190 equivale a **0,0190%**, não 0,000190%.
Por exemplo, quatro fraudes entre 21.029 transações correspondem a
100 × 4 / 21.029 ≈ **0,0190%**. A proporção original e o percentual são
duas representações do mesmo número; uma métrica percentual não deve ser
multiplicada por 100 novamente.

## EDA: fraude observada por faixa

Em reports/metrics/eda_feature_bins.csv:

| Coluna | Significado e denominador |
|---|---|
| feature | Feature numérica usada para definir o segmento, por exemplo V10. |
| bin | Intervalo de valores da feature. Em (a, b], a é excluído e b é incluído. Os limites exibidos são arredondados pelo pandas. |
| size | Número total de transações do TRAIN nessa faixa, antes de qualquer SMOTE. |
| fraud_count | Número de transações Class=1 nessa faixa. |
| legitimate_count | Número de transações Class=0 nessa faixa. |
| mean | fraud_count / size, em escala 0–1; nome antigo mantido por compatibilidade. |
| fraud_rate_pct | 100 × fraud_count / size: porcentagem de transações fraudulentas dentro da faixa. |
| share_of_train_frauds_pct | 100 × fraud_count / todas as fraudes do TRAIN: parcela das fraudes do TRAIN nessa faixa. |
| source | Origem dos números; aqui sempre TRAIN original sem reamostragem. |

Exemplo: uma faixa pode ter taxa de fraude de 0,02% e ainda concentrar uma parcela
relevante das fraudes do TRAIN se tiver muitas transações. As duas colunas têm
denominadores diferentes. Dez quantis tentam produzir dez grupos de tamanho
parecido; valores repetidos podem reduzir o número de grupos. Ausentes e
features constantes recebem segmentos explícitos. Cada feature particiona o
TRAIN separadamente: **não some linhas de features diferentes**, pois as mesmas
transações reaparecem em cada análise.

Em eda_time_prevalence.csv e eda_interaction.csv, size é contagem do grupo e
mean é proporção de fraude (multiplicar por 100 para exibir em %). As faixas
temporais e os cruzamentos são exploratórios, sem interpretação causal.

## Matriz de confusão: contagens de decisões do modelo

O threshold converte probabilidade prevista em alerta: probabilidade >= threshold
gera classe 1. Essas contagens precisam estar associadas ao conjunto e threshold.

| Sigla | Significado |
|---|---|
| TP | Fraude real que gerou alerta: fraude detectada. |
| FN | Fraude real sem alerta: fraude não detectada. |
| FP | Transação legítima que gerou alerta: falso alarme. |
| TN | Transação legítima sem alerta. |

Taxa de fraude observada não é taxa de erro do modelo: é uma característica
dos rótulos. Um falso positivo requer uma previsão, além do rótulo.

## Métricas de classificação

| Métrica | Fórmula/interpretação | Preferência |
|---|---|---|
| Recall / sensibilidade | TP / (TP+FN): dentre todas as fraudes reais, qual fração foi detectada? | Maior; reduz FN. |
| Precision / precisão dos alertas | TP / (TP+FP): dentre os alertas, qual fração era fraude real? | Maior; reduz alertas inúteis. |
| F1 | 2 × Precision × Recall / (Precision+Recall); média harmônica dos dois. | Maior; equilíbrio simétrico. |
| F2 | 5 × Precision × Recall / (4×Precision+Recall); dá maior ênfase ao Recall. | Maior; objetivo das rodadas posteriores. Não representa um custo financeiro conhecido. |
| Specificity / especificidade | TN / (TN+FP): fração de legítimas corretamente liberadas. | Maior. |
| Balanced Accuracy | (Recall+Specificity)/2; média das sensibilidades das duas classes. | Maior. |
| Accuracy / acurácia | (TP+TN)/(TP+TN+FP+FN); proporção total de acertos. | Pode ser enganosa em classe rara. |
| FPR / taxa de falsos positivos | FP/(FP+TN): entre legítimas, qual fração foi alertada? | Menor. |
| FNR / taxa de fraude não detectada | FN/(FN+TP) = 1−Recall. | Menor. |
| Parcela de alertas falsos | FP/(FP+TP) = 1−Precision; não confundir com FPR. | Menor. |
| Taxa de alertas | (TP+FP)/n: fração de transações encaminhadas para revisão/bloqueio. | Depende da capacidade operacional e custo de fraude. |

As tabelas do modelo usam escala 0–1 salvo sufixo _pct, que indica escala
0–100. Nos notebooks, Precision, Recall, F1 e F2 são exibidos como porcentagens.
Para métricas agregadas do código sklearn, zero_division=0 pode gerar score zero
quando não há alertas. Nas tabelas por segmento, denominador zero fica vazio/NaN:
isso significa **não definido**, não desempenho nulo.

## Qualidade dos scores e probabilidades

| Métrica | Como interpretar |
|---|---|
| Average Precision (AP) | Resumo de Precision ao aumentar Recall sobre scores contínuos, ponderado pelo ganho de Recall. Usa probabilidades/scores, não rótulos. Maior é melhor. Referência ingênua próxima à prevalência da fraude. |
| Curva Precision–Recall | Eixo X: Recall; eixo Y: Precision. Cada ponto corresponde a um threshold. Mudar só o threshold escolhe outro ponto da mesma curva. |
| AUPRC trapezoidal | Área por integração trapezoidal da curva PR; pode diferir de AP. Neste projeto o resumo principal persistido é AP, indicado explicitamente nas legendas. |
| ROC-AUC | Capacidade de ordenar fraude acima de legítima, resumindo TPR contra FPR. Maior é melhor, mas pode parecer alto mesmo com muitos falsos alertas em classe rara. |
| Log Loss | Penaliza probabilidades erradas, especialmente erros confiantes. Menor é melhor. |
| Brier score | Média do erro quadrático entre probabilidade prevista e rótulo 0/1. Menor é melhor; não comprova calibração por si só. |
| Prevalência | Número de fraudes / total; descreve o conjunto, não habilidade do modelo. |

## CV, OOF e threshold

TRAIN contém o período inicial e é o único usado para escolhas de modelagem.
Cada fold temporal ajusta no passado e valida em uma janela posterior. **Média**
de CV é a média aritmética das métricas dos folds; cada fold tem o mesmo peso,
mesmo com tamanhos diferentes. **Desvio-padrão** descreve dispersão entre folds,
não um intervalo de confiança nem prova de significância.

OOF agrega previsões em janelas não usadas no ajuste dos respectivos modelos.
Só parte do TRAIN tem previsão OOF temporal: o prefixo inicial serve só para
treinar. Métrica OOF agregada não precisa coincidir com a média dos folds.

O threshold é o limiar de alerta. GridSearchCV de threshold usa janela interna
ao treino e F2 como objetivo. Outra janela externa avalia essa política de ajuste;
cada fold pode escolher um limiar diferente. O limiar final é escolhido novamente
no TRAIN, portanto os números externos avaliam a política, não esse limiar final
isoladamente. Receitas também são selecionadas com esses dados: os números do
vencedor podem ser otimistas. TEST histórico não é reutilizado nas rodadas novas;
uma avaliação em novo período continua necessária.

## Correlação, importância e regra de retirada

Pearson com Class mede associação linear ponto-bisserial e varia entre −1 e 1.
Usamos o **módulo** da correlação na regra, pois uma associação negativa forte
também pode prever fraude. Spearman mede associação monotônica por postos.

Importância por permutação é a **queda de AP** quando uma feature é embaralhada
numa janela de validação interna. Não é porcentagem nem soma obrigatoriamente 1.
Pode ser negativa por variação amostral, redundância ou ruído. Desvio das
permutações mede variação entre embaralhamentos, não incerteza populacional.

Importância interna da árvore (feature_importances_) usa outro cálculo e outra
escala; não substitui a queda de AP na regra definida nesta rodada.
Importância de árvores tem vieses e distribui crédito entre variáveis correlacionadas.

Regra: remover **somente se** abs(Pearson) < 0,090 **e** permutation_AP < 0,100.
Valores iguais ao corte são mantidos. Os cortes foram solicitados pelo usuário,
não descobertos como ótimos. O conjunto é reaprendido dentro de cada treino,
usando referência XGBoost numa validação interna para a importância. Isso evita
usar a janela externa na seleção. A ablação compara a regra com todas as features.
Correlação e importância não provam causalidade.

## Erros do modelo por faixa

Em category_error_metrics.csv, os limites dos quantis são aprendidos no treino do
fold e aplicados à validação externa; extremos são estendidos para cobrir valores
futuros. Não são as mesmas fronteiras da EDA do TRAIN completo.

Cada linha identifica fold, feature e bin, com count, fraud_count, legitimate_count,
TP, FP, FN e TN. fraud_rate_pct, recall_pct, precision_pct, false_positive_rate_pct,
missed_fraud_rate_pct, false_alert_share_pct e alert_rate_pct seguem as fórmulas
acima ×100. Taxas instáveis em faixas com poucas fraudes devem ser lidas junto
com as contagens; não se deve concluir bom desempenho a partir de 1 ou 2 eventos.

## Scaling, balanceamento e outros diagnósticos

StandardScaler transforma cada coluna em (x−média_treino)/desvio_treino. Média e
desvio são ajustados no treino do fold. Não limita valores a [0,1], não remove
outliers e pode ser sensível a eles. Árvores normalmente mudam pouco com scaling;
SMOTE usa distâncias e é sensível às escalas.

SMOTE cria exemplos numéricos sintéticos da classe minoritária somente no treino.
sampling_strategy=0,1 significa n_fraudes/n_legítimas = 0,1 depois da amostragem,
não 10% do total; equivale a cerca de 9,09% do total quando há duas classes.
Os dados sintéticos não são fraudes reais novas. Validação mantém a distribuição
original. SMOTENC é para features numéricas e categóricas, não para um target
categórico quando todas as entradas são numéricas.

Cardinalidade é número de valores distintos. Skew é assimetria. Outliers por IQR
ficam abaixo de Q1−1,5×IQR ou acima de Q3+1,5×IQR; são sinais de inspeção, não
exclusão automática. Na diversidade OOF, disagreement é fração de rótulos
previstos diferentes; double-fault é fração em que ambos os modelos erram.
Baixa double-fault ou alta discordância, sozinhas, não provam ganho de ensemble.

Fontes técnicas:
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html
- https://scikit-learn.org/stable/modules/model_evaluation.html


## F3

F3 = 10 × Precision × Recall / (9 × Precision + Recall)
= 10TP/(10TP+9FN+FP). Favorece Recall ainda mais que F2, mas não maximiza Recall
isoladamente e não representa um custo financeiro conhecido. Em empate, a grade
deste projeto prioriza Recall, depois Precision e maior threshold. Aumentar beta
não garante ganho no TEST. A nova política F3 é validada em janelas de TRAIN
externas ao ajuste do threshold, condicionada à receita escolhida anteriormente.
