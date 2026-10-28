# Protocolo e limites

Unidade: uma transação. Target positivo: Class=1 (fraude). Momento: autorização,
assumindo que V1–V28 e Amount já estejam disponíveis. Time é tempo decorrido da
primeira transação: em produção seria necessário um marco temporal consistente.
Os componentes PCA vieram prontos: não temos dados ou metadados para auditar se
sua transformação original foi ajustada em todo o conjunto.

O arquivo original permanece intacto. Cópias exatas de registros são removidas de
forma determinística antes da separação, mantendo apenas uma ocorrência; Time faz
parte da identidade. Não se deduplicam transações por V1–V28/Amount isoladamente,
pois podem ser transações distintas. Sem cartão/cliente, não é possível auditar grupos.

O último quinto da duração observada fica reservado ao TEST, e não necessariamente
20% das linhas. Isso reserva uma janela futura com milhares de transações, mantendo
o maior período inicial para CV expansiva em três janelas. Empates de Time ficam
juntos. O corte usa somente o relógio, sem otimizar sobre rótulos. Cada janela deve
conter ambas as classes; o código interrompe se isso não acontecer.

Dados de apenas dois dias não permitem demonstrar estabilidade de longo prazo.
A base já foi explorada pelo projeto antigo: esta correção é uma reavaliação
retrospectiva e não torna os dados historicamente intocados.

Comparação pequena, sem busca de hiperparâmetros: Random Forest e XGBoost, cada um
com e sem under+SMOTE, mais DummyClassifier como referência sem capacidade preditiva.
Não foi adicionado um terceiro modelo treinável; regressão logística fica como
próximo baseline interpretável se o escopo for ampliado. RF sem amostragem usa
balanced_subsample; RF reamostrada usa pesos uniformes. XGBoost usa pesos uniformes
em ambas. Assim, comparamos receitas completas, não isolamos causalmente o efeito
da amostragem. Configurações, semente e versões ficam persistidas.

Under ratio=0.1 significa n_minoritária/n_majoritária após undersampling;
SMOTE ratio=0.6 aumenta essa razão para 0.6. Não significa reduzir a maioria a
10% da minoria nem criar 1.5 vezes mais fraudes que não fraudes.
Scaling serve às distâncias do SMOTE; é dispensado nas receitas de árvores sem SMOTE.
Imputação, scaling e reamostragem são ajustados em cada treino de fold por
imblearn.Pipeline. Validações e TEST mantêm prevalência não reamostrada.
Nenhuma feature é eliminada pela correlação observada.

Seleção: maior Average Precision média entre folds, inclusive versus Dummy.
AP usa probabilidades e não é a integral trapezoidal da curva PR.
Threshold=0.5 é fixo em todas as receitas; não foi otimizado nem calibrado.
AP é primária pela raridade da fraude; Recall, Precision e F1 quantificam decisões.
Sem custos de FN/FP ou capacidade de revisão, não se alega threshold operacional ótimo.

CV usada para comparar receitas tende a ser otimista para o vencedor. OOF temporal
cobre apenas as janelas de validação (o prefixo inicial não recebe previsão);
a métrica OOF agregada difere da média dos folds. Mesmos folds para todas as receitas.
Dispersão entre três folds é descritiva e não um intervalo de confiança.

Correlação é calculada no TRAIN e permutação na última validação do TRAIN usando
o estimador que não treinou nessa janela. A escolha do modelo já usou essa janela;
importância é diagnóstico condicionado à seleção, não confirmação independente.
Importância de árvore tem vieses e divide crédito entre features correlacionadas.
Escalas de correlação e queda de AP não são comparáveis; nenhuma implica causalidade.

Curvas XGBoost usam iteration_range no modelo já ajustado na última janela.
Não há staged_predict no estimador utilizado. O número de árvores foi prefixado.
Early stopping seria útil numa busca maior, mas exigiria validação interna por fold,
transformada pelo pré-processador treinado no subtreino. Não usamos TEST como eval_set.
Curvas entre folds misturam tamanho crescente e mudança temporal; não são learning
curves controladas. Gini, profundidade e min_samples_leaf regularizam RF; L1/L2,
profundidade, taxa e subsampling controlam XGBoost. Não houve evidência comparativa
para trocar critério, podar, reduzir dimensão ou adicionar ensembles agora.

Diversidade usa OOF alinhado: disagreement, double-fault e acertos/erros conjuntos.
Divergência não prova ganho de ensemble. Voting/Stacking exigiriam nova validação
dentro do TRAIN e novo TEST independente se houver iteração após o resultado final.

TEST avalia somente o vencedor congelado e produz erro por faixa temporal.
Não há intervalo de incerteza formal nesta primeira execução; bootstrap i.i.d.
ignoraria dependência temporal e apenas dois dias não sustentam inferência robusta.
Uma avaliação prospectiva com mais períodos é necessária para isso.
Não foram demonstradas prontidão de produção, calibração ou economia financeira.

Fontes técnicas:
- https://imbalanced-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html


## Rodada posterior: GridSearchCV de threshold

A descrição acima documenta a primeira rodada, com threshold fixo. A rodada
posterior mantém a receita xgboost__none e varia somente o threshold numa grade
explícita, com F2 (beta=2) escolhido pelo usuário para enfatizar Recall.

Dentro de cada treino de fold externo, separa-se o quinto final da duração
como janela interna de threshold. O pipeline aprende apenas no prefixo.
GridSearchCV pesquisa threshold em FixedThresholdClassifier com FrozenEstimator,
pontuando apenas na janela interna. O estimator congelado evita que o refit do
wrapper treine sobre a janela que escolhe o limiar. Após a escolha, o pipeline
é reajustado no treino externo inteiro e avaliado na janela externa posterior.
Empates de F2 usam Recall, depois Precision e maior threshold.

Ao final, repete-se a escolha interna no TRAIN inteiro e faz-se refit do pipeline
no TRAIN completo, salvando recall_pipeline.joblib separadamente. Não se usam
valores de TEST para otimização ou avaliação nesta rodada. A CV externa mede a
política de threshold, não o threshold final específico; os limiares por fold
podem variar. A seleção prévia da receita usou os mesmos folds, portanto a
estimativa está condicionada à receita escolhida e não elimina viés de seleção
da primeira etapa. Uma nova base/período externo continua necessário.

A grade inclui 0 para mostrar a solução degenerada de Recall puro: marcar tudo
como positivo. A política adotada é F2; não foi introduzida uma restrição arbitrária
de Precision ou volume de revisão. F2 não é um custo financeiro estimado.

Referência:
https://scikit-learn.org/stable/modules/classification_threshold.html
