import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import seaborn as sb
from notebooks.eda import *


#Verifica as colunas as 5 colunas com maiores correlação negativa e positiva
top_5_positivas = correlacao[1:6] 
top_5_negativas = correlacao[-5:]

# Lista das colunas com maiores correlações
columns_corr= ['V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V19', 'V2', 'V21', 'V4']

# Metodo de equilibrio entre amostras da classe majoritária e minoritária para que o 
# modelo não seja enviesado em favor da classe majoritária.

#RandomUnderSampler(sampling_strategy=0.1): Isso significa que a classe majoritária 
# será reduzida para 10% do tamanho original da classe minoritária
majoritaria = RandomUnderSampler(sampling_strategy = 0.1)

# SMOTE(sampling_strategy=0.5): aumentando a classe minoritária para 50% 
# do tamanho da classe majoritária, ou seja, a classe minoritária será aumentada para 
# ser 1,5 vezes maior que a classe majoritária após o balanceamento.
minioritaria = SMOTE(sampling_strategy=0.6)

#Matrix
previsores = dados[columns_corr].values
classes = dados.iloc[:, 30].values
#Ajusta por etapas
steps = [('maj', majoritaria),('min', minioritaria)]
pipeline = Pipeline(steps=steps)
previsores, classes = pipeline.fit_resample(previsores, classes)
#Divisão do modelo entre treino 70% e teste 30%
x_train, x_test, y_train, y_test = train_test_split(previsores, classes, test_size=0.3, random_state=1)

#Modelo usado
modelo = XGBClassifier(objective='binary:logistic',  # Classificação binária
    eval_metric='aucpr',            # Métrica de avaliação
    n_estimators=300,             # Número de árvores
    learning_rate=0.03,           # Taxa de aprendizado
    max_depth=10,                  # Profundidade das árvores
    subsample=0.5,                # Amostragem para evitar overfitting
    colsample_bytree=0.8,         # Porcentagem de colunas usadas
    gamma=1,                      # Evita overfitting
    reg_lambda=1,                 # Regularização L2
    reg_alpha=0,                   # Regularização L1
    scale_pos_weight=10,             # dá mais peso para a classe minoritária
    tree_method='auto',             
    min_child_weight=1
)

#Validação cruzada
results = cross_val_score(modelo, x_train, y_train, cv=5) 
print(f'Cross Validation: {results}')

#treinamento do modelo
modelo.fit(x_train, y_train)

#Previsao
previsao = modelo.predict_proba(x_test)[:, 1]
print(previsao)
#  converte as probabilidades preditas em rótulos de classe (0 ou 1), com 0.5 como o limite para a classificação. 
# Se a probabilidade da classe 1 for maior que 0.5, a amostra será classificada como fraude (1), caso contrário, como não fraude (0).
prev_prob = (previsao > 0.5).astype(int)


# Calcular AUPRC
# AUPRC é útil para modelos em dados desbalanceados e mede a área sob a curva de 
# precisão vs. recall. Quanto maior a AUPRC, melhor o modelo!
auprc = average_precision_score(y_test, prev_prob)
print(f'Auprc: {auprc}')


# Calcular a precisão e o recall para as previsões
precision, recall, _ = precision_recall_curve(y_test, previsao)

# Plotar a Curva de Precisão vs. Recall
plt.figure(figsize=(8,6))
plt.plot(recall, precision, color='blue', label='Curva de Precisão vs. Recall')
plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.title('Curva de Precisão vs. Recall')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# accuracy, macro avg, weighted avg
#accuracy: porcentagem de previsões do modelo que estavam corretas.

#macro avg: A média macro calcula a média das métricas de cada classe

#weighted avg: A média ponderada leva em consideração o número de amostras de cada classe, 
# ou seja, ela ajusta as métricas com base na proporção das classes no conjunto de dados.
scores = classification_report(y_test, prev_prob)
print(f'Scores: {scores}')

#Dados FP FV VP VF
confusion = confusion_matrix(y_test, prev_prob)
print(f'Confusão: {confusion}')

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sb.heatmap(confusion, annot=True, fmt='g', cmap='Blues', cbar=False, 
            xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])

# Adicionar título e rótulos
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()
