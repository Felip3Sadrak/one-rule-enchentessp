import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados
data = pd.read_csv('dados_alagamento.csv')

# Exibir as primeiras linhas do conjunto de dados
print("Dados Carregados:")
print(data.head())

# Definindo a coluna de risco e as características
features = ['infraestrutura_drenagem', 'tipo_solo', 'topografia']
target = 'risco'

# Converter as variáveis categóricas para numéricas
data_encoded = pd.get_dummies(data[features], drop_first=True)
data_encoded[target] = data[target]

# Dividir os dados em conjuntos de treino e teste
X = data_encoded.drop(target, axis=1)
y = data_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Função para calcular a regra única
def one_rule(X_train, y_train):
    best_rule = None
    best_accuracy = 0

    for feature in X_train.columns:
        # Agrupar dados pela feature e calcular a classe mais comum
        grouped = pd.Series(y_train).groupby(X_train[feature]).agg(lambda x: Counter(x).most_common(1)[0][0])
        
        # Calcular a acurácia
        predictions_train = X_train[feature].map(grouped)
        accuracy_train = accuracy_score(y_train, predictions_train)
        
        # Avaliar a acurácia no conjunto de teste
        predictions_test = X_test[feature].map(grouped)
        accuracy_test = accuracy_score(y_test, predictions_test)

        if accuracy_test > best_accuracy:
            best_accuracy = accuracy_test
            best_rule = (feature, grouped)

    return best_rule, best_accuracy

# Aplicar a regra única
best_rule, best_accuracy = one_rule(X_train, y_train)

# Exibir os resultados
print(f"\nMelhor Regra: {best_rule[0]}")
print(f"Acurácia no Conjunto de Teste: {best_accuracy:.2f}")
print("Distribuição da Regra:")
print(best_rule[1])
