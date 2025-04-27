import numpy as np
import math
import matplotlib.pyplot as plt
import os
os.system("cls|clear")


# ===============================
# 1. DADOS DO EXPERIMENTO (resposta Y)
# ===============================
# Experimentos realizados com duplicata nas 4 combinações de A e B (-1 e 1)
# Ordem: (-1,-1), (-1,1), (1,-1), (1,1), com duas repetições
Y = np.array([8, 42, 32, 95, 14, 48, 39, 98])  # Substitua por seus próprios dados

# ===============================
# 2. TABELA DOS NÍVEIS DOS FATORES
# ===============================
# Fator A (nível baixo: -1, alto: 1)
A = np.array([-1, -1, 1, 1, -1, -1, 1, 1])

# Fator B
B = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

# Interação A*B
AB = A * B

# Mostrando a tabela dos níveis dos fatores
print("\nTabela dos níveis dos fatores (com duplicata):")
print("Exec\tA\tB\tA*B\tY")
for i in range(len(Y)):
    print(f"{i+1}\t{A[i]}\t{B[i]}\t{AB[i]}\t{Y[i]}")

# ===============================
# 3. MATRIZ DOS EFEITOS (Design Matrix)
# ===============================
# Matriz X com coluna de 1s (intercepto)
X = np.column_stack((np.ones(8), A, B, AB))

# Mostrando a matriz dos efeitos
print("\nMatriz dos efeitos (X):")
print("       I    A    B   AB")
for linha in X:
    print("   ", linha)

# ===============================
# 4. ESTIMAÇÃO DOS COEFICIENTES VIA MÍNIMOS QUADRADOS
# ===============================
XTX_inv = np.linalg.inv(X.T @ X)
beta = XTX_inv @ X.T @ Y

# ===============================
# 5. RESÍDUOS, SOMA DE QUADRADOS E ERRO
# ===============================
Y_pred = X @ beta
residuos = Y - Y_pred
SQres = np.sum(residuos**2)

n = len(Y)        # número de observações
p = len(beta)     # número de coeficientes
gl = n - p        # graus de liberdade

MQres = SQres / gl  # erro médio quadrático

# ===============================
# 6. ERRO PADRÃO, TESTE T E P-VALORES (sem scipy)
# ===============================
erro_padrao = np.sqrt(np.diag(MQres * XTX_inv))
t_valores = beta / erro_padrao

# Cálculo aproximado do p-valor manualmente
def student_t_cdf(t, df):
    x = df / (df + t**2)
    a = df / 2.0
    b = 0.5

    def beta_inc(x, a, b, n_terms=100):
        total = 0.0
        for i in range(1, n_terms + 1):
            xi = x * i / n_terms
            total += xi**(a - 1) * (1 - xi)**(b - 1)
        total *= x / n_terms
        return total / beta(a, b)

    def beta(a, b):
        return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    return 1 - 0.5 * beta_inc(x, a, b)

p_valores = [2 * (1 - student_t_cdf(abs(t), gl)) for t in t_valores]

# ===============================
# 7. RESULTADOS NUMÉRICOS
# ===============================
efeitos = ['Geral', 'RAM', 'CPU', 'RAMxCPU']

print("\nResultados da regressão e teste t:")
print("Efeito     | Coef.  |   t     | p-valor")
print("--------------------------------------------")
for i in range(len(beta)):
    print(f"{efeitos[i]:<11} | {beta[i]:>6.3f} | {t_valores[i]:>7.3f} | {p_valores[i]:>8.4f}")

# ===============================
# 8. GRÁFICO DE PARETO DOS EFEITOS
# ===============================
# Não inclui o intercepto no gráfico
efeitos_plot = efeitos[1:]
valores_plot = np.abs(beta[1:])

# Ordenar para Pareto
ordem = np.argsort(valores_plot)[::-1]
efeitos_ordenados = [efeitos_plot[i] for i in ordem]
valores_ordenados = valores_plot[ordem]

# Gráfico
plt.figure(figsize=(8, 5))
plt.bar(efeitos_ordenados, valores_ordenados)
plt.title('Gráfico de Pareto dos Efeitos (2² com duplicata)')
plt.ylabel('Valor Absoluto dos Efeitos')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""
Interpretação (Exemplo):
    -Fator A e B são significativos (p < 0.05): eles afetam a resposta.

    - Interação A:B não é significativa (p > 0.05): não há efeito combinado notável entre A e B.

    - O gráfico de Pareto reforça visualmente quais efeitos têm maior influência.
"""



