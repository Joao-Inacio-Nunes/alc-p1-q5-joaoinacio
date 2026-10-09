import numpy as np

def resolve_lu(A, b):
    A = np.array(A, dtype=float)  # cópia para não modificar o original
    
    # Confere se a matriz é quadrada
    nrows, ncols = A.shape
    if nrows != ncols:
        raise ValueError("A matriz não é quadrada. Insira uma matriz quadrada.")
    else:
        n = nrows

    # Verifica se a dimensão de B está correta
    if np.array(b).shape not in ((nrows,), (nrows, 1)):
        raise ValueError("O vetor de termos independentes deve ter o mesmo num. de linhas que a matriz.")

    # Garante que b seja 1D para que b[i] seja escalar
    b = np.array(b)
    if b.ndim > 1:
        b = b[:, 0]

    L = np.eye(n)
    U = np.zeros((n, n))

    ### Laço de cálculo de L e U
    for i in range(n):
        # Verifica pivô nulo
        if A[i, i] == 0:
            raise Exception("Pivô nulo encontrado. Utilize uma função alternativa.")

        # Preenche a i-ésima linha em U, que não será alterada
        U[i, i:] = A[i, i:]

        for k in range(i + 1, n):
            # Calcula o multiplicador e guarda em L
            L[k, i] = A[k, i]/A[i, i]

            # Elimina o elemento abaixo do pivô utilizando o multiplicador
            A[k, i:] -= A[i, i:]*L[k, i]

    y = np.zeros(n)

    ### Resolver Ly = b -> substituição progressiva
    for i in range(n):
        y_sum = 0
        for j in range (0, i):
            # Soma, na linha i, os valores das colunas anteriores a L[i,i]
            y_sum += L[i, j] * y[j]
        
        y[i] = b[i] - y_sum

    x = np.zeros(n)

    ### Resolver Ux = y -> substituição regressiva
    for i in range(n-1, -1, -1):
        x_sum = 0
        for j in range(n-1, i, -1):
            # Soma, na linha i, os valores das colunas posteriores a U[i,i]
            x_sum += U[i, j] * x[j]

        x[i] = (y[i] - x_sum)/U[i,i]

    return L, U, x

# Teste:
A = np.array([
    [ 2, 1, 1],
    [ 4, 1, 0],
    [-2, 2, 1]
])
b = np.array([4, 5, 1])

L, U, x = resolve_lu(A, b)

print(f"L =\n{L}")
print(f"U =\n{U}")
print(f"x = {x}")
print(f"Verificação: A @ x ={A @ x}, esperado: b = {b}")