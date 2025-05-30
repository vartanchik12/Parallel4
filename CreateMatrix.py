import numpy as np

rowsA, colsA = 1000, 1000
matrixA = np.random.randint(0, 10, size=(rowsA, colsA))

rowsB, colsB = 1000, 1000
matrixB = np.random.randint(0, 10, size=(rowsB, colsB))

if colsA != rowsB:
    print("columns of matrix A is not equal rows of Matrix B, you can't multiply")
    exit()

def save_matrix(filename, matrix):
    with open(filename, "w") as f:
        f.write(f"{matrix.shape[0]} {matrix.shape[1]}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

save_matrix("matrixA.txt", matrixA)
save_matrix("matrixB.txt", matrixB)